"""Read loss weights from files keyed by genomic coordinates.

Two inputs, both optional:

1. Bin weights via region classes.
   - ``bin_classes``: BED, no header, columns ``chrom start end class`` (0-based,
     half-open, hg38, non-overlapping within a chromosome). A 32 bp output bin gets
     the class of the interval containing the bin's midpoint; bins outside every
     interval get ``default_class`` (an error if that is unset).
   - ``class_weights``: TSV with header ``class weight`` (one weight per class, shared by
     all tracks), or ``class <group> <group> ...`` (one weight per class and track group,
     e.g. per assay). Every class used in the BED (and ``default_class``) must be
     listed. Row order defines the integer class id.
   - Track groups: each track's value in the targets table column ``group_by`` (default
     ``assay``), optionally renamed by ``group_map`` (e.g. ``{"rna3": "rna"}``); every
     group must have a column in a per-group table.
   The dataset returns integer class ids per bin; the trainer looks weights up in a
   (num_classes, num_groups) tensor (num_groups = 1 for a shared table), so an online
   method can change that tensor in place.

2. Sample (window) weights.
   - ``sample_weights``: TSV with header ``chrom start end weight``, keyed by the
     window coordinates in the dataset's ``sequences.bed``. Windows not listed get 1.

Weights are rescaled to mean 1 over the training set (class weights by bin counts
over all training windows, overlaps counted as trained, separately within each track
group, so each group keeps its share of the loss; sample weights over training
windows), because the loss divides by fixed constants, never per batch. Sample and
class weights are rescaled separately, not their product.
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


@dataclasses.dataclass
class RegionClassMap:
    """Interval -> class id lookup for output bins."""

    class_names: list[str]
    class_weights: np.ndarray  # (num_classes, num_groups) raw weights from the TSV
    group_names: list[str | None]  # [None] for a shared table
    default_class: int | None
    # per chromosome: sorted starts, ends, class ids
    intervals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]

    @classmethod
    def from_files(
        cls,
        bin_classes: str | Path,
        class_weights: str | Path,
        default_class: str | None = None,
    ) -> RegionClassMap:
        with open(class_weights) as f:
            header = f.readline().rstrip("\n").split("\t")
        if len(set(header)) != len(header):
            raise ValueError(f"{class_weights}: duplicate columns in {header}")
        table = pd.read_csv(class_weights, sep="\t", dtype={"class": str})
        if table.columns[0] != "class" or len(table.columns) < 2:
            raise ValueError(
                f"{class_weights}: expected header 'class<TAB>weight' or "
                "'class<TAB><group>...'"
            )
        if table["class"].duplicated().any():
            raise ValueError(f"{class_weights}: duplicate classes")
        if list(table.columns) == ["class", "weight"]:
            group_names = [None]
        else:
            group_names = [str(g) for g in table.columns[1:]]
            if "weight" in group_names:
                raise ValueError(f"{class_weights}: 'weight' can't be a group column")
        weights = table.iloc[:, 1:].to_numpy(dtype=np.float64)
        if (weights < 0).any() or not np.isfinite(weights).all():
            raise ValueError(f"{class_weights}: weights must be finite and >= 0")
        names = table["class"].tolist()
        index = {n: i for i, n in enumerate(names)}

        intervals = read_class_intervals(bin_classes, index, class_weights)

        default = None
        if default_class is not None:
            if default_class not in index:
                raise ValueError(
                    f"default_class {default_class!r} not in {class_weights}"
                )
            default = index[default_class]
        return cls(names, weights, group_names, default, intervals)

    @classmethod
    def unit_map(cls, bin_classes: str | Path, unit_classes) -> RegionClassMap:
        """A map labelling bins 1..n for ``unit_classes`` and 0 ("other") elsewhere.

        Only the rows of those classes are read, so a genome-wide class BED stays small.
        """
        names = ["other", *unit_classes]
        index = {n: i for i, n in enumerate(names)}
        intervals = read_class_intervals(bin_classes, index, keep=set(unit_classes))
        if not intervals:
            raise ValueError(
                f"{bin_classes} has no intervals of classes {unit_classes}"
            )
        return cls(names, np.ones((len(names), 1)), [None], 0, intervals)

    @property
    def num_groups(self) -> int:
        return len(self.group_names)

    @property
    def shared(self) -> bool:
        """A plain ``class weight`` table: one weight per class for every track."""
        return self.group_names == [None]

    def track_groups(
        self,
        targets_df: pd.DataFrame,
        group_by: str = "assay",
        group_map=None,
        strand_pair=None,
    ) -> np.ndarray:
        """Group index (into ``group_names``) of each track; all 0 for a shared table.

        A table with group columns, even a single one, must name every track's group;
        strand partners (``strand_pair``, positional) must share a group.
        """
        if self.shared:
            return np.zeros(len(targets_df), dtype=np.int64)
        if group_by not in targets_df.columns:
            raise ValueError(
                f"targets table has no {group_by!r} column for track groups"
            )
        groups = targets_df[group_by].astype(str)
        if group_map:
            groups = groups.replace(group_map)
        index = {g: i for i, g in enumerate(self.group_names)}
        missing = sorted(set(groups) - set(index))
        if missing:
            raise ValueError(f"track groups without a class-weight column: {missing}")
        unused = sorted(set(index) - set(groups))
        if unused:
            print(f"warning: class-weight columns with no tracks: {unused}", flush=True)
        group_index = groups.map(index).to_numpy(dtype=np.int64)
        if strand_pair is not None:
            split = np.flatnonzero(group_index != group_index[np.asarray(strand_pair)])
            if len(split):
                raise ValueError(
                    f"strand partners in different groups: tracks {split[:10].tolist()}"
                )
        return group_index

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def label_bins(
        self, chrom: str, start: int, num_bins: int, bin_width: int
    ) -> np.ndarray:
        """Class id (int16) of each bin ``[start + j*w, start + (j+1)*w)`` by its midpoint."""
        mids = start + bin_width * np.arange(num_bins, dtype=np.int64) + bin_width // 2
        labels = np.full(num_bins, -1, dtype=np.int16)
        if chrom in self.intervals:
            starts, ends, ids = self.intervals[chrom]
            i = np.searchsorted(starts, mids, side="right") - 1
            inside = (i >= 0) & (mids < ends[np.clip(i, 0, None)])
            labels[inside] = ids[i[inside]]
        uncovered = labels < 0
        if uncovered.any():
            if self.default_class is None:
                raise ValueError(
                    f"{uncovered.sum()} bins of {chrom}:{start} have no class and no default_class is set"
                )
            labels[uncovered] = self.default_class
        return labels


def read_class_intervals(
    bin_classes, index: dict, weights_file=None, keep=None
) -> dict:
    """Class BED -> {chrom: (sorted starts, ends, class ids)}; ``keep`` limits classes."""
    bed = pd.read_csv(
        bin_classes,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "class"],
        dtype={"chrom": str, "class": str},
        comment="#",
    )
    if keep is not None:
        bed = bed[bed["class"].isin(keep)]
    missing = set(bed["class"]) - set(index)
    if missing:
        raise ValueError(
            f"classes in {bin_classes} missing from {weights_file}: {sorted(missing)[:10]}"
        )
    intervals = {}
    for chrom, d in bed.groupby("chrom", sort=False):
        d = d.sort_values("start")
        starts = d["start"].to_numpy(np.int64)
        ends = d["end"].to_numpy(np.int64)
        if (ends <= starts).any():
            raise ValueError(f"{bin_classes}: empty or negative interval on {chrom}")
        if (starts[1:] < ends[:-1]).any():
            raise ValueError(f"{bin_classes}: overlapping intervals on {chrom}")
        intervals[chrom] = (starts, ends, d["class"].map(index).to_numpy(np.int16))
    return intervals


def normalize_class_weights(weights: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    """Rescale class weights so the mean weight over training bins is 1 (per column,
    i.e. within each track group, for a (num_classes, num_groups) table)."""
    total = bin_counts.sum()
    means = (weights * bin_counts[:, None]).sum(axis=0) / max(total, 1)
    if (means <= 0).any():
        raise ValueError("class weights are zero on every training bin (in some group)")
    return weights / means


def read_sample_weights(path: str | Path, windows: pd.DataFrame) -> np.ndarray:
    """Weight per row of ``windows`` (columns chrom, start, end); unlisted windows get 1."""
    table = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    if list(table.columns[:4]) != ["chrom", "start", "end", "weight"]:
        raise ValueError(f"{path}: expected header 'chrom start end weight'")
    if table.duplicated(["chrom", "start", "end"]).any():
        raise ValueError(f"{path}: duplicate windows")
    w = table["weight"].to_numpy(np.float64)
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError(f"{path}: weights must be finite and >= 0")
    keys = set(zip(windows["chrom"], windows["start"], windows["end"]))
    unmatched = sum(
        k not in keys for k in zip(table["chrom"], table["start"], table["end"])
    )
    if unmatched:
        raise ValueError(f"{path}: {unmatched} windows not in sequences.bed")
    merged = windows[["chrom", "start", "end"]].merge(
        table, how="left", on=["chrom", "start", "end"]
    )
    return merged["weight"].fillna(1.0).to_numpy(np.float64)


def normalize_sample_weights(weights: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    """Rescale so the mean over training windows is 1."""
    mean = float(weights[train_rows].mean())
    if mean <= 0:
        raise ValueError("sample weights are zero on every training window")
    return weights / mean


def class_bin_counts(ds) -> np.ndarray:
    """Output bins per class over every example of a ``SeqDataset`` with a class_map."""
    counts = np.zeros(ds.class_map.num_classes, dtype=np.int64)
    for i in range(len(ds)):
        counts += np.bincount(ds.bin_labels(i), minlength=ds.class_map.num_classes)
    return counts


def split_of_rows(ds) -> np.ndarray:
    """ "train", "valid", "test" (or "held_out" outside folds mode) per sequences.bed row."""
    split = np.full(len(ds.seqs_bed), "held_out", dtype=object)
    labels = ds.seqs_bed["label"].to_numpy()
    if ds.fold is not None:
        from baskerville.dataset import compute_fold_splits  # noqa: PLC0415

        num_folds = len({lab for lab in labels if str(lab).startswith("fold")})
        folds = compute_fold_splits(num_folds, ds.fold, ds.cross)
        split[labels == f"fold{folds['valid']}"] = "valid"
        split[labels == f"fold{folds['test']}"] = "test"
    split[ds.seq_indexes] = "train"
    return split


def class_bin_counts_by_split(ds) -> dict[str, np.ndarray]:
    """Output bins per class over every sequences.bed window, by split.

    ``["train"]`` equals ``class_bin_counts(ds)`` (the normalization counts); valid and
    test windows are only counted, to catch labelling errors.
    """
    cm = ds.class_map
    split = split_of_rows(ds)
    counts = {s: np.zeros(cm.num_classes, dtype=np.int64) for s in np.unique(split)}
    for row, s in zip(ds.seqs_bed.itertuples(index=False), split):
        labels = cm.label_bins(
            row.chrom, int(row.start) + ds.bin_offset, ds.target_length, ds.pool_width
        )
        counts[s] += np.bincount(labels, minlength=cm.num_classes)
    counts.setdefault("train", np.zeros(cm.num_classes, dtype=np.int64))
    return counts


def check_class_coverage(ds, counts_by_split: dict[str, np.ndarray]) -> list[str]:
    """Raise on labelling errors; return, and warn about, held-out-only classes.

    - Raise if no chromosome name is shared between sequences.bed and the class BED.
    - Raise if a class with weight > 0 has no bins in any sequences.bed window (a class
      missing from the BED, misnamed chromosomes, intervals that never hold a 32 bp
      bin midpoint).
    - Warn if such a class has bins only in valid/test windows (allowed: a class can
      sit on held-out chromosomes), about BED chromosomes absent from sequences.bed,
      and about intervals shorter than a bin.
    """
    cm = ds.class_map
    seq_chroms = set(ds.seqs_bed["chrom"])
    bed_chroms = set(cm.intervals)
    if not seq_chroms & bed_chroms:
        raise ValueError(
            "no sequences.bed chromosome is in the region-class BED "
            f"(sequences: {sorted(seq_chroms)[:5]}, BED: {sorted(bed_chroms)[:5]}); "
            "check chromosome names"
        )
    absent = sorted(bed_chroms - seq_chroms)
    if absent:
        print(f"warning: class BED chromosomes not in sequences.bed: {absent[:20]}")
    short = sum(
        int(((e - s) < ds.pool_width).sum()) for s, e, _ in cm.intervals.values()
    )
    if short:
        print(
            f"warning: {short} class intervals are shorter than a {ds.pool_width} bp bin"
        )

    weighted = cm.class_weights.max(axis=1) > 0
    total = sum(counts_by_split.values())
    train = counts_by_split["train"]
    missing = [n for n, w, c in zip(cm.class_names, weighted, total) if w and c == 0]
    if missing:
        raise ValueError(
            f"classes with weight > 0 but no bins in any sequences.bed window: {missing}"
        )
    held_out = [
        n
        for n, w, c, t in zip(cm.class_names, weighted, total, train)
        if w and c and not t
    ]
    if held_out:
        print(
            f"warning: classes with weight > 0 only in valid/test windows: {held_out}"
        )
    return held_out


def write_class_weights(
    path, class_map: RegionClassMap, weights, counts_by_split
) -> None:
    """Raw and normalized weights per class (columns per group for a per-group table)
    and bin counts per split, so partial labelling losses are visible."""
    table = {"class": class_map.class_names}
    for gi, g in enumerate(class_map.group_names):
        suffix = "" if g is None else f"_{g}"
        table[f"weight_raw{suffix}"] = class_map.class_weights[:, gi]
        table[f"weight{suffix}"] = weights[:, gi]
    for split in ("train", "valid", "test", "held_out"):
        if split in counts_by_split:
            table[f"{split}_bins"] = counts_by_split[split]
    pd.DataFrame(table).to_csv(path, sep="\t", index=False)


@dataclasses.dataclass
class StepInfo:
    """What an online-reweighting hook's ``after_step(trainer, step)`` receives.

    Hooks may also define ``on_train_start(trainer)`` and ``on_epoch_end(trainer,
    epoch)``, and change weights by editing ``trainer.class_weights[di]`` or
    ``trainer.sample_weights[di]`` in place.
    """

    epoch: int
    step: int
    di: int
    seq_index: object  # (B,) tensor or None
    bin_labels: object  # (B, L) long tensor on the device, or None
    y_pred: object  # (B, T, L), detached
    y_true: object  # (B, T, L)
    loss: float


def load_hook(spec: dict | None):
    """``{"class": "module:Name", "args": {...}}`` -> hook instance, or None."""
    if not spec:
        return None
    module, _, name = spec["class"].partition(":")
    return getattr(importlib.import_module(module), name)(**spec.get("args", {}))


def project_weights(weights, bin_counts, lo: float, hi: float, iters: int = 60):
    """Map a (num_classes, num_groups) table onto {lo <= w <= hi, mean 1 over training
    bins within each group}.

    Per group, finds the factor s with mean(clip(s * w, lo, hi)) = 1 by bisection
    (the mean is monotone in s), so both conditions hold at once, unlike a single
    clip followed by rescaling. Zero (or negative) weights end at ``lo``. Raises when
    a group can't reach mean 1 within the bounds (lo <= 1 <= hi is necessary; with
    zero weights, hi on the positive classes must suffice). Works on numpy arrays and
    torch tensors (no gradient through the result).
    """
    if not lo <= 1.0 <= hi:
        raise ValueError("project_weights needs lo <= 1 <= hi")
    is_torch = hasattr(weights, "detach")
    w = (
        weights.detach().cpu().double().numpy()
        if is_torch
        else np.asarray(weights, float)
    )
    counts = np.asarray(bin_counts, dtype=float)
    p = counts / counts.sum()
    out = np.empty_like(w)
    for g in range(w.shape[1]):
        col = np.maximum(w[:, g], 0.0)
        positive = (col > 0) & (p > 0)
        if not positive.any():
            raise ValueError(
                f"project_weights: group {g} has no positive weight on training bins"
            )
        top = float((np.where(positive, hi, lo) * p).sum())  # the mean as s -> infinity
        if top < 1.0 - 1e-12:
            raise ValueError(
                f"project_weights: group {g} can't reach mean 1 within [{lo}, {hi}] "
                f"(at most {top:.4g} with its zero weights at lo)"
            )

        def mean_at(s, col=col):
            return float((np.clip(s * col, lo, hi) * p).sum())

        a, b = 0.0, 1.0
        while mean_at(b) < 1.0:
            b *= 2.0
            if b > 1e300:
                raise ValueError(f"project_weights: no scale found for group {g}")
        for _ in range(iters):
            mid = (a + b) / 2
            a, b = (mid, b) if mean_at(mid) < 1.0 else (a, mid)
        out[:, g] = np.clip(b * col, lo, hi)
        out[:, g] /= (out[:, g] * p).sum()  # remove the bisection residue (~1e-18)
        out[:, g] = np.clip(out[:, g], lo, hi)
    if is_torch:
        import torch  # noqa: PLC0415

        return torch.tensor(out, dtype=weights.dtype, device=weights.device)
    return out


def rows_in_bed(ds, bed, invert: bool = False) -> np.ndarray:
    """Example indexes of ``ds`` whose sequences.bed row (chrom, start, end) is listed
    in ``bed`` (or, with ``invert``, is not)."""
    table = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 1, 2], comment="#")
    keys = set(zip(table[0].astype(str), table[1].astype(int), table[2].astype(int)))
    rows = ds.seqs_bed.iloc[ds.seq_indexes]
    hit = np.array(
        [
            (c, int(s), int(e)) in keys
            for c, s, e in zip(rows["chrom"], rows["start"], rows["end"])
        ]
    )
    return np.flatnonzero(~hit if invert else hit)


def rows_on_chroms(ds, chroms, invert: bool = False) -> np.ndarray:
    """Example indexes of ``ds`` on (or, with ``invert``, off) the given chromosomes."""
    on = ds.seqs_bed["chrom"].iloc[ds.seq_indexes].isin(set(chroms)).to_numpy()
    return np.flatnonzero(~on if invert else on)
