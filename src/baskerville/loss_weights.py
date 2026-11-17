"""Read loss weights from files keyed by genomic coordinates.

Two inputs, both optional:

1. Bin weights via region classes.
   - ``bin_classes``: BED, no header, columns ``chrom start end class`` (0-based,
     half-open, hg38, non-overlapping within a chromosome). A 32 bp output bin gets
     the class of the interval containing the bin's midpoint; bins outside every
     interval get ``default_class`` (an error if that is unset).
   - ``class_weights``: TSV with header ``class weight``. Every class used in the BED
     (and ``default_class``) must be listed. Row order defines the integer class id.
   The dataset returns integer class ids per bin; the trainer looks weights up in a
   (num_classes,) tensor, so an online method can change that tensor in place.

2. Sample (window) weights.
   - ``sample_weights``: TSV with header ``chrom start end weight``, keyed by the
     window coordinates in the dataset's ``sequences.bed``. Windows not listed get 1.

Weights are rescaled to mean 1 over the training set (class weights by bin counts
over all training windows, overlaps counted as trained; sample weights over
training windows), because the loss divides by fixed constants, never per batch.
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
    class_weights: np.ndarray  # (num_classes,) raw weights from the TSV
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
        table = pd.read_csv(class_weights, sep="\t", dtype={"class": str})
        if list(table.columns[:2]) != ["class", "weight"]:
            raise ValueError(f"{class_weights}: expected header 'class<TAB>weight'")
        if table["class"].duplicated().any():
            raise ValueError(f"{class_weights}: duplicate classes")
        weights = table["weight"].to_numpy(dtype=np.float64)
        if (weights < 0).any() or not np.isfinite(weights).all():
            raise ValueError(f"{class_weights}: weights must be finite and >= 0")
        names = table["class"].tolist()
        index = {n: i for i, n in enumerate(names)}

        bed = pd.read_csv(
            bin_classes,
            sep="\t",
            header=None,
            usecols=[0, 1, 2, 3],
            names=["chrom", "start", "end", "class"],
            dtype={"chrom": str, "class": str},
            comment="#",
        )
        missing = set(bed["class"]) - set(index)
        if missing:
            raise ValueError(
                f"classes in {bin_classes} missing from {class_weights}: {sorted(missing)[:10]}"
            )
        intervals = {}
        for chrom, d in bed.groupby("chrom", sort=False):
            d = d.sort_values("start")
            starts = d["start"].to_numpy(np.int64)
            ends = d["end"].to_numpy(np.int64)
            if (ends <= starts).any():
                raise ValueError(
                    f"{bin_classes}: empty or negative interval on {chrom}"
                )
            if (starts[1:] < ends[:-1]).any():
                raise ValueError(f"{bin_classes}: overlapping intervals on {chrom}")
            ids = d["class"].map(index).to_numpy(np.int16)
            intervals[chrom] = (starts, ends, ids)

        default = None
        if default_class is not None:
            if default_class not in index:
                raise ValueError(
                    f"default_class {default_class!r} not in {class_weights}"
                )
            default = index[default_class]
        return cls(names, weights, default, intervals)

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


def normalize_class_weights(weights: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    """Rescale class weights so the mean weight over training bins is 1."""
    total = bin_counts.sum()
    mean = float((weights * bin_counts).sum() / total) if total else 0.0
    if mean <= 0:
        raise ValueError("class weights are zero on every training bin")
    return weights / mean


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


def check_class_coverage(ds, counts) -> list[str]:
    """Fail if no training sequence lies on a chromosome of the class BED (e.g. "chr1"
    vs "1"); return, and warn about, nonzero-weight classes with no training bins.

    Empty classes are allowed: a class can sit only on held-out chromosomes.
    """
    chroms = set(ds.seqs_bed["chrom"].iloc[ds.seq_indexes])
    bed_chroms = set(ds.class_map.intervals)
    if not chroms & bed_chroms:
        raise ValueError(
            "no training sequence is on a chromosome of the region-class BED "
            f"(training: {sorted(chroms)[:5]}, BED: {sorted(bed_chroms)[:5]}); "
            "check chromosome names"
        )
    empty = [
        name
        for name, w, n in zip(
            ds.class_map.class_names, ds.class_map.class_weights, counts
        )
        if w > 0 and n == 0
    ]
    if empty:
        print(
            f"warning: classes with weight > 0 but no training bins: {empty}",
            flush=True,
        )
    return empty


def write_class_weights(path, class_map: RegionClassMap, weights, counts) -> None:
    pd.DataFrame(
        {
            "class": class_map.class_names,
            "weight_raw": class_map.class_weights,
            "weight": weights,
            "train_bins": counts,
        }
    ).to_csv(path, sep="\t", index=False)


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
