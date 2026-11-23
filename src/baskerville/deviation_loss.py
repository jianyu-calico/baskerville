"""Cell-type deviation loss: reward correct differences between cell types.

For one window and one assay (e.g. bulk RNA), with samples s (a track, or a strand
pair summed: + and - strand of one library) in cell-type groups g(s), and units k
(blocks of ``block_bins`` output bins, summing only the bins whose region class is a
unit class, e.g. gene exons/TSS):

    u_sk    = log(sum_{t in s} sum_{i in k} y_ti + c),  uhat_sk likewise
    U_gk    = mean of u_sk over the samples of group g (each cell type counts once)
    d_gk    = U_gk - mean_g U_gk,  dhat_gk likewise
    L_dev   = sum_k w_k mean_g (dhat_gk - d_gk)^2 / D

In ``space="untransformed"``, targets and predictions are first mapped back to
coverage per track (soft clip, sqrt and scale undone, as for the specificity metric:
``((y + 1)^2 - 1) / scale`` for ``sum_sqrt`` tracks; the default); ``"transformed"`` uses the
dataset's values as stored. w_k = min(1, E_k / threshold) * position weight of the
block, with E_k the unit's target sum averaged over the samples (targets only, no
gradient). D is a fixed denominator (the average per-window sum of w_k over training
windows, times the batch size), so every unit counts the same in every batch. With
several assays, L_dev is the mean of the per-assay terms. The trainer adds
``lambda * L_dev`` to the coverage loss; validation and early stopping are unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch

SPACES = ("transformed", "untransformed")


@dataclasses.dataclass
class AssayGroups:
    """Tracks of one assay, their samples (strand pairs merged) and cell types."""

    assay: str
    tracks: torch.Tensor  # (T_a,) positions in the model's target order
    sample_sums: torch.Tensor  # (T_a, S) 0/1: which sample each track adds to
    group_means: torch.Tensor  # (S, G) averaging matrix, columns sum to 1
    group_names: list[str]
    # per track, to undo the dataset transform (space="untransformed")
    scale: torch.Tensor  # (T_a,)
    clip_soft: torch.Tensor  # (T_a,), inf where unset
    sqrt: torch.Tensor  # (T_a,) bool, sum_stat is *_sqrt
    unsupported_sum_stats: list = dataclasses.field(default_factory=list)


def assay_groups(
    targets_df: pd.DataFrame,
    cell_types: pd.DataFrame,
    assays,
    strand_pair=None,
    group_column: str = "cell_type_group",
) -> list[AssayGroups]:
    """Group each assay's samples by cell type (``cell_types`` keyed by identifier).

    Strand partners (``strand_pair``, positional) form one sample and must share a
    cell type; their unit sums are added before the log.
    """
    groups_of = dict(zip(cell_types["identifier"], cell_types[group_column]))
    sp = np.arange(len(targets_df)) if strand_pair is None else np.asarray(strand_pair)
    out = []
    for assay in assays:
        tracks = np.flatnonzero(targets_df["assay"].to_numpy() == assay)
        if len(tracks) == 0:
            raise ValueError(
                f"deviation loss: no {assay!r} tracks in the targets table"
            )
        ids = targets_df["identifier"].to_numpy()[tracks]
        missing = [i for i in ids if i not in groups_of or pd.isna(groups_of[i])]
        if missing:
            raise ValueError(
                f"deviation loss: tracks without a cell type: {missing[:10]}"
            )
        group_of_track = {t: groups_of[i] for t, i in zip(tracks, ids)}

        # sample of each track: the lower index of its strand pair (within the assay)
        sample_of = {}
        for t in tracks:
            partner = sp[t]
            if partner != t and partner in group_of_track:
                if group_of_track[partner] != group_of_track[t]:
                    raise ValueError(
                        f"deviation loss: strand partners {t}, {partner} differ"
                    )
                sample_of[t] = min(t, partner)
            else:
                sample_of[t] = t
        samples = sorted(set(sample_of.values()))
        col = {s: j for j, s in enumerate(samples)}
        sample_sums = np.zeros((len(tracks), len(samples)))
        for row, t in enumerate(tracks):
            sample_sums[row, col[sample_of[t]]] = 1.0

        labels = np.array([group_of_track[s] for s in samples])
        names = sorted(set(labels))
        if len(names) < 2:
            raise ValueError(f"deviation loss: {assay!r} needs >= 2 cell-type groups")
        member = (labels[:, None] == np.array(names)[None, :]).astype(np.float64)

        rows = targets_df.iloc[tracks]
        scale = (
            rows["scale"].to_numpy(np.float64)
            if "scale" in rows
            else np.ones(len(tracks))
        )
        clip_soft = (
            rows["clip_soft"].fillna(np.inf).to_numpy(np.float64)
            if "clip_soft" in rows
            else np.full(len(tracks), np.inf)
        )
        sum_stat = (
            rows["sum_stat"].astype(str)
            if "sum_stat" in rows
            else pd.Series(["sum"] * len(tracks))
        )
        sqrt = sum_stat.str.endswith("_sqrt").to_numpy()
        other = sorted(set(sum_stat[~sqrt & ~sum_stat.isin(["sum", "mean", "avg"])]))
        out.append(
            AssayGroups(
                assay,
                torch.tensor(tracks),
                torch.tensor(sample_sums, dtype=torch.float32),
                torch.tensor(member / member.sum(axis=0), dtype=torch.float32),
                names,
                torch.tensor(scale, dtype=torch.float32),
                torch.tensor(clip_soft, dtype=torch.float32),
                torch.tensor(sqrt),
                other,
            )
        )
    return out


def untransform(y, scale, clip_soft, sqrt):
    """Dataset values (B, T, L) -> coverage: undo soft clip, sqrt and scale per track."""
    # soft clip: cs - 1 + (y - cs + 1)^2 above cs, i.e. y + e (e + 1) with e = relu(y - cs);
    # written without torch.where so an infinite cs (unset) gives no NaN gradient
    e = torch.relu(y - clip_soft[None, :, None])
    y = y + e * (e + 1)
    y = torch.where(sqrt[None, :, None], (y + 1) ** 2 - 1, y)
    return y / scale[None, :, None]


def per_assay(value, n: int, name: str) -> list[float]:
    """A scalar or one value per assay -> a list of n positive floats."""
    values = [float(value)] * n if np.isscalar(value) else [float(v) for v in value]
    if len(values) != n or min(values) <= 0:
        raise ValueError(f"deviation loss: need one positive {name} per assay")
    return values


def deviation_from_logs(u, uhat, group_means, weights):
    """sum_k w_k mean_g (dhat_gk - d_gk)^2 for log-unit values (B, S, K)."""
    U = torch.einsum("bsk,sg->bgk", u, group_means)
    Uhat = torch.einsum("bsk,sg->bgk", uhat, group_means)
    d = U - U.mean(dim=1, keepdim=True)
    dhat = Uhat - Uhat.mean(dim=1, keepdim=True)
    err = (dhat - d).pow(2).mean(dim=1)  # B x K
    return (weights * err).sum()


class DeviationLoss(torch.nn.Module):
    """L_dev over a batch; see the module docstring.

    Args:
      groups: per-assay samples and cell types (``assay_groups``).
      position_weights: (L,) centre-vs-edge weights of the main loss.
      block_bins: unit size in bins.
      pseudocount, threshold: c and the w_k threshold, per assay (a scalar applies to
        every assay); target scales differ a lot between assays and spaces.
      denominators: per assay, the fixed average per-window sum of w_k on training
        windows (a scalar applies to every assay).
      space: "untransformed" (coverage, default) or "transformed" (dataset values).
    """

    def __init__(
        self,
        groups: list[AssayGroups],
        position_weights: torch.Tensor,
        block_bins: int = 4,
        pseudocount=0.1,
        threshold=1.0,
        denominators=1.0,
        space: str = "untransformed",
    ):
        super().__init__()
        if position_weights.numel() % block_bins:
            raise ValueError(
                "deviation loss: target length must be a multiple of block_bins"
            )
        if space not in SPACES:
            raise ValueError(f"deviation loss: space must be one of {SPACES}")
        if space == "untransformed":
            bad = sorted({s for g in groups for s in g.unsupported_sum_stats})
            if bad:
                raise ValueError(f"deviation loss: can't untransform sum_stat {bad}")
        self.groups = groups
        self.block_bins = block_bins
        self.space = space
        self.pseudocounts = per_assay(pseudocount, len(groups), "pseudocount")
        self.thresholds = per_assay(threshold, len(groups), "threshold")
        self.denominators = per_assay(denominators, len(groups), "denominator")
        block_pw = position_weights.reshape(-1, block_bins).mean(dim=1)
        self.register_buffer("block_position_weights", block_pw.float())
        for i, g in enumerate(groups):
            for name in (
                "tracks",
                "sample_sums",
                "group_means",
                "scale",
                "clip_soft",
                "sqrt",
            ):
                self.register_buffer(f"{name}_{i}", getattr(g, name))

    def _buf(self, name, i):
        return getattr(self, f"{name}_{i}")

    def assay_values(self, y, i):
        """(B, T, L) model-order values -> (B, T_a, L) for assay i, in self.space."""
        ya = y[:, self._buf("tracks", i)]
        if self.space == "untransformed":
            ya = untransform(
                ya,
                self._buf("scale", i),
                self._buf("clip_soft", i),
                self._buf("sqrt", i),
            )
        return ya

    def block_sums(self, y, unit_mask):
        """(B, T, L) values, (B, L) unit mask -> (B, T, K) unit sums, (B, K) has-unit."""
        B, T, L = y.shape
        m = unit_mask.to(y.dtype)
        sums = (y * m[:, None, :]).reshape(B, T, L // self.block_bins, self.block_bins)
        has_unit = m.reshape(B, -1, self.block_bins).sum(dim=-1) > 0
        return sums.sum(dim=-1), has_unit

    def sample_unit_sums(self, y, unit_mask, i):
        """(B, T, L) model-order values -> (B, S, K) unit sums per sample, (B, K)."""
        sums, has_unit = self.block_sums(self.assay_values(y, i), unit_mask)
        return torch.einsum("btk,ts->bsk", sums, self._buf("sample_sums", i)), has_unit

    def unit_weights(self, target_sums, has_unit, i=0):
        """w_k from targets only: min(1, E_k / threshold_i) x block position weight."""
        expr = target_sums.mean(dim=1)
        w = (
            torch.clamp(expr / self.thresholds[i], max=1.0)
            * self.block_position_weights
        )
        return (w * has_unit).detach()

    def forward(self, y_pred, y_true, unit_mask):
        # float32 throughout: autocast would run the group means (einsum) in bf16
        with torch.autocast(device_type=y_true.device.type, enabled=False):
            return self._forward(y_pred.float(), y_true.float(), unit_mask)

    def _forward(self, y_pred, y_true, unit_mask):
        terms = []
        for i in range(len(self.groups)):
            s_true, has_unit = self.sample_unit_sums(y_true, unit_mask, i)
            s_pred, _ = self.sample_unit_sums(y_pred, unit_mask, i)
            w = self.unit_weights(s_true, has_unit, i)
            c = self.pseudocounts[i]
            u = torch.log(s_true + c)
            uhat = torch.log(s_pred.clamp(min=0) + c)
            dev = deviation_from_logs(u, uhat, self._buf("group_means", i), w)
            terms.append(dev / (self.denominators[i] * y_true.shape[0]))
        return torch.stack(terms).mean()

    def unit_weight_sums(self, y_true, unit_mask):
        """Per-window sum of w_k for each assay, (B, n_assays), for the denominator."""
        out = []
        for i in range(len(self.groups)):
            s_true, has_unit = self.sample_unit_sums(y_true.float(), unit_mask, i)
            out.append(self.unit_weights(s_true, has_unit, i).sum(dim=1))
        return torch.stack(out, dim=1)


def estimate_denominators(
    loss: DeviationLoss, dataset, n_windows: int = 512, seed: int = 0
):
    """Average per-window sum of w_k per assay over ``n_windows`` training windows.

    Sampling depends only on ``seed`` (not the training seed), so every training seed
    of one config gets the same denominators. Put the result in the config.
    """
    rng = np.random.default_rng([seed, 12345])
    idx = rng.choice(len(dataset), size=min(n_windows, len(dataset)), replace=False)
    device = loss.block_position_weights.device
    total = 0
    for i in np.sort(idx):
        item = (
            dataset[(int(i), 0, False)] if dataset.mode == "train" else dataset[int(i)]
        )
        y = item.coverage_targets.float()[None].to(device)
        mask = item.unit_mask[None].to(device)
        total = total + loss.unit_weight_sums(y, mask)[0].cpu()
    mean = total / len(idx)
    if (mean <= 0).any():
        raise ValueError("deviation loss: no weighted units in the sampled windows")
    return mean.tolist()
