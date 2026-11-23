"""Cell-type deviation loss: reward correct differences between cell types.

For one window and one assay (e.g. bulk RNA), with tracks t in cell-type groups g(t)
and units k (blocks of ``block_bins`` output bins, summing only the bins whose region
class is a unit class, e.g. gene exons/TSS):

    u_tk    = log(sum_{i in k} y_ti + c),  uhat_tk likewise from predictions
    U_gk    = mean of u_tk over the tracks of group g (each cell type counts once)
    d_gk    = U_gk - mean_g U_gk,  dhat_gk likewise
    L_dev   = sum_k w_k mean_g (dhat_gk - d_gk)^2 / D

w_k = min(1, E_k / threshold) * position weight of the block, with E_k the unit's
target sum averaged over the assay's tracks (targets only, no gradient). D is a fixed
denominator (the average per-window sum of w_k over training windows, times the
batch size), so every unit counts the same in every batch. With several assays,
L_dev is the mean of the per-assay terms. The trainer adds ``lambda * L_dev`` to the
coverage loss; validation and early stopping are unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch


@dataclasses.dataclass
class AssayGroups:
    """Tracks of one assay and their cell-type group membership."""

    assay: str
    tracks: torch.Tensor  # (T_a,) positions in the model's target order
    group_means: torch.Tensor  # (T_a, G) averaging matrix, columns sum to 1
    group_names: list[str]


def assay_groups(
    targets_df: pd.DataFrame,
    cell_types: pd.DataFrame,
    assays,
    strand_pair=None,
    group_column: str = "cell_type_group",
) -> list[AssayGroups]:
    """Group each assay's tracks by cell type (``cell_types`` keyed by identifier)."""
    groups_of = dict(zip(cell_types["identifier"], cell_types[group_column]))
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
        labels = np.array([groups_of[i] for i in ids])
        if strand_pair is not None:
            sp = np.asarray(strand_pair)
            group_of_track = dict(zip(tracks, labels))
            for t in tracks:
                if (
                    sp[t] in group_of_track
                    and group_of_track[sp[t]] != group_of_track[t]
                ):
                    raise ValueError(
                        f"deviation loss: strand partners {t}, {sp[t]} differ"
                    )
        names = sorted(set(labels))
        if len(names) < 2:
            raise ValueError(f"deviation loss: {assay!r} needs >= 2 cell-type groups")
        member = (labels[:, None] == np.array(names)[None, :]).astype(np.float64)
        out.append(
            AssayGroups(
                assay,
                torch.tensor(tracks),
                torch.tensor(member / member.sum(axis=0), dtype=torch.float32),
                names,
            )
        )
    return out


def per_assay(value, n: int, name: str) -> list[float]:
    """A scalar or one value per assay -> a list of n positive floats."""
    values = [float(value)] * n if np.isscalar(value) else [float(v) for v in value]
    if len(values) != n or min(values) <= 0:
        raise ValueError(f"deviation loss: need one positive {name} per assay")
    return values


def deviation_from_logs(u, uhat, group_means, weights):
    """sum_k w_k mean_g (dhat_gk - d_gk)^2 for log-unit values (B, T_a, K)."""
    U = torch.einsum("btk,tg->bgk", u, group_means)
    Uhat = torch.einsum("btk,tg->bgk", uhat, group_means)
    d = U - U.mean(dim=1, keepdim=True)
    dhat = Uhat - Uhat.mean(dim=1, keepdim=True)
    err = (dhat - d).pow(2).mean(dim=1)  # B x K
    return (weights * err).sum()


class DeviationLoss(torch.nn.Module):
    """L_dev over a batch; see the module docstring.

    Args:
      groups: per-assay track groups (``assay_groups``).
      position_weights: (L,) centre-vs-edge weights of the main loss.
      block_bins: unit size in bins.
      pseudocount, threshold: c and the w_k threshold, per assay (a scalar applies to
        every assay); target scales differ a lot between assays.
      denominators: per assay, the fixed average per-window sum of w_k on training
        windows (a scalar applies to every assay).
    """

    def __init__(
        self,
        groups: list[AssayGroups],
        position_weights: torch.Tensor,
        block_bins: int = 4,
        pseudocount: float = 0.1,
        threshold: float = 1.0,
        denominators=1.0,
    ):
        super().__init__()
        if position_weights.numel() % block_bins:
            raise ValueError(
                "deviation loss: target length must be a multiple of block_bins"
            )
        self.groups = groups
        self.block_bins = block_bins
        self.pseudocounts = per_assay(pseudocount, len(groups), "pseudocount")
        self.thresholds = per_assay(threshold, len(groups), "threshold")
        self.denominators = per_assay(denominators, len(groups), "denominator")
        block_pw = position_weights.reshape(-1, block_bins).mean(dim=1)
        self.register_buffer("block_position_weights", block_pw.float())
        for i, g in enumerate(groups):
            self.register_buffer(f"tracks_{i}", g.tracks)
            self.register_buffer(f"group_means_{i}", g.group_means)

    def block_sums(self, y, unit_mask):
        """(B, T, L) values, (B, L) unit mask -> (B, T, K) unit sums, (B, K) has-unit."""
        B, T, L = y.shape
        m = unit_mask.to(y.dtype)
        sums = (y * m[:, None, :]).reshape(B, T, L // self.block_bins, self.block_bins)
        has_unit = m.reshape(B, -1, self.block_bins).sum(dim=-1) > 0
        return sums.sum(dim=-1), has_unit

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
            tracks = getattr(self, f"tracks_{i}")
            s_true, has_unit = self.block_sums(y_true[:, tracks], unit_mask)
            s_pred, _ = self.block_sums(y_pred[:, tracks], unit_mask)
            w = self.unit_weights(s_true, has_unit, i)
            c = self.pseudocounts[i]
            u = torch.log(s_true + c)
            uhat = torch.log(s_pred + c)
            dev = deviation_from_logs(u, uhat, getattr(self, f"group_means_{i}"), w)
            terms.append(dev / (self.denominators[i] * y_true.shape[0]))
        return torch.stack(terms).mean()

    def unit_weight_sums(self, y_true, unit_mask):
        """Per-window sum of w_k for each assay, (B, n_assays), for the denominator."""
        out = []
        for i in range(len(self.groups)):
            tracks = getattr(self, f"tracks_{i}")
            s_true, has_unit = self.block_sums(y_true.float()[:, tracks], unit_mask)
            out.append(self.unit_weights(s_true, has_unit, i).sum(dim=1))
        return torch.stack(out, dim=1)
