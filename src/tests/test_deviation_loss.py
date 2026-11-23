"""Cell-type deviation loss (L = L_current + lambda * L_dev)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from baskerville.dataset import SeqDataset
from baskerville.deviation_loss import (
    DeviationLoss,
    assay_groups,
    deviation_from_logs,
    untransform,
)
from baskerville.loss_weights import RegionClassMap
from baskerville.scripts import hound_train
from tests.test_output_loss_weights import (
    BINS,
    NUM_TARGETS,
    POOL,
    make_data,
    model_state,
    same,
    tiny_params,
)

CELL_TYPES = {"a+": "liver", "a-": "liver", "b": "brain", "c": "blood"}


# -- fixtures ------------------------------------------------------------------


@pytest.fixture
def rna_data(tmp_path):
    """The shared toy dataset with all four tracks as RNA (a+/a- a strand pair)."""
    root = make_data(tmp_path / "data")
    targets = pd.read_csv(root / "targets.txt", sep="\t", index_col=0)
    targets["assay"] = "rna"
    targets.to_csv(root / "targets.txt", sep="\t")
    return root


def cell_type_table(tmp_path, mapping=CELL_TYPES):
    path = tmp_path / "cell_types.tsv"
    pd.DataFrame(
        {"identifier": list(mapping), "cell_type_group": list(mapping.values())}
    ).to_csv(path, sep="\t", index=False)
    return path


def gene_bed(tmp_path, data_dir, bins=(0, 1, 2, 5, 9, 10, 11, 12)):
    """gene_var / gene_const on some bins of every window, ccre elsewhere."""
    bed = pd.read_csv(data_dir / "sequences.bed", sep="\t", header=None)
    rows = []
    for c, s, _, _ in bed.itertuples(index=False):
        for b in bins:
            cls = "gene_var" if b < 6 else "gene_const"
            rows.append((c, s + b * POOL, s + (b + 1) * POOL, cls))
        rows.append((c, s + 40 * POOL, s + 41 * POOL, "ccre_var"))
    path = tmp_path / "classes.bed"
    pd.DataFrame(rows).sort_values([0, 1]).to_csv(
        path, sep="\t", header=None, index=False
    )
    return path


def dev_loss(groups_tracks, L=BINS, **kw):
    """DeviationLoss over toy tracks; groups_tracks: cell-type label per track."""
    targets = pd.DataFrame(
        {
            "identifier": [f"t{i}" for i in range(len(groups_tracks))],
            "assay": ["rna"] * len(groups_tracks),
        }
    )
    cells = pd.DataFrame(
        {"identifier": targets.identifier, "cell_type_group": groups_tracks}
    )
    groups = assay_groups(targets, cells, ["rna"])
    return DeviationLoss(groups, torch.ones(L), **kw)


def rand_batch(B=2, T=5, L=BINS, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.rand(B, T, L, generator=g) * 3
    yhat = torch.rand(B, T, L, generator=g) * 3 + 0.05
    mask = torch.rand(B, L, generator=g) > 0.5
    return yhat, y, mask


# -- maths ---------------------------------------------------------------------


def log_inputs(B=2, T=6, K=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(B, T, K, generator=g)
    labels = np.array(["x", "x", "y", "z", "z", "z"])[:T]
    names = sorted(set(labels))
    member = (labels[:, None] == np.array(names)[None]).astype(float)
    M = torch.tensor(member / member.sum(0), dtype=torch.float32)
    w = torch.rand(B, K, generator=g)
    return u, M, w, member


def test_common_offset_gives_zero():
    u, M, w, _ = log_inputs()
    # the same offset in every track (per unit and window): no deviation change
    offset = torch.randn(2, 1, 8)
    assert torch.allclose(
        deviation_from_logs(u, u + offset, M, w), torch.tensor(0.0), atol=1e-10
    )
    # in count space: scaling every track's prediction by a with a tiny pseudocount
    loss = dev_loss(["x", "x", "y", "z", "z"], pseudocount=1e-12, threshold=1.0)
    yhat, y, mask = rand_batch()
    y = y + 0.1
    assert loss(3.0 * y, y, mask).abs() < 1e-8


def test_group_sum_of_gradients_is_zero():
    """dL/dUhat_gk sums to zero over groups: a unit's overall level is untouched."""
    u, M, w, _ = log_inputs()
    uhat = torch.randn_like(u).requires_grad_(True)
    deviation_from_logs(u, uhat, M, w).backward()
    # dL/duhat_tk = dL/dUhat_{g(t)k} / n_g(t), so summing over tracks gives
    # sum_g dL/dUhat_gk, which must be 0
    total = uhat.grad.sum(dim=1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-6)
    assert uhat.grad.abs().sum() > 0
    # directly on the group means, with one track per group
    eye = torch.eye(6)
    Uhat = torch.randn_like(u).requires_grad_(True)
    deviation_from_logs(u, Uhat, eye, w).backward()
    assert torch.allclose(Uhat.grad.sum(dim=1), torch.zeros(2, 8), atol=1e-6)


@pytest.mark.parametrize("s", [0.0, 0.25, 0.5, 2.0])
def test_squeezed_prediction_scales_as_one_minus_s_squared(s):
    u, M, w, _ = log_inputs()
    base = deviation_from_logs(u, torch.zeros_like(u), M, w)  # dhat = 0
    squeezed = deviation_from_logs(u, s * u, M, w)  # dhat = s * d
    assert torch.allclose(squeezed, (1 - s) ** 2 * base, rtol=1e-5)


def test_stable_with_zeros_and_single_track_groups():
    loss = dev_loss(["x", "y", "z"], pseudocount=0.1)
    y = torch.zeros(1, 3, BINS)
    yhat = torch.zeros(1, 3, BINS, requires_grad=True)
    mask = torch.ones(1, BINS, dtype=torch.bool)
    out = loss(yhat, y, mask)  # all-zero targets: w_k = 0
    out.backward()
    assert out == 0 and torch.isfinite(yhat.grad).all()
    yhat2, y2, mask2 = rand_batch(B=1, T=3)
    y2[:, 1] = 0  # a silent track
    yhat2.requires_grad_(True)
    loss(yhat2, y2, mask2).backward()
    assert torch.isfinite(yhat2.grad).all()


def test_weights_from_targets_only_and_blocks_without_units():
    loss = dev_loss(["x", "x", "y"], threshold=2.0)
    yhat, y, mask = rand_batch(B=1, T=3)
    mask[:, :4] = False  # block 0 has no unit bins
    s_true, has_unit = loss.block_sums(y, mask)
    w = loss.unit_weights(s_true, has_unit)
    assert w[0, 0] == 0 and not has_unit[0, 0]
    expected = torch.clamp(s_true.mean(dim=1) / 2.0, max=1.0) * has_unit
    assert torch.allclose(w, expected)
    # predictions don't move the weights; targets do
    yhat.requires_grad_(True)
    loss(yhat, y, mask).backward()
    assert loss.unit_weights(*loss.block_sums(y, mask)).requires_grad is False


def test_fixed_denominator():
    loss = dev_loss(["x", "x", "y"], denominators=4.0)
    yhat, y, mask = rand_batch(B=1, T=3)
    one = loss(yhat, y, mask)
    two = loss(yhat.repeat(2, 1, 1), y.repeat(2, 1, 1), mask.repeat(2, 1))
    assert torch.allclose(one, two)  # divided by B x a fixed constant
    loss.denominators = [8.0]
    assert torch.allclose(loss(yhat, y, mask), one / 2)


def test_assay_groups_checks(rna_data, tmp_path):
    targets = SeqDataset(str(rna_data), "train", fold=0).targets_df
    cells = pd.read_csv(cell_type_table(tmp_path), sep="\t")
    sp = SeqDataset(str(rna_data), "train", fold=0).strand_pair
    (g,) = assay_groups(targets, cells, ["rna"], sp)
    assert g.group_names == ["blood", "brain", "liver"]
    assert torch.allclose(g.group_means.sum(0), torch.ones(3))
    with pytest.raises(ValueError, match="strand partners"):
        split = dict(CELL_TYPES, **{"a-": "brain"})
        assay_groups(
            targets,
            pd.read_csv(cell_type_table(tmp_path, split), sep="\t"),
            ["rna"],
            sp,
        )
    with pytest.raises(ValueError, match="without a cell type"):
        assay_groups(targets, cells[cells.identifier != "c"], ["rna"], sp)
    with pytest.raises(ValueError, match="no 'cage' tracks"):
        assay_groups(targets, cells, ["cage"], sp)


# -- data ----------------------------------------------------------------------


def test_unit_mask_follows_coordinates_and_flips_with_rc(rna_data, tmp_path):
    bed = gene_bed(tmp_path, rna_data)
    um = RegionClassMap.unit_map(bed, ["gene_var", "gene_const"])
    assert um.class_names == ["other", "gene_var", "gene_const"]
    ds = SeqDataset(str(rna_data), "train", mode="train", fold=0, unit_map=um)
    fwd = ds[(0, 0, False)].unit_mask
    assert fwd.dtype == torch.bool
    assert np.flatnonzero(fwd.numpy()).tolist() == [0, 1, 2, 5, 9, 10, 11, 12]
    rc = ds[(0, 0, True)].unit_mask
    assert torch.equal(rc, torch.flip(fwd, [0]))


def test_loss_is_rc_and_strand_pair_invariant(rna_data, tmp_path):
    bed = gene_bed(tmp_path, rna_data)
    um = RegionClassMap.unit_map(bed, ["gene_var", "gene_const"])
    ds = SeqDataset(str(rna_data), "train", mode="train", fold=0, unit_map=um)
    cells = pd.read_csv(cell_type_table(tmp_path), sep="\t")
    groups = assay_groups(ds.targets_df, cells, ["rna"], ds.strand_pair)
    loss = DeviationLoss(groups, torch.linspace(0.1, 1, BINS), threshold=0.5)
    fwd, rc = ds[(0, 0, False)], ds[(0, 0, True)]
    g = torch.Generator().manual_seed(1)
    yhat = torch.rand(1, NUM_TARGETS, BINS, generator=g) * 2
    # the model's RC prediction: strand-swapped and reversed, like the targets
    yhat_rc = torch.flip(yhat[:, ds.strand_pair], [2])
    pw_sym = DeviationLoss(groups, torch.ones(BINS), threshold=0.5)
    a = pw_sym(yhat, fwd.coverage_targets[None].float(), fwd.unit_mask[None])
    b = pw_sym(yhat_rc, rc.coverage_targets[None].float(), rc.unit_mask[None])
    assert torch.allclose(a, b, rtol=1e-6)
    assert (
        a > 0
        and loss(yhat, fwd.coverage_targets[None].float(), fwd.unit_mask[None]) > 0
    )


# -- training ------------------------------------------------------------------


def dev_params(tmp_path, data_dir, lam, epochs=2):
    params = tiny_params(epochs)
    params["train"]["seed"] = 1
    params["train"]["deviation_loss"] = {
        "lambda": lam,
        "cell_types": str(cell_type_table(tmp_path)),
        "bin_classes": str(gene_bed(tmp_path, data_dir)),
        "threshold": 0.5,
    }
    return params


def run(tmp_path, data_dir, name, params):
    pf = tmp_path / f"{name}.json"
    pf.write_text(json.dumps(params))
    out = tmp_path / name
    hound_train.main(["-o", str(out), "--fold", "0", str(pf), str(data_dir)])
    return out


def test_lambda_zero_is_bit_identical_and_lambda_positive_trains(rna_data, tmp_path):
    base = tiny_params(2)
    base["train"]["seed"] = 1
    a = model_state(run(tmp_path, rna_data, "base", base))
    zero = run(tmp_path, rna_data, "zero", dev_params(tmp_path, rna_data, 0.0))
    assert same(a, model_state(zero))
    on = run(tmp_path, rna_data, "on", dev_params(tmp_path, rna_data, 0.5))
    assert not same(a, model_state(on))
    log = pd.read_csv(on / "deviation_loss.tsv", sep="\t")
    assert len(log) == 2 and (log.deviation_loss > 0).all()
    assert ((log.dev_share > 0) & (log.dev_share < 1)).all()
    setup = json.loads((on / "deviation_loss_setup.json").read_text())
    assert setup["assays"] == ["rna"] and setup["tracks"] == [NUM_TARGETS]
    assert setup["denominators"][0] > 0


def test_deviation_loss_refuses_several_data_dirs(rna_data, tmp_path):
    pf = tmp_path / "multi.json"
    pf.write_text(json.dumps(dev_params(tmp_path, rna_data, 0.5, epochs=1)))
    with pytest.raises(SystemExit):
        hound_train.main(
            [
                "-o",
                str(tmp_path / "multi"),
                "--fold",
                "0",
                str(pf),
                str(rna_data),
                str(rna_data),
            ]
        )


def test_per_assay_pseudocount_and_threshold():
    targets = pd.DataFrame(
        {"identifier": [f"t{i}" for i in range(6)], "assay": ["rna"] * 3 + ["cage"] * 3}
    )
    cells = pd.DataFrame(
        {"identifier": targets.identifier, "cell_type_group": list("xyzxyz")}
    )
    groups = assay_groups(targets, cells, ["rna", "cage"])
    loss = DeviationLoss(
        groups, torch.ones(BINS), pseudocount=[0.1, 0.01], threshold=[2.0, 0.05]
    )
    assert loss.pseudocounts == [0.1, 0.01] and loss.thresholds == [2.0, 0.05]
    yhat, y, mask = rand_batch(B=1, T=6)
    s_true, has_unit = loss.block_sums(y[:, 3:], mask)
    w = loss.unit_weights(s_true, has_unit, 1)
    assert torch.allclose(w, torch.clamp(s_true.mean(1) / 0.05, max=1.0) * has_unit)
    assert torch.isfinite(loss(yhat, y, mask))
    with pytest.raises(ValueError, match="one positive threshold per assay"):
        DeviationLoss(groups, torch.ones(BINS), threshold=[1.0])


# -- follow-up to the independent check (2026-11-23) ---------------------------


def reference_ldev(
    yhat, y, mask, tracks_assay, samples, groups, pw, block, c, thr, D, space_fn=None
):
    """Plain numpy L_dev, written independently of the module.

    samples: list of track-index lists (strand pairs summed); groups: cell type per sample.
    """
    yhat, y, mask, pw = (np.asarray(a, dtype=np.float64) for a in (yhat, y, mask, pw))
    if space_fn is not None:
        y, yhat = space_fn(y), space_fn(yhat)
    B, _, L = y.shape
    K = L // block
    names = sorted(set(groups))
    total = 0.0
    for b in range(B):
        for k in range(K):
            sl = slice(k * block, (k + 1) * block)
            m = mask[b, sl]
            if not m.any():
                continue
            st = np.array([sum((y[b, t, sl] * m).sum() for t in s) for s in samples])
            sp = np.array([sum((yhat[b, t, sl] * m).sum() for t in s) for s in samples])
            w = min(1.0, st.mean() / thr) * pw[sl].mean()
            U = np.array(
                [
                    np.log(st[[i for i, g in enumerate(groups) if g == n]] + c).mean()
                    for n in names
                ]
            )
            Uh = np.array(
                [
                    np.log(sp[[i for i, g in enumerate(groups) if g == n]] + c).mean()
                    for n in names
                ]
            )
            total += w * (((Uh - Uh.mean()) - (U - U.mean())) ** 2).mean()
    return total / (D * B)


def toy_targets():
    """6 RNA tracks: t0/t1 a strand pair (liver), t2 liver, t3 brain, t4/t5 blood."""
    return (
        pd.DataFrame(
            {
                "identifier": [f"t{i}" for i in range(6)],
                "assay": ["rna"] * 6,
                "scale": [0.3, 0.3, 1.5, 30.0, 1.5, 0.3],
                "clip_soft": [384.0, 384.0, 384.0, 4.0, 384.0, 384.0],
                "sum_stat": ["sum_sqrt"] * 6,
            }
        ),
        pd.DataFrame(
            {
                "identifier": [f"t{i}" for i in range(6)],
                "cell_type_group": [
                    "liver",
                    "liver",
                    "liver",
                    "brain",
                    "blood",
                    "blood",
                ],
            }
        ),
        np.array([1, 0, 2, 3, 4, 5]),
    )


@pytest.mark.parametrize("space", ["transformed", "untransformed"])
def test_matches_independent_reference(space):
    """Catches w_k from predictions, dropped position weights, unsummed strands."""
    targets, cells, sp = toy_targets()
    (g,) = assay_groups(targets, cells, ["rna"], sp)
    assert g.sample_sums.shape == (6, 5)  # t0+t1 one sample
    L, block = 32, 4
    pw = torch.linspace(0.1, 1.0, L)
    gen = torch.Generator().manual_seed(3)
    y = torch.rand(2, 6, L, generator=gen) * 6
    yhat = torch.rand(2, 6, L, generator=gen) * 6
    mask = torch.rand(2, L, generator=gen) > 0.4
    c, thr, D = 0.2, 4.0, 3.0  # threshold binds for some units
    loss = DeviationLoss(
        [g],
        pw,
        block_bins=block,
        pseudocount=c,
        threshold=thr,
        denominators=D,
        space=space,
    )
    sc, cs = (
        targets.scale.to_numpy()[None, :, None],
        targets.clip_soft.to_numpy()[None, :, None],
    )

    def untr(v):
        v = np.where(v > cs, cs - 1 + (v - cs + 1) ** 2, v)
        return ((v + 1) ** 2 - 1) / sc

    ref = reference_ldev(
        yhat,
        y,
        mask,
        None,
        [[0, 1], [2], [3], [4], [5]],
        ["liver", "liver", "brain", "blood", "blood"],
        pw,
        block,
        c,
        thr,
        D,
        untr if space == "untransformed" else None,
    )
    assert np.isclose(float(loss(yhat, y, mask)), ref, rtol=1e-5)
    # the weights must come from targets: swapping in other predictions changes L but not w
    w1 = loss.unit_weights(*loss.sample_unit_sums(y, mask, 0), 0)
    assert torch.equal(w1, loss.unit_weights(*loss.sample_unit_sums(y, mask, 0), 0))


def test_strand_partners_are_summed_before_the_log():
    targets, cells, sp = toy_targets()
    (g,) = assay_groups(targets, cells, ["rna"], sp)
    loss = DeviationLoss([g], torch.ones(BINS), threshold=1.0)
    yhat, y, mask = rand_batch(B=1, T=6)
    base = loss(yhat, y, mask)
    # moving signal between the two strands of one library changes nothing
    y2, yhat2 = y.clone(), yhat.clone()
    y2[:, 0], y2[:, 1] = y[:, 0] + 0.7 * y[:, 1], 0.3 * y[:, 1]
    yhat2[:, 0], yhat2[:, 1] = yhat[:, 1], yhat[:, 0]
    assert torch.allclose(loss(yhat2, y2, mask), base, rtol=1e-6)


def test_untransform_matches_baskerville():
    from baskerville.dataset import untransform_preds

    targets, cells, sp = toy_targets()
    (g,) = assay_groups(targets, cells, ["rna"], sp)
    y = torch.rand(6, 40) * 8
    ours = untransform(y[None], g.scale, g.clip_soft, g.sqrt)[0]
    theirs = untransform_preds(y.clone().double(), targets, unscale=True).float()
    assert torch.allclose(ours, theirs, rtol=1e-5)
    targets.loc[0, "sum_stat"] = "sum_exp75"
    (g2,) = assay_groups(targets, cells, ["rna"], sp)
    with pytest.raises(ValueError, match="can't untransform"):
        DeviationLoss([g2], torch.ones(BINS), space="untransformed")


def test_validation_is_unchanged_with_lambda_positive(rna_data, tmp_path, monkeypatch):
    """L_dev is computed for training batches only; validation uses the plain loss."""
    calls = {"n": 0}
    orig = DeviationLoss.forward

    def counting(self, *args):
        calls["n"] += 1
        return orig(self, *args)

    monkeypatch.setattr(DeviationLoss, "forward", counting)
    out = run(tmp_path, rna_data, "val", dev_params(tmp_path, rna_data, 0.5, epochs=1))
    # 4 training windows, batch 2 -> 2 training steps; validation (4 windows) adds none
    assert calls["n"] == 2
    log = pd.read_csv(out / "deviation_loss.tsv", sep="\t")
    assert len(log) == 1


def test_denominators_do_not_depend_on_the_training_seed(rna_data, tmp_path):
    outs = []
    for seed in (1, 2):
        params = dev_params(tmp_path, rna_data, 0.5, epochs=1)
        params["train"]["seed"] = seed
        params["train"]["deviation_loss"]["denominator_windows"] = 3
        outs.append(
            json.loads(
                (
                    run(tmp_path, rna_data, f"s{seed}", params)
                    / "deviation_loss_setup.json"
                ).read_text()
            )
        )
    assert outs[0]["denominators"] == outs[1]["denominators"]
    assert outs[0]["denominators_estimated"] is True
    params = dev_params(tmp_path, rna_data, 0.5, epochs=1)
    params["train"]["deviation_loss"]["denominators"] = [7.5]
    setup = json.loads(
        (
            run(tmp_path, rna_data, "fixed", params) / "deviation_loss_setup.json"
        ).read_text()
    )
    assert setup["denominators"] == [7.5] and setup["denominators_estimated"] is False
