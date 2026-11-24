"""Online-reweighting hook support (track 3A): pre-step hook, primitives, meta loader,
early-stopping subset, checkpointed state, projection, and the alignment hook."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from baskerville.dataset import SeqDataset
from baskerville.loss_weights import project_weights, rows_in_bed
from baskerville.online_weights import GradientAlignmentHook
from baskerville.scripts import hound_train
from tests.test_output_loss_weights import (
    make_data,
    model_state,
    same,
    tiny_params,
    write_group_table,
)


@pytest.fixture
def data_dir(tmp_path):
    return make_data(tmp_path / "data")


def classes(tmp_path, data_dir, table):
    """Class 'hot' on the first half of every chr1 window; per-assay table."""
    bed = pd.read_csv(data_dir / "sequences.bed", sep="\t", header=None)
    rows = [
        (c, s, s + (e - s) // 2, "hot")
        for c, s, e, _ in bed.itertuples(index=False)
        if c == "chr1"
    ]
    path = tmp_path / "classes.bed"
    pd.DataFrame(rows).to_csv(path, sep="\t", header=False, index=False)
    return path, write_group_table(tmp_path, table)


def meta_bed(tmp_path, data_dir, n=2):
    """The first n validation-fold (fold1) windows."""
    bed = pd.read_csv(data_dir / "sequences.bed", sep="\t", header=None)
    rows = bed[bed[3] == "fold1"].iloc[:n, :3]
    path = tmp_path / "meta.bed"
    rows.to_csv(path, sep="\t", header=False, index=False)
    return path


TABLE = {
    "bg": {"rna": 1.0, "dnase": 1.0, "chip": 1.0},
    "hot": {"rna": 3.0, "dnase": 0.5, "chip": 2.0},
}
ONES = {
    "bg": {"rna": 1.0, "dnase": 1.0, "chip": 1.0},
    "hot": {"rna": 1.0, "dnase": 1.0, "chip": 1.0},
}


def params_for(
    tmp_path,
    data_dir,
    table=TABLE,
    hook=None,
    epochs=2,
    meta=True,
    dtype="bfloat16",
    dropout=0.0,
    **train,
):
    p = tiny_params(epochs)
    p["model"]["trunk"][1]["norm_type"] = "batch"  # BatchNorm, to check running stats
    if dropout:  # a block with dropout, to check the hook leaves the global RNG alone
        p["model"]["trunk"].append(
            {
                "name": "MBConvBlock",
                "channels": 16,
                "expand": 2.0,
                "kernel_size": 3,
                "gated": True,
                "dropout": dropout,
                "layer_scale": 0,  # full residual branch, so dropout matters
            }
        )
    p["train"].update({"seed": 3, "mix_dtype": dtype, **train})
    if dtype == "float32":
        del p["train"]["mix_dtype"]  # the trainer's default
    bed, tsv = classes(tmp_path, data_dir, table)
    lw = {"bin_classes": str(bed), "class_weights": str(tsv), "default_class": "bg"}
    if meta:
        mb = str(meta_bed(tmp_path, data_dir))
        lw["meta_valid"] = {"bed": mb, "batch_size": 2, "seed": 5}
        p["train"]["early_stop_valid"] = {"exclude_bed": mb}
    if hook:
        lw["hook"] = hook
    p["train"]["loss_weights"] = lw
    return p


def run(tmp_path, data_dir, name, params):
    pf = tmp_path / f"{name}.json"
    pf.write_text(json.dumps(params))
    out = tmp_path / name
    hound_train.main(["-o", str(out), "--fold", "0", str(pf), str(data_dir)])
    return out


def hook_spec(cls, **args):
    return {"class": cls, "args": args}


# -- R6 projection -------------------------------------------------------------


def test_projection_holds_bounds_and_group_means():
    counts = np.array([90, 5, 5])
    w = np.array([[1.0, 1.0], [40.0, 0.01], [0.01, 9.0]])
    out = project_weights(w, counts, 0.2, 5.0)
    assert (out >= 0.2 - 1e-12).all() and (out <= 5.0 + 1e-12).all()
    assert np.allclose((out * counts[:, None]).sum(0) / counts.sum(), 1.0)
    # a single clip-then-rescale would leave the bounds here
    once = np.clip(w, 0.2, 5.0)
    once = once / ((once * counts[:, None]).sum(0) / counts.sum())
    assert (once < 0.2).any() or (once > 5.0).any()
    t = project_weights(torch.tensor(w, dtype=torch.float32), counts, 0.2, 5.0)
    assert torch.allclose(t, torch.tensor(out, dtype=torch.float32))


# -- R8.1 eta = 0 ----------------------------------------------------------------


def test_eta_zero_hook_is_bit_identical_to_static_weights(tmp_path, data_dir):
    static = run(
        tmp_path, data_dir, "static", params_for(tmp_path, data_dir, dropout=0.3)
    )
    hook = hook_spec(
        "baskerville.online_weights:GradientAlignmentHook", eta=0.0, every=1
    )
    online = run(
        tmp_path,
        data_dir,
        "eta0",
        params_for(tmp_path, data_dir, hook=hook, dropout=0.3),
    )
    assert same(model_state(static), model_state(online))
    log = pd.read_csv(online / "online_weights.tsv", sep="\t")
    assert len(log) == 4 and log.step.tolist() == [0, 1, 2, 3]  # 2 steps x 2 epochs
    meta = pd.read_csv(online / "meta_valid.tsv", sep="\t")
    assert len(meta) == 2 and (meta.meta_loss_pmn > 0).all()


# -- R8.2 a change in before_step applies to this batch; hook grads are dropped ---


class SetWeightsAtStart:
    """At step 0, sets the table to a target and leaves junk gradients behind."""

    target = None

    def before_step(self, trainer, batch, di):
        if trainer._step != 0:
            return
        sum(p.sum() for p in trainer.trainable_params()).backward()  # junk grads
        with torch.no_grad():
            trainer.class_weights[0].copy_(SetWeightsAtStart.target)


def test_before_step_change_applies_to_this_batch(tmp_path, data_dir):
    static = run(
        tmp_path, data_dir, "static", params_for(tmp_path, data_dir, meta=False)
    )
    target = pd.read_csv(static / "class_weights_ds0.tsv", sep="\t")
    SetWeightsAtStart.target = torch.tensor(
        target[["weight_rna", "weight_dnase", "weight_chip"]].to_numpy(),
        dtype=torch.float32,
    )
    hook = hook_spec(f"{__name__}:SetWeightsAtStart")
    online = run(
        tmp_path,
        data_dir,
        "set",
        params_for(tmp_path, data_dir, table=ONES, hook=hook, meta=False),
    )
    assert same(model_state(static), model_state(online))


# -- R8.3 primitives leave parameters and BatchNorm stats untouched -------------


class CheckPrimitives:
    ok = None
    detail = None

    def before_step(self, trainer, batch, di):
        if trainer._step != 1:
            return
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        rng = torch.get_rng_state()
        hook = GradientAlignmentHook()
        direction, *_ = hook.val_direction(trainer, 1)
        a = trainer.forward(batch, rng_seed=7)
        trainer.perturb_params(direction, 0.01)
        trainer.forward(batch, rng_seed=7)
        trainer.restore_params()
        b = trainer.forward(batch, rng_seed=7)
        after = trainer.model.state_dict()
        d = {
            "state": [k for k in before if not torch.equal(before[k], after[k])],
            "same_output": torch.equal(a, b),
            "rng": torch.equal(rng, torch.get_rng_state()),
            "grads_none": all(p.grad is None for p in trainer.trainable_params()),
        }
        CheckPrimitives.detail = d
        CheckPrimitives.ok = (
            not d["state"] and d["same_output"] and d["rng"] and d["grads_none"]
        )


def test_primitives_leave_params_buffers_and_rng_unchanged(tmp_path, data_dir):
    CheckPrimitives.ok = None
    hook = hook_spec(f"{__name__}:CheckPrimitives")
    run(
        tmp_path,
        data_dir,
        "prims",
        params_for(tmp_path, data_dir, hook=hook, epochs=1, dropout=0.3),
    )
    assert CheckPrimitives.ok is True, CheckPrimitives.detail


# -- R8.4 finite differences vs exact double backward ---------------------------


class CompareAlignment:
    """At step 1: the hook's finite-difference alignment and the exact one, on the
    same dropout masks (the hook's FD seed) and in the same precision."""

    result = None
    dtype = "float32"
    eps = 1e-3

    def before_step(self, trainer, batch, di):
        if trainer._step != 1:
            return
        hook = GradientAlignmentHook(
            eps=CompareAlignment.eps, fd_dtype=CompareAlignment.dtype
        )
        dtype = hook.resolved_fd_dtype(trainer)
        direction, *_ = hook.val_direction(trainer, 1)
        W = trainer.class_weights[0]
        a_fd = hook.alignment(trainer, batch, direction, 1, W)
        Wl = W.detach().clone().requires_grad_(True)
        params = trainer.trainable_params()
        y = trainer.forward(batch, rng_seed=hook.fd_seed(1), grad=True, dtype=dtype)
        loss = trainer.loss_from_pred(y, batch, Wl, dtype=dtype)
        g = torch.autograd.grad(loss, params, create_graph=True)
        s = sum((gi * dir_).sum() for gi, dir_ in zip(g, direction))
        (a_exact,) = torch.autograd.grad(s, Wl)
        CompareAlignment.result = (a_fd.flatten(), a_exact.detach().flatten())


@pytest.mark.parametrize(
    "dtype, eps, dropout, min_corr",
    [
        ("float32", 1e-3, 0.0, 0.9999),
        ("float32", 1e-3, 0.3, 0.9999),  # +eps / -eps share dropout masks
        ("auto", 1e-3, 0.3, 0.9999),  # auto = float32 FD under bf16 training
        # bf16 FD needs a larger step: tiny model corr 0.21 / 0.85 / 0.99 at eps 1e-3 / 1e-2 / 1e-1
        ("bfloat16", 1e-1, 0.0, 0.98),
    ],
)
def test_finite_difference_alignment_matches_exact(
    tmp_path, data_dir, dtype, eps, dropout, min_corr
):
    CompareAlignment.result = None
    CompareAlignment.dtype, CompareAlignment.eps = dtype, eps
    hook = hook_spec(f"{__name__}:CompareAlignment")
    params = params_for(tmp_path, data_dir, hook=hook, epochs=1, dropout=dropout)
    run(tmp_path, data_dir, f"fd_{dtype}_{dropout}", params)
    fd, exact = CompareAlignment.result
    corr = float(torch.corrcoef(torch.stack([fd.float(), exact.float()]))[0, 1])
    assert corr > min_corr


# -- R8.5 resume restores weights, hook state, step and meta position ------------


def test_resume_restores_online_state(tmp_path, data_dir):
    hook = hook_spec(
        "baskerville.online_weights:GradientAlignmentHook", eta=0.5, every=1
    )
    full = run(
        tmp_path, data_dir, "full", params_for(tmp_path, data_dir, hook=hook, epochs=3)
    )
    part = params_for(tmp_path, data_dir, hook=hook, epochs=1)
    run(tmp_path, data_dir, "resumed", part)
    resumed = run(
        tmp_path,
        data_dir,
        "resumed",
        params_for(tmp_path, data_dir, hook=hook, epochs=3),
    )
    a = torch.load(full / "checkpoint.pth", map_location="cpu")
    b = torch.load(resumed / "checkpoint.pth", map_location="cpu")
    assert same(a["model"], b["model"])
    assert torch.equal(a["class_weights"][0], b["class_weights"][0])
    assert not torch.allclose(
        a["class_weights"][0],
        torch.tensor(
            project_weights(
                np.array([[1.0, 1.0, 1.0], [3.0, 0.5, 2.0]]),
                np.array([192, 64]),
                0.2,
                5.0,
            ),
            dtype=torch.float32,
        ),
    )
    assert a["hook_state"] == b["hook_state"] and a["hook_state"]["updates"] == 6
    assert a["step"] == b["step"] == 6 and a["meta_position"] == b["meta_position"] == 6


# -- R8.6 meta loader seeded and worker-independent; early stopping excludes it --


class RecordMeta:
    seen = []

    def before_step(self, trainer, batch, di):
        RecordMeta.seen.append(trainer.next_meta_batch().seq_index.tolist())


def test_meta_loader_is_seeded_and_worker_independent(tmp_path, data_dir):
    streams = []
    for workers in (0, 2):
        RecordMeta.seen = []
        p = params_for(
            tmp_path, data_dir, hook=hook_spec(f"{__name__}:RecordMeta"), epochs=2
        )
        p["train"]["loss_weights"]["meta_valid"]["num_workers"] = workers
        run(tmp_path, data_dir, f"meta{workers}", p)
        streams.append(list(RecordMeta.seen))
    assert streams[0] == streams[1] and len(streams[0]) == 4
    bed = pd.read_csv(data_dir / "sequences.bed", sep="\t", header=None)
    meta_rows = set(np.flatnonzero(bed[3].to_numpy() == "fold1")[:2])
    assert {i for batch in streams[0] for i in batch} == meta_rows


def test_early_stopping_subset_excludes_meta_windows(tmp_path, data_dir):
    valid = SeqDataset(str(data_dir), "valid", fold=0)
    mb = meta_bed(tmp_path, data_dir)
    sub = hound_train.early_stop_subset(
        valid, {"early_stop_valid": {"exclude_bed": str(mb)}}
    )
    assert len(sub) == len(valid) - 2
    meta_idx = set(rows_in_bed(valid, mb).tolist())
    assert not meta_idx & set(sub.indices)
    chrom = hound_train.early_stop_subset(
        valid, {"early_stop_valid": {"chroms": ["chr1"]}}
    )
    assert all(
        valid.seqs_bed.chrom.iloc[valid.seq_indexes[i]] == "chr1" for i in chrom.indices
    )
    assert hound_train.early_stop_subset(valid, {}) is valid


# -- eps_rel, fd_scan, init_model (requests of 2026-11-23) -----------------------


def test_relative_step_follows_the_parameter_norm(tmp_path, data_dir):
    class Fake:
        def __init__(self, scale):
            self.p = [torch.full((4,), scale), torch.full((5,), scale)]

        def trainable_params(self):
            return self.p

    hook = GradientAlignmentHook(eps=0.5, eps_rel=1e-3)
    assert np.isclose(hook.step_size(Fake(1.0)), 1e-3 * 3.0)  # ||theta|| = 3
    assert np.isclose(hook.step_size(Fake(2.0)), 1e-3 * 6.0)
    assert GradientAlignmentHook(eps=0.5).step_size(Fake(1.0)) == 0.5
    spec = hook_spec(
        "baskerville.online_weights:GradientAlignmentHook",
        eta=0.5,
        every=1,
        eps_rel=1e-3,
    )
    out = run(tmp_path, data_dir, "rel", params_for(tmp_path, data_dir, hook=spec))
    eps = pd.read_csv(out / "online_weights.tsv", sep="\t").eps
    assert (eps > 0).all() and eps.nunique() > 1  # recomputed as the parameters move


def test_fd_scan_logs_and_leaves_training_unchanged(tmp_path, data_dir):
    base = dict(eta=0.5, every=1, eps_rel=1e-2)
    plain = run(
        tmp_path,
        data_dir,
        "plain",
        params_for(
            tmp_path,
            data_dir,
            hook=hook_spec("baskerville.online_weights:GradientAlignmentHook", **base),
            dropout=0.3,
        ),
    )
    scan = {"rel": [1e-3, 1e-2], "ref_rel": [1e-5, 1e-4], "every_updates": 2}
    scanned = run(
        tmp_path,
        data_dir,
        "scan",
        params_for(
            tmp_path,
            data_dir,
            hook=hook_spec(
                "baskerville.online_weights:GradientAlignmentHook", fd_scan=scan, **base
            ),
            dropout=0.3,
        ),
    )
    a = torch.load(plain / "checkpoint.pth", map_location="cpu")
    b = torch.load(scanned / "checkpoint.pth", map_location="cpu")
    assert same(a["model"], b["model"])
    assert torch.equal(a["class_weights"][0], b["class_weights"][0])
    log = pd.read_csv(scanned / "fd_scan.tsv", sep="\t")
    assert sorted(log["update"].unique()) == [0, 2]  # every 2nd of 4 updates
    assert len(log) == 2 * 4  # 2 fp32 refs + 2 bf16 steps per scanned update
    first = log[log["precision"] == "float32"].groupby("update").head(1)
    assert np.allclose(first.corr_vs_ref, 1.0)
    refs = log[(log.precision == "float32") & (log.rel == 1e-4)]
    assert (
        refs.corr_vs_ref > 0.99
    ).all()  # fp32 references agree (0.998 with dropout here)


def test_init_model_starts_from_given_weights_with_a_fresh_optimizer(
    tmp_path, data_dir
):
    first = run(tmp_path, data_dir, "first", params_for(tmp_path, data_dir, meta=False))
    p = params_for(tmp_path, data_dir, meta=False, epochs=1, learning_rate=0.0)
    p["train"]["init_model"] = str(first / "model_best.pth")
    second = run(tmp_path, data_dir, "second", p)
    best = torch.load(first / "model_best.pth", map_location="cpu")
    ck = torch.load(second / "checkpoint.pth", map_location="cpu")
    params = [k for k in best if "running" not in k and "num_batches" not in k]
    assert all(torch.equal(best[k], ck["model"][k]) for k in params)  # lr 0: unchanged
    steps = {float(s["step"]) for s in ck["optimizer"]["state"].values()}
    assert steps == {2.0}  # a fresh optimizer: only this run's 2 steps


# -- follow-up to the hook check (2026-11-23) -------------------------------------

ALIGN = "baskerville.online_weights:GradientAlignmentHook"


def test_update_moves_weights_toward_positive_alignment(tmp_path, data_dir):
    """A flipped update sign fails here: within each group, the hot/bg weight ratio
    must move in the direction of a[hot] - a[bg]."""
    spec = hook_spec(ALIGN, eta=0.1, every=1, lo=0.01, hi=100.0)
    out = run(
        tmp_path, data_dir, "sign", params_for(tmp_path, data_dir, hook=spec, epochs=1)
    )
    start = pd.read_csv(out / "class_weights_ds0.tsv", sep="\t")
    row = pd.read_csv(out / "online_weights.tsv", sep="\t").iloc[0]
    checked = 0
    for gi, g in enumerate(["rna", "dnase", "chip"]):
        old = np.log(start[f"weight_{g}"][1] / start[f"weight_{g}"][0])
        new = np.log(row[f"w[1,{gi}]"] / row[f"w[0,{gi}]"])
        da = row[f"a[1,{gi}]"] - row[f"a[0,{gi}]"]
        if abs(da) > 1e-12:
            assert np.sign(new - old) == np.sign(da)
            checked += 1
    assert checked >= 2


def test_fd_check_is_logged_and_can_abort(tmp_path, data_dir):
    out = run(
        tmp_path,
        data_dir,
        "check",
        params_for(
            tmp_path, data_dir, hook=hook_spec(ALIGN, eta=0.1, every=1), epochs=1
        ),
    )
    ratio = pd.read_csv(out / "online_weights.tsv", sep="\t").fd_check_ratio
    assert np.allclose(ratio, 1.0, atol=0.02)  # fp32 FD resolves L_val's derivative
    strict = hook_spec(ALIGN, eta=0.1, every=1, check_tol=-1.0, check_action="abort")
    with pytest.raises(RuntimeError, match="finite difference isn't resolved"):
        run(
            tmp_path,
            data_dir,
            "abort",
            params_for(tmp_path, data_dir, hook=strict, epochs=1),
        )


def test_non_finite_updates_are_skipped(tmp_path, data_dir, monkeypatch):
    def nan_alignment(
        self, trainer, batch, direction, step, weights, eps=None, dtype=None
    ):
        return torch.full_like(weights, float("nan"))

    monkeypatch.setattr(GradientAlignmentHook, "alignment", nan_alignment)
    with pytest.warns(UserWarning, match="update skipped"):
        out = run(
            tmp_path,
            data_dir,
            "nan",
            params_for(
                tmp_path,
                data_dir,
                hook=hook_spec(ALIGN, eta=0.5, every=1, check=False),
                epochs=1,
            ),
        )
    log = pd.read_csv(out / "online_weights.tsv", sep="\t")
    assert (log.skipped == 1).all()
    ck = torch.load(out / "checkpoint.pth", map_location="cpu")
    start = pd.read_csv(out / "class_weights_ds0.tsv", sep="\t")
    expected = start[["weight_rna", "weight_dnase", "weight_chip"]].to_numpy()
    assert np.allclose(ck["class_weights"][0].numpy(), expected)  # unchanged
    assert ck["hook_state"]["scale"] is None and ck["hook_state"]["skipped"] == 2


def test_precondition_warns_without_second_moments(tmp_path, data_dir):
    spec = hook_spec(ALIGN, eta=0.1, every=1, precondition=True)
    with pytest.warns(UserWarning, match="exp_avg_sq are missing"):
        run(
            tmp_path,
            data_dir,
            "precond",
            params_for(tmp_path, data_dir, hook=spec, epochs=1),
        )


def test_projection_zero_weights_and_impossible_bounds():
    counts = np.array([50, 50])
    out = project_weights(np.array([[0.0], [3.0]]), counts, 0.2, 5.0)
    assert np.isclose(out[0, 0], 0.2) and np.isclose(
        (out[:, 0] * [0.5, 0.5]).sum(), 1.0
    )
    with pytest.raises(ValueError, match="can't reach mean 1"):
        project_weights(
            np.array([[0.0], [3.0]]), counts, 0.2, 1.5
        )  # 0.5*0.2 + 0.5*1.5 < 1
    with pytest.raises(ValueError, match="no positive weight"):
        project_weights(np.array([[0.0], [0.0]]), counts, 0.2, 5.0)


def test_meta_windows_in_the_early_stopping_set_raise(tmp_path, data_dir):
    p = params_for(
        tmp_path, data_dir, hook=hook_spec(ALIGN, eta=0.1, every=1), epochs=1
    )
    del p["train"]["early_stop_valid"]
    with pytest.raises(SystemExit):
        run(tmp_path, data_dir, "overlap", p)


def test_resume_drops_log_rows_written_after_the_checkpoint(tmp_path, data_dir):
    spec = hook_spec(ALIGN, eta=0.5, every=1)
    run(tmp_path, data_dir, "logs", params_for(tmp_path, data_dir, hook=spec, epochs=1))
    out = tmp_path / "logs"
    # a preempted second epoch: rows written after the epoch-1 checkpoint
    for name, col, value in (
        ("online_weights.tsv", "step", 2),
        ("meta_valid.tsv", "epoch", 1),
    ):
        log = pd.read_csv(out / name, sep="\t")
        extra = log.tail(1).copy()
        extra[col] = value
        pd.concat([log, extra]).to_csv(out / name, sep="\t", index=False)
    run(tmp_path, data_dir, "logs", params_for(tmp_path, data_dir, hook=spec, epochs=2))
    steps = pd.read_csv(out / "online_weights.tsv", sep="\t").step.tolist()
    epochs = pd.read_csv(out / "meta_valid.tsv", sep="\t").epoch.tolist()
    assert steps == [0, 1, 2, 3] and epochs == [0, 1]


def test_primitives_with_batchnorm_in_checkpointed_blocks(tmp_path, data_dir):
    CheckPrimitives.ok = None
    p = params_for(
        tmp_path, data_dir, hook=hook_spec(f"{__name__}:CheckPrimitives"), epochs=1
    )
    p["model"]["trunk"][1]["grad_checkpoint"] = True
    run(tmp_path, data_dir, "ckpt_bn", p)
    assert CheckPrimitives.ok is True, CheckPrimitives.detail


def test_frozen_norm_stats_covers_cond_batchnorm():
    from baskerville.condnorm import CondBatchNorm1d
    from baskerville.trainer import Trainer

    class Holder:
        model = torch.nn.ModuleList([CondBatchNorm1d(2, 4), torch.nn.BatchNorm1d(4)])

    h = Holder()
    h.model.train()
    before = {k: v.clone() for k, v in h.model.state_dict().items()}
    x = torch.randn(3, 4, 5) * 3 + 1
    with Trainer.frozen_norm_stats(h):
        h.model[0](x, 1)
        h.model[1](x)
    after = h.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    h.model[0](x, 1)  # outside the block, stats move
    assert not torch.equal(
        before["0.running_mean"], h.model.state_dict()["0.running_mean"]
    )


def test_resume_in_the_first_epoch_drops_its_log_rows(tmp_path, data_dir):
    out = tmp_path / "first"
    out.mkdir()
    pd.DataFrame({"step": [0, 1], "x": [1, 2]}).to_csv(
        out / "online_weights.tsv", sep="\t", index=False
    )
    pd.DataFrame({"epoch": [0], "meta_loss_pmn": [1.0]}).to_csv(
        out / "meta_valid.tsv", sep="\t", index=False
    )
    spec = hook_spec(ALIGN, eta=0.1, every=1)
    run(
        tmp_path, data_dir, "first", params_for(tmp_path, data_dir, hook=spec, epochs=1)
    )
    log = pd.read_csv(out / "online_weights.tsv", sep="\t")
    assert log.step.tolist() == [0, 1] and "x" not in log  # rewritten by this run only
    assert pd.read_csv(out / "meta_valid.tsv", sep="\t").epoch.tolist() == [0]


def test_init_model_does_not_override_a_checkpoint(tmp_path, data_dir):
    other = run(
        tmp_path, data_dir, "other", params_for(tmp_path, data_dir, meta=False, seed=9)
    )
    run(tmp_path, data_dir, "a", params_for(tmp_path, data_dir, meta=False, epochs=1))
    run(tmp_path, data_dir, "b", params_for(tmp_path, data_dir, meta=False, epochs=1))
    plain = run(
        tmp_path, data_dir, "a", params_for(tmp_path, data_dir, meta=False, epochs=2)
    )
    p = params_for(tmp_path, data_dir, meta=False, epochs=2)
    p["train"]["init_model"] = str(other / "model_best.pth")
    resumed = run(tmp_path, data_dir, "b", p)  # out_dir has a checkpoint: resume wins
    assert same(model_state(plain), model_state(resumed))


def test_resume_loads_the_checkpoint_on_the_cpu(tmp_path, data_dir, monkeypatch):
    """A checkpoint saved on a GPU must resume anywhere: torch.load maps to the CPU."""
    run(
        tmp_path,
        data_dir,
        "cpu_resume",
        params_for(tmp_path, data_dir, meta=False, epochs=1),
    )
    calls = []
    orig = torch.load

    def recording(path, *args, **kwargs):
        calls.append((str(path), kwargs.get("map_location")))
        return orig(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording)
    run(
        tmp_path,
        data_dir,
        "cpu_resume",
        params_for(tmp_path, data_dir, meta=False, epochs=2),
    )
    ck = [m for p, m in calls if p.endswith("checkpoint.pth")]
    assert ck and all(m == "cpu" for m in ck)


def test_require_cuda_fails_fast_without_a_gpu(tmp_path, data_dir, monkeypatch):
    monkeypatch.setenv("BASKERVILLE_REQUIRE_CUDA", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        run(
            tmp_path,
            data_dir,
            "nocuda",
            params_for(tmp_path, data_dir, meta=False, epochs=1),
        )
