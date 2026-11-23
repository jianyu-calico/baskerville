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
            }
        )
    p["train"].update({"seed": 3, "mix_dtype": dtype, **train})
    if dtype == "float32":
        del p["train"]["mix_dtype"]  # the trainer's default
    bed, tsv = classes(tmp_path, data_dir, table)
    lw = {"bin_classes": str(bed), "class_weights": str(tsv), "default_class": "bg"}
    if meta:
        lw["meta_valid"] = {
            "bed": str(meta_bed(tmp_path, data_dir)),
            "batch_size": 2,
            "seed": 5,
        }
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
        direction, _, _ = hook.val_direction(trainer, 1)
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
    """At step 1: the hook's finite-difference alignment and the exact one."""

    result = None
    dtype = torch.float32
    eps = 1e-3

    def before_step(self, trainer, batch, di):
        if trainer._step != 1:
            return
        saved, trainer.mix_dtype = trainer.mix_dtype, CompareAlignment.dtype
        hook = GradientAlignmentHook(eps=CompareAlignment.eps)
        direction, _, _ = hook.val_direction(trainer, 1)
        W = trainer.class_weights[0]
        a_fd = hook.alignment(trainer, batch, direction, 1, W)
        Wl = W.detach().clone().requires_grad_(True)
        params = trainer.trainable_params()
        y = trainer.forward(batch, rng_seed=0, grad=True)
        g = torch.autograd.grad(
            trainer.loss_from_pred(y, batch, Wl), params, create_graph=True
        )
        s = sum((gi * dir_).sum() for gi, dir_ in zip(g, direction))
        (a_exact,) = torch.autograd.grad(s, Wl)
        CompareAlignment.result = (a_fd.flatten(), a_exact.detach().flatten())
        trainer.mix_dtype = saved


@pytest.mark.parametrize(
    "dtype, eps, min_corr",
    [
        (torch.float32, 1e-3, 0.9999),
        # bf16 needs a larger step: tiny model corr 0.21 / 0.85 / 0.99 at eps 1e-3 / 1e-2 / 1e-1
        (torch.bfloat16, 1e-1, 0.98),
    ],
)
def test_finite_difference_alignment_matches_exact(
    tmp_path, data_dir, dtype, eps, min_corr
):
    CompareAlignment.result = None
    CompareAlignment.dtype, CompareAlignment.eps = dtype, eps
    hook = hook_spec(f"{__name__}:CompareAlignment")
    run(
        tmp_path,
        data_dir,
        f"fd_{eps}",
        params_for(tmp_path, data_dir, hook=hook, epochs=1),
    )
    fd, exact = CompareAlignment.result
    corr = float(torch.corrcoef(torch.stack([fd.float(), exact.float()]))[0, 1])
    sign = float((torch.sign(fd) == torch.sign(exact)).float().mean())
    print("FD", dtype, eps, "corr", round(corr, 6), "sign", sign)
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
    assert (refs.corr_vs_ref > 0.999).all()  # converged fp32 references agree


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
