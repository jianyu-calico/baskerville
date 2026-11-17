"""Output-bin and per-sequence loss weights, seeded sampling, and the online-weight hook."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from baskerville import dna
from baskerville.dataset import MultiDataset, MultiSampler, SeqDataset
from baskerville.loss_weights import (
    RegionClassMap,
    normalize_class_weights,
    read_sample_weights,
)
from baskerville.metrics import PoissonMultinomialLoss
from baskerville.scripts import hound_train
from baskerville.types import BatchData

SEQ_LEN, POOL, NUM_TARGETS = 2048, 32, 4
BINS = SEQ_LEN // POOL
WINDOWS_PER_FOLD, NUM_FOLDS = 4, 3
KW = {"total_weight": 0.25, "weight_range": 10, "weight_exp": 8}  # Cerberus poisson_mn


# -- fixtures ------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path):
    """3 folds x 4 windows on chr1/chr2; folds interleaved in sequences.bed."""
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    (root / "examples").mkdir(parents=True)
    rows = []
    for k in range(NUM_FOLDS):
        for i in range(WINDOWS_PER_FOLD):
            start = 10_000 * (k * WINDOWS_PER_FOLD + (WINDOWS_PER_FOLD - 1 - i))
            rows.append(
                ("chr1" if i % 2 == 0 else "chr2", start, start + SEQ_LEN, f"fold{k}")
            )
    order = [r for i in range(WINDOWS_PER_FOLD) for r in rows[i::WINDOWS_PER_FOLD]]
    pd.DataFrame(order).to_csv(
        root / "sequences.bed", sep="\t", header=False, index=False
    )
    for k in range(NUM_FOLDS):
        g = zarr.open_group(str(root / "examples" / f"fold{k}.zarr"), mode="w")
        n = WINDOWS_PER_FOLD
        g.create_array(
            "sequence", data=rng.integers(0, 4, size=(n, SEQ_LEN), dtype=np.uint8)
        )
        g.create_array(
            "target",
            data=rng.gamma(1.0, 2.0, size=(n, NUM_TARGETS, BINS)).astype(np.float16),
        )
    pd.DataFrame(
        {
            "identifier": ["a+", "a-", "b", "c"],
            "description": ["RNA:a", "RNA:a", "DNASE:b", "DNASE:c"],
            "strand_pair": [1, 0, 2, 3],
        }
    ).to_csv(root / "targets.txt", sep="\t")
    stats = {
        "num_targets": NUM_TARGETS,
        "seq_length": SEQ_LEN,
        "pool_width": POOL,
        "crop_bp": 0,
        "target_length": BINS,
    }
    (root / "statistics.json").write_text(json.dumps(stats))
    return root


def write_classes(tmp_path, rows, weights, default="bg"):
    bed, tsv = tmp_path / "classes.bed", tmp_path / "weights.tsv"
    pd.DataFrame(rows).to_csv(bed, sep="\t", header=False, index=False)
    pd.DataFrame({"class": list(weights), "weight": list(weights.values())}).to_csv(
        tsv, sep="\t", index=False
    )
    return bed, tsv, RegionClassMap.from_files(bed, tsv, default)


def onehot(seqi):
    x = np.zeros((5, len(seqi)), dtype=bool)
    x[seqi, np.arange(len(seqi))] = 1
    return torch.from_numpy(x[:-1]).half()


def loss_data(B=3, T=4, L=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, T, L, generator=g) * 5 + 0.1, torch.rand(
        B, T, L, generator=g
    ) * 5


# -- loss ----------------------------------------------------------------------


def test_all_ones_weights_equal_unweighted_loss():
    y_pred, y_true = loss_data()
    tw = torch.rand(1, 4) + 0.5
    fn = PoissonMultinomialLoss(**KW)
    ref = fn(y_pred, y_true, tw)
    out = fn(
        y_pred, y_true, tw, bin_weights=torch.ones(3, 64), sample_weights=torch.ones(3)
    )
    assert torch.equal(out, ref)


def test_zero_bin_weight_removes_bin_from_gradient():
    y_pred, y_true = loss_data()
    w = torch.ones(3, 64)
    w[0, 10] = 0
    y_pred.requires_grad_(True)
    loss = PoissonMultinomialLoss(**KW)(y_pred, y_true, bin_weights=w)
    loss.backward()
    assert torch.all(y_pred.grad[0, :, 10] == 0)
    assert torch.all(y_pred.grad[1, :, 10] != 0)
    y_true2 = y_true.clone()
    y_true2[0, :, 10] = 1000
    loss2 = PoissonMultinomialLoss(**KW)(y_pred.detach(), y_true2, bin_weights=w)
    assert torch.allclose(loss.detach(), loss2)


def test_zero_sample_weight_removes_sequence_from_gradient():
    y_pred, y_true = loss_data()
    y_pred.requires_grad_(True)
    sw = torch.tensor([1.0, 0.0, 1.0])
    PoissonMultinomialLoss(**KW)(y_pred, y_true, sample_weights=sw).backward()
    assert torch.all(y_pred.grad[1] == 0)
    assert torch.all(y_pred.grad[0] != 0)


def test_fixed_divisor():
    """No per-batch normalization: scaling every sample weight scales the loss."""
    y_pred, y_true = loss_data()
    fn = PoissonMultinomialLoss(**KW)
    v = torch.tensor([1.0, 2.0, 0.5])
    assert torch.allclose(
        fn(y_pred, y_true, sample_weights=2 * v),
        2 * fn(y_pred, y_true, sample_weights=v),
    )
    assert torch.allclose(
        fn(y_pred, y_true, sample_weights=torch.full((3,), 5.0)), 5 * fn(y_pred, y_true)
    )


def test_per_track_bin_weights_match_shared():
    y_pred, y_true = loss_data()
    w = torch.rand(3, 64) * 2
    fn = PoissonMultinomialLoss(**KW)
    shared = fn(y_pred, y_true, bin_weights=w)
    assert torch.allclose(
        shared, fn(y_pred, y_true, bin_weights=w[:, None].expand(3, 4, 64))
    )


# -- weight files --------------------------------------------------------------


def test_label_bins_by_midpoint(tmp_path):
    _, _, cm = write_classes(
        tmp_path,
        [("chr1", 100, 164, "a"), ("chr1", 164, 200, "b")],
        {"bg": 1.0, "a": 2.0, "b": 3.0},
    )
    # bin mids 112, 144 -> a; 176 -> b; 208 -> bg
    assert cm.label_bins("chr1", 96, 4, 32).tolist() == [1, 1, 2, 0]
    assert cm.label_bins("chrX", 0, 2, 32).tolist() == [0, 0]
    cm.default_class = None
    with pytest.raises(ValueError):
        cm.label_bins("chrX", 0, 2, 32)


def test_overlapping_intervals_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlapping"):
        write_classes(
            tmp_path, [("chr1", 0, 100, "a"), ("chr1", 50, 150, "a")], {"bg": 1, "a": 2}
        )


def test_class_weight_normalization():
    w = normalize_class_weights(np.array([1.0, 3.0]), np.array([30, 10]))
    assert np.allclose(w, [1 / 1.5, 2.0])


def test_read_sample_weights(data_dir, tmp_path):
    ds = SeqDataset(str(data_dir), "train", fold=0)
    row = ds.seqs_bed.iloc[3]
    p = tmp_path / "sw.tsv"
    pd.DataFrame(
        {"chrom": [row.chrom], "start": [row.start], "end": [row.end], "weight": [4.0]}
    ).to_csv(p, sep="\t", index=False)
    sw = read_sample_weights(p, ds.seqs_bed)
    assert sw[3] == 4.0 and (np.delete(sw, 3) == 1).all()
    pd.DataFrame({"chrom": ["chr9"], "start": [0], "end": [1], "weight": [4.0]}).to_csv(
        p, sep="\t", index=False
    )
    with pytest.raises(ValueError):
        read_sample_weights(p, ds.seqs_bed)


# -- dataset -------------------------------------------------------------------


def test_seq_indexes_match_sequences_bed(data_dir):
    ds = SeqDataset(str(data_dir), "train", fold=0)
    bed = pd.read_csv(data_dir / "sequences.bed", sep="\t", header=None)
    # test fold 0, valid fold 1, train fold 2
    assert np.array_equal(ds.seq_indexes, np.flatnonzero(bed[3].to_numpy() == "fold2"))
    assert ds[0].seq_index.item() == ds.seq_indexes[0]


def test_labels_follow_sequence_coordinates(data_dir, tmp_path):
    row = SeqDataset(str(data_dir), "train", fold=0).seqs_bed
    row = row.iloc[SeqDataset(str(data_dir), "train", fold=0).seq_indexes[1]]
    _, _, cm = write_classes(
        tmp_path,
        [(row.chrom, row.start + 3 * POOL, row.start + 6 * POOL, "hot")],
        {"bg": 1, "hot": 5},
    )
    ds = SeqDataset(str(data_dir), "train", fold=0, class_map=cm)
    assert np.flatnonzero(ds.bin_labels(1)).tolist() == [3, 4, 5]
    assert not ds.bin_labels(0).any()


def test_explicit_shift_and_rc_flip_targets_and_labels(data_dir, tmp_path):
    base = SeqDataset(str(data_dir), "train", fold=0)
    row = base.seqs_bed.iloc[base.seq_indexes[2]]
    _, _, cm = write_classes(
        tmp_path,
        [(row.chrom, row.start, row.start + 2 * POOL, "hot")],
        {"bg": 1, "hot": 5},
    )
    ds = SeqDataset(str(data_dir), "train", mode="train", fold=0, class_map=cm)
    z = zarr.open(base.zarr_files[0], mode="r")
    seqi, tgt = z["sequence"][2], torch.from_numpy(z["target"][2])
    for shift in (-1, 0, 1):
        for rc in (False, True):
            b = ds[(2, shift, rc)]
            s = dna.seqi_shift(seqi.copy(), shift, 4)
            t = tgt
            if rc:
                s, t = dna.seqi_rc(s), torch.flip(tgt[base.strand_pair], [1])
            assert torch.equal(b.sequence, onehot(s)), (shift, rc)
            assert torch.equal(b.coverage_targets, t), (shift, rc)
            hot = np.flatnonzero(b.bin_labels.numpy()).tolist()
            assert hot == ([BINS - 2, BINS - 1] if rc else [0, 1])
    # eval mode ignores the augmentation request
    ev = SeqDataset(str(data_dir), "train", fold=0, class_map=cm)
    assert torch.equal(ev[(2, 1, True)].coverage_targets, tgt)


def test_collate_stacks_new_fields(data_dir, tmp_path):
    _, _, cm = write_classes(tmp_path, [("chr1", 0, 10, "hot")], {"bg": 1, "hot": 5})
    ds = SeqDataset(str(data_dir), "train", fold=0, class_map=cm)
    b = BatchData.collate([ds[0], ds[1]])
    assert b.seq_index.shape == (2,) and b.bin_labels.shape == (2, BINS)


# -- seeded sampler ------------------------------------------------------------


def batches(sampler, epoch):
    sampler.set_epoch(epoch)
    return [tuple(b) for b in sampler]


def test_seeded_sampler(data_dir):
    multi = MultiDataset([SeqDataset(str(data_dir), "*", mode="train", fold=0)])
    a, b, c = (MultiSampler(multi, 2, seed=s) for s in (7, 7, 8))
    assert batches(a, 0) == batches(b, 0)
    assert batches(a, 0) != batches(c, 0)
    assert batches(a, 0) != batches(a, 1)
    items = [it for batch in batches(a, 3) for it in batch]
    assert sorted(i for i, _, _ in items) == list(range(len(multi)))
    assert {s for _, s, _ in items} <= {-1, 0, 1}


def test_unseeded_sampler_unchanged(data_dir):
    multi = MultiDataset([SeqDataset(str(data_dir), "*", mode="train", fold=0)])
    np.random.seed(0)
    out = list(MultiSampler(multi, 2))
    assert all(isinstance(i, int) for batch in out for i in batch)


def test_seeded_loader_independent_of_workers(data_dir):
    multi = MultiDataset([SeqDataset(str(data_dir), "train", mode="train", fold=0)])

    def run(workers):
        s = MultiSampler(multi, 2, seed=3)
        s.set_epoch(0)
        dl = torch.utils.data.DataLoader(
            multi,
            batch_sampler=s,
            num_workers=workers,
            collate_fn=lambda items: BatchData.collate([it[1] for it in items]),
        )
        return [
            (
                b.seq_index.tolist(),
                b.sequence.sum().item(),
                b.coverage_targets.float().sum().item(),
            )
            for b in dl
        ]

    assert run(0) == run(2)


# -- end to end ----------------------------------------------------------------


def tiny_params(epochs=2):
    return {
        "train": {
            "batch_size": 2,
            "optimizer": "adam",
            "learning_rate": 0.002,
            "train_epochs_max": epochs,
            "loss": "poisson_mn",
            "total_weight": 0.25,
            "weight_exp": 8,
            "weight_range": 10,
            "num_workers": 0,
            "mix_dtype": "bfloat16",
            "seed": 0,
        },
        "model": {
            "seq_length": SEQ_LEN,
            "act_func": "gelu",
            "trunk": [
                {
                    "name": "ConvDNA",
                    "out_channels": 8,
                    "kernel_size": 5,
                    "pool_size": 2,
                },
                {
                    "name": "ConvTower",
                    "in_channels": 8,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "divisible_by": 8,
                    "pool_size": 2,
                    "repeat": 4,
                },
            ],
            "head_human": {
                "name": "Final",
                "in_channels": 16,
                "num_targets": NUM_TARGETS,
            },
        },
    }


def hot_weights(tmp_path, data_dir, hot):
    """Class 'hot' on the first half of every chr1 window."""
    bed = pd.read_csv(data_dir / "sequences.bed", sep="\t", header=None)
    rows = [
        (c, s, s + (e - s) // 2, "hot")
        for c, s, e, _ in bed.itertuples(index=False)
        if c == "chr1"
    ]
    bed_file, tsv, _ = write_classes(tmp_path, rows, {"bg": 1.0, "hot": hot})
    return {
        "bin_classes": str(bed_file),
        "class_weights": str(tsv),
        "default_class": "bg",
    }


def train(tmp_path, data_dir, name, seed, loss_weights=None, epochs=2):
    params = tiny_params(epochs)
    params["train"]["seed"] = seed
    if loss_weights is not None:
        params["train"]["loss_weights"] = loss_weights
    pf = tmp_path / f"{name}.json"
    pf.write_text(json.dumps(params))
    out = tmp_path / name
    hound_train.main(["-o", str(out), "--fold", "0", str(pf), str(data_dir)])
    return out


def model_state(out):
    return torch.load(out / "checkpoint.pth", map_location="cpu")["model"]


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_seeded_training_and_all_ones_weights(tmp_path, data_dir):
    a = model_state(train(tmp_path, data_dir, "a", 1))
    assert same(a, model_state(train(tmp_path, data_dir, "b", 1)))
    assert not same(a, model_state(train(tmp_path, data_dir, "c", 2)))
    ones = train(tmp_path, data_dir, "ones", 1, hot_weights(tmp_path, data_dir, 1.0))
    assert same(a, model_state(ones))
    hot = train(tmp_path, data_dir, "hot", 1, hot_weights(tmp_path, data_dir, 4.0))
    assert not same(a, model_state(hot))
    w = pd.read_csv(hot / "class_weights_ds0.tsv", sep="\t")
    assert abs((w.weight * w.train_bins).sum() / w.train_bins.sum() - 1) < 1e-9


class RecordingHook:
    """Records calls and scales the 'hot' weight at each epoch end."""

    calls: list = []  # noqa: RUF012

    def __init__(self, factor):
        self.factor = factor

    def on_train_start(self, trainer):
        RecordingHook.calls.append(("start",))

    def after_step(self, trainer, step):
        assert step.bin_labels.shape == (2, step.y_true.shape[-1])
        RecordingHook.calls.append(("step", step.epoch, step.step))

    def on_epoch_end(self, trainer, epoch):
        trainer.class_weights[0][1] *= self.factor
        RecordingHook.calls.append(("epoch", epoch))


def test_hook_and_resume_restore_weights(tmp_path, data_dir):
    lw = hot_weights(tmp_path, data_dir, 4.0)
    lw["hook"] = {"class": f"{__name__}:RecordingHook", "args": {"factor": 0.5}}
    RecordingHook.calls = []
    out = train(tmp_path, data_dir, "hook", 0, lw, epochs=2)
    calls = RecordingHook.calls
    assert calls[0] == ("start",)
    # 4 train sequences (fold 2), batch 2 -> 2 steps per epoch
    steps = [c for c in calls if c[0] == "step"]
    assert steps == [("step", 0, 0), ("step", 0, 1), ("step", 1, 2), ("step", 1, 3)]
    w0 = pd.read_csv(out / "class_weights_ds0.tsv", sep="\t").weight.to_numpy()
    state = torch.load(out / "loss_weights_state.pt")
    expected = torch.tensor([w0[0], w0[1] * 0.25], dtype=torch.float32)
    assert torch.allclose(state["class_weights"][0], expected)

    # resume for one more epoch: weights come back from the state file, no on_train_start
    RecordingHook.calls = []
    train(tmp_path, data_dir, "hook", 0, lw, epochs=3)
    assert ("start",) not in RecordingHook.calls
    state = torch.load(out / "loss_weights_state.pt")
    assert torch.allclose(
        state["class_weights"][0], expected * torch.tensor([1.0, 0.5])
    )
