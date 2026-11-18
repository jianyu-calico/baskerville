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
    check_class_coverage,
    class_bin_counts,
    class_bin_counts_by_split,
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


def make_data(root, crop_bp=0):
    """3 folds x 4 windows on chr1/chr2; folds interleaved in sequences.bed.

    As hound_data writes it, sequences.bed holds the target region and the stored DNA
    extends crop_bp past it on each side.
    """
    rng = np.random.default_rng(0)
    (root / "examples").mkdir(parents=True)
    seq_len = SEQ_LEN + 2 * crop_bp
    rows = []
    for k in range(NUM_FOLDS):
        for i in range(WINDOWS_PER_FOLD):
            start = 10_000 * (k * WINDOWS_PER_FOLD + (WINDOWS_PER_FOLD - 1 - i))
            chrom = "chr1" if i % 2 == 0 else "chr2"
            rows.append((chrom, start, start + SEQ_LEN, f"fold{k}"))
    order = [r for i in range(WINDOWS_PER_FOLD) for r in rows[i::WINDOWS_PER_FOLD]]
    pd.DataFrame(order).to_csv(
        root / "sequences.bed", sep="\t", header=False, index=False
    )
    for k in range(NUM_FOLDS):
        g = zarr.open_group(str(root / "examples" / f"fold{k}.zarr"), mode="w")
        n = WINDOWS_PER_FOLD
        seqs = rng.integers(0, 4, size=(n, seq_len), dtype=np.uint8)
        g.create_array("sequence", data=seqs)
        targets = rng.gamma(1.0, 2.0, size=(n, NUM_TARGETS, BINS)).astype(np.float16)
        g.create_array("target", data=targets)
    pd.DataFrame(
        {
            "identifier": ["a+", "a-", "b", "c"],
            "description": ["RNA:a", "RNA:a", "DNASE:b", "CHIP:c"],
            "assay": ["rna", "rna", "dnase", "chip"],
            "strand_pair": [1, 0, 2, 3],
        }
    ).to_csv(root / "targets.txt", sep="\t")
    stats = {
        "num_targets": NUM_TARGETS,
        "seq_length": seq_len,
        "pool_width": POOL,
        "crop_bp": crop_bp,
        "target_length": BINS,
    }
    (root / "statistics.json").write_text(json.dumps(stats))
    return root


@pytest.fixture
def data_dir(tmp_path):
    return make_data(tmp_path / "data")


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
    w = normalize_class_weights(np.array([[1.0], [3.0]]), np.array([30, 10]))
    assert np.allclose(w[:, 0], [1 / 1.5, 2.0])
    # per group: each column has mean 1 over the training bins
    w = normalize_class_weights(np.array([[1.0, 2.0], [3.0, 0.0]]), np.array([30, 10]))
    assert np.allclose((w * np.array([30, 10])[:, None]).sum(axis=0) / 40, [1.0, 1.0])
    assert np.allclose(w, [[1 / 1.5, 4 / 3], [2.0, 0.0]])


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


@pytest.mark.parametrize(
    "crop_bp, extra_crop_bins, hot", [(64, None, [3, 4, 5]), (64, 2, [1, 2, 3])]
)
def test_labels_with_cropping(tmp_path, crop_bp, extra_crop_bins, hot):
    """sequences.bed is the target region, so crop_bp doesn't move the bins."""
    data = make_data(tmp_path / "data", crop_bp=crop_bp)
    base = SeqDataset(str(data), "train", fold=0, extra_crop_bins=extra_crop_bins)
    row = base.seqs_bed.iloc[base.seq_indexes[1]]
    _, _, cm = write_classes(
        tmp_path,
        [(row.chrom, row.start + 3 * POOL, row.start + 6 * POOL, "hot")],
        {"bg": 1, "hot": 5},
    )
    ds = SeqDataset(
        str(data), "train", fold=0, class_map=cm, extra_crop_bins=extra_crop_bins
    )
    labels = ds[1].bin_labels.numpy()
    assert len(labels) == ds[1].coverage_targets.shape[-1]
    assert np.flatnonzero(labels).tolist() == hot


def test_chromosome_name_mismatch_raises(tmp_path, data_dir):
    bed = pd.read_csv(data_dir / "sequences.bed", sep="\t", header=None)
    rows = [
        (c.removeprefix("chr"), s, e, "hot")
        for c, s, e, _ in bed.itertuples(index=False)
    ]
    bed_file, tsv, _ = write_classes(tmp_path, rows, {"bg": 1.0, "hot": 4.0})
    lw = {
        "bin_classes": str(bed_file),
        "class_weights": str(tsv),
        "default_class": "bg",
    }
    with pytest.raises(ValueError, match="chromosome"):
        train(tmp_path, data_dir, "mismatch", 0, lw, epochs=1)


def window(data_dir, label):
    """The first sequences.bed row of a fold (fold0 test, fold1 valid, fold2 train)."""
    bed = SeqDataset(str(data_dir), "train", fold=0).seqs_bed
    return bed[bed.label == label].iloc[0]


def test_heldout_only_class_warns_and_counts_by_split(tmp_path, data_dir, capsys):
    tr, te = window(data_dir, "fold2"), window(data_dir, "fold0")
    _, _, cm = write_classes(
        tmp_path,
        [
            (tr.chrom, tr.start, tr.start + 64, "hot"),
            (te.chrom, te.start, te.start + 96, "test_only"),
        ],
        {"bg": 1, "hot": 2, "test_only": 3},
    )
    ds = SeqDataset(str(data_dir), "train", fold=0, class_map=cm)
    by_split = class_bin_counts_by_split(ds)
    assert np.array_equal(by_split["train"], class_bin_counts(ds))
    assert by_split["train"].tolist()[1:] == [2, 0]
    assert by_split["test"].tolist()[1:] == [0, 3]
    assert by_split["valid"].tolist()[1:] == [0, 0]
    assert check_class_coverage(ds, by_split) == ["test_only"]
    assert "only in valid/test" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["missing_from_bed", "short_intervals", "misnamed"])
def test_weighted_class_without_bins_anywhere_raises(tmp_path, data_dir, case):
    tr = window(data_dir, "fold2")
    rows = [(tr.chrom, tr.start, tr.start + 64, "hot")]
    if case == "short_intervals":  # 10 bp, never holding a bin midpoint (offset 16)
        rows.append((tr.chrom, tr.start + 64, tr.start + 74, "lost"))
    if case == "misnamed":
        rows.append((tr.chrom.removeprefix("chr"), tr.start, tr.start + 64, "lost"))
    _, _, cm = write_classes(tmp_path, rows, {"bg": 1, "hot": 2, "lost": 3})
    ds = SeqDataset(str(data_dir), "train", fold=0, class_map=cm)
    with pytest.raises(ValueError, match="no bins in any"):
        check_class_coverage(ds, class_bin_counts_by_split(ds))


def test_partly_misnamed_class_is_visible(tmp_path, data_dir, capsys):
    """Half the intervals on '1' instead of 'chr1': warned, and visible in the counts."""
    bed = SeqDataset(str(data_dir), "train", fold=0).seqs_bed
    rows = [
        (c if i % 2 else c.removeprefix("chr"), s0, s0 + 64, "hot")
        for i, (c, s0) in enumerate(zip(bed.chrom, bed.start))
    ]
    _, _, cm = write_classes(tmp_path, rows, {"bg": 1, "hot": 2})
    ds = SeqDataset(str(data_dir), "train", fold=0, class_map=cm)
    by_split = class_bin_counts_by_split(ds)
    check_class_coverage(ds, by_split)
    assert "not in sequences.bed" in capsys.readouterr().out
    assert sum(by_split.values())[1] == 2 * len(bed) // 2  # half the windows lost


def test_loss_weights_refuse_several_data_dirs(tmp_path, data_dir):
    params = tiny_params(1)
    params["train"]["loss_weights"] = hot_weights(tmp_path, data_dir, 2.0)
    pf = tmp_path / "multi.json"
    pf.write_text(json.dumps(params))
    with pytest.raises(SystemExit):
        hound_train.main(
            [
                "-o",
                str(tmp_path / "multi"),
                "--fold",
                "0",
                str(pf),
                str(data_dir),
                str(data_dir),
            ]
        )


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
    expected = torch.tensor([[w0[0]], [w0[1] * 0.25]], dtype=torch.float32)
    assert torch.allclose(state["class_weights"][0], expected)

    # resume for one more epoch: weights come back from the state file, no on_train_start
    RecordingHook.calls = []
    train(tmp_path, data_dir, "hook", 0, lw, epochs=3)
    assert ("start",) not in RecordingHook.calls
    state = torch.load(out / "loss_weights_state.pt")
    assert torch.allclose(
        state["class_weights"][0], expected * torch.tensor([[1.0], [0.5]])
    )


# -- per-group (per-assay) class weights ------------------------------------------


def write_group_table(tmp_path, table):
    """table: {class: {group: weight}} -> per-group class-weight TSV."""
    tsv = tmp_path / "group_weights.tsv"
    df = pd.DataFrame(table).T.rename_axis("class").reset_index()
    df.to_csv(tsv, sep="\t", index=False)
    return tsv


def group_class_map(tmp_path, data_dir, table):
    """Class 'hot' on bins 0-1 of every training window, 'bg' elsewhere."""
    ds = SeqDataset(str(data_dir), "train", fold=0)
    rows = []
    for i in range(len(ds)):
        r = ds.seqs_bed.iloc[ds.seq_indexes[i]]
        rows.append((r.chrom, r.start, r.start + 2 * POOL, "hot"))
    bed = tmp_path / "classes.bed"
    pd.DataFrame(rows).to_csv(bed, sep="\t", header=False, index=False)
    tsv = write_group_table(tmp_path, table)
    return bed, tsv, RegionClassMap.from_files(bed, tsv, "bg")


def test_group_table_parsing_and_track_groups(tmp_path, data_dir):
    table = {
        "bg": {"rna": 1, "dnase": 1, "chip": 1},
        "hot": {"rna": 3, "dnase": 0, "chip": 1},
    }
    _, _, cm = group_class_map(tmp_path, data_dir, table)
    assert cm.group_names == ["rna", "dnase", "chip"]
    assert cm.class_weights.shape == (2, 3)
    targets = SeqDataset(str(data_dir), "train", fold=0).targets_df
    assert cm.track_groups(targets).tolist() == [0, 0, 1, 2]
    # merging assays into groups, and a missing group column
    merged = {"bg": {"rna": 1, "open": 1}, "hot": {"rna": 2, "open": 2}}
    _, _, cm2 = group_class_map(tmp_path, data_dir, merged)
    groups = cm2.track_groups(targets, group_map={"dnase": "open", "chip": "open"})
    assert groups.tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="without a class-weight column"):
        cm2.track_groups(targets)
    # a shared table puts every track in group 0
    _, _, shared = write_classes(
        tmp_path, [("chr1", 0, 64, "hot")], {"bg": 1, "hot": 2}
    )
    assert shared.group_names == [None]
    assert shared.track_groups(targets).tolist() == [0] * NUM_TARGETS


def lookup_trainer(class_map, data_dir):
    """A Trainer holding only the loss-weight state, to test the per-batch lookup."""
    from baskerville.trainer import Trainer

    ds = SeqDataset(str(data_dir), "train", mode="train", fold=0, class_map=class_map)
    t = Trainer.__new__(Trainer)
    t.device = "cpu"
    t.class_weights = [torch.tensor(class_map.class_weights, dtype=torch.float32)]
    t.sample_weights = [None]
    groups = class_map.track_groups(ds.targets_df)
    t.track_groups = [torch.tensor(groups) if class_map.num_groups > 1 else None]
    return t, ds


def test_group_lookup_orientation_and_rc(tmp_path, data_dir):
    table = {
        "bg": {"rna": 1, "dnase": 1, "chip": 1},
        "hot": {"rna": 3, "dnase": 0, "chip": 2},
    }
    _, _, cm = group_class_map(tmp_path, data_dir, table)
    t, ds = lookup_trainer(cm, data_dir)
    for rc in (False, True):
        batch = BatchData.collate([ds[(0, 0, rc)], ds[(1, 0, rc)]])
        _, bin_w, _ = t._batch_loss_weights(0, batch)
        assert bin_w.shape == (2, NUM_TARGETS, BINS)
        hot = [BINS - 2, BINS - 1] if rc else [0, 1]
        others = [b for b in range(BINS) if b not in hot]
        # tracks 0,1 rna (3), track 2 dnase (0), track 3 chip (2); other bins 1
        for track, w in enumerate([3.0, 3.0, 0.0, 2.0]):
            assert torch.all(bin_w[:, track, hot] == w)
            assert torch.all(bin_w[:, track, others] == 1.0)


def test_group_zero_weight_leaves_other_groups_untouched(tmp_path, data_dir):
    table = {
        "bg": {"rna": 1, "dnase": 1, "chip": 1},
        "hot": {"rna": 1, "dnase": 0, "chip": 1},
    }
    _, _, cm = group_class_map(tmp_path, data_dir, table)
    t, ds = lookup_trainer(cm, data_dir)
    batch = BatchData.collate([ds[(0, 0, False)], ds[(1, 0, False)]])
    _, bin_w, _ = t._batch_loss_weights(0, batch)
    y_true = batch.coverage_targets.float()
    fn = PoissonMultinomialLoss(**KW)

    def grad(weights):
        y_pred = (y_true + 0.5).clone().requires_grad_(True)
        fn(y_pred, y_true, bin_weights=weights).backward()
        return y_pred.grad

    g, g_ref = grad(bin_w), grad(torch.ones_like(bin_w))
    assert torch.equal(g[:, [0, 1, 3]], g_ref[:, [0, 1, 3]])  # rna, chip unchanged
    assert torch.all(g[:, 2, :2] == 0)  # dnase hot bins removed
    assert not torch.equal(g[:, 2], g_ref[:, 2])


def test_all_ones_group_table_equals_unweighted_training(tmp_path, data_dir):
    a = model_state(train(tmp_path, data_dir, "a", 1))
    table = {
        "bg": {"rna": 1, "dnase": 1, "chip": 1},
        "hot": {"rna": 1, "dnase": 1, "chip": 1},
    }
    bed, tsv, _ = group_class_map(tmp_path, data_dir, table)
    lw = {"bin_classes": str(bed), "class_weights": str(tsv), "default_class": "bg"}
    assert same(a, model_state(train(tmp_path, data_dir, "group_ones", 1, lw)))
    table["hot"]["rna"] = 4
    lw["class_weights"] = str(write_group_table(tmp_path, table))
    hot = train(tmp_path, data_dir, "group_hot", 1, lw)
    assert not same(a, model_state(hot))
    w = pd.read_csv(hot / "class_weights_ds0.tsv", sep="\t")
    for g in ("rna", "dnase", "chip"):  # mean 1 within each group
        mean = (w[f"weight_{g}"] * w.train_bins).sum() / w.train_bins.sum()
        assert abs(mean - 1) < 1e-9


def test_single_group_column_is_validated(tmp_path, data_dir):
    """A table with one group column is per-group, not shared: every track needs it."""
    table = {"bg": {"dnase": 1}, "hot": {"dnase": 3}}
    bed, tsv, cm = group_class_map(tmp_path, data_dir, table)
    assert not cm.shared and cm.num_groups == 1
    targets = SeqDataset(str(data_dir), "train", fold=0).targets_df
    with pytest.raises(ValueError, match="without a class-weight column"):
        cm.track_groups(targets)
    lw = {"bin_classes": str(bed), "class_weights": str(tsv), "default_class": "bg"}
    with pytest.raises(ValueError, match="without a class-weight column"):
        train(tmp_path, data_dir, "one_group", 0, lw, epochs=1)
    # mapping every assay into that group makes it valid
    lw["group_map"] = {"rna": "dnase", "chip": "dnase"}
    out = train(tmp_path, data_dir, "one_group_mapped", 0, lw, epochs=1)
    assert "weight_dnase" in pd.read_csv(out / "class_weights_ds0.tsv", sep="\t")


def test_group_table_checks(tmp_path, data_dir, capsys):
    tsv = tmp_path / "dup.tsv"
    tsv.write_text("class\tdnase\tdnase\nbg\t1\t1\nhot\t2\t2\n")
    with pytest.raises(ValueError, match="duplicate columns"):
        RegionClassMap.from_files(tmp_path / "missing.bed", tsv, "bg")
    table = {
        "bg": {"rna": 1, "dnase": 1, "chip": 1, "cage": 1},
        "hot": {"rna": 2, "dnase": 2, "chip": 2, "cage": 2},
    }
    _, _, cm = group_class_map(tmp_path, data_dir, table)
    ds = SeqDataset(str(data_dir), "train", fold=0)
    cm.track_groups(ds.targets_df, strand_pair=ds.strand_pair)
    assert "no tracks: ['cage']" in capsys.readouterr().out
    # strand partners a+ / a- (tracks 0, 1) split across groups
    with pytest.raises(ValueError, match="strand partners"):
        cm.track_groups(
            ds.targets_df,
            group_by="identifier",
            group_map={"a+": "rna", "a-": "dnase", "b": "dnase", "c": "chip"},
            strand_pair=ds.strand_pair,
        )


def test_group_options_with_shared_table_warn(tmp_path, data_dir, capsys):
    lw = hot_weights(tmp_path, data_dir, 2.0)
    lw["group_by"] = "assay"
    train(tmp_path, data_dir, "shared_group_by", 0, lw, epochs=1)
    assert "ignored for a shared table" in capsys.readouterr().out
