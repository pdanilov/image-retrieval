"""The loop's contracts: margins, checkpoint layout, and validation output.

The loop itself is not unit-tested — it is hours of GPU over 91k images, and a test that
faked all of that would be testing the fake. What *is* tested is everything around it
that can be wrong silently: the wrong margin for a backbone, a checkpoint the evaluation
path cannot read back, or a validation call that quietly reports nothing.
"""

from __future__ import annotations

import pytest
import torch

from cbir.training.loop import VAL_METRICS, TrainConfig, _save
from cbir.training.net import RetrievalNet


def test_each_backbone_resolves_to_its_published_margin():
    assert TrainConfig(backbone="resnet101").resolved_margin() == 0.85
    assert TrainConfig(backbone="vgg16").resolved_margin() == 0.7


def test_an_explicit_margin_overrides_the_published_one():
    assert TrainConfig(backbone="vgg16", margin=0.9).resolved_margin() == 0.9


def test_a_backbone_with_no_published_margin_refuses_to_guess():
    # Picking some other backbone's margin silently would be a different experiment
    # wearing this one's name.
    with pytest.raises(ValueError, match="no published margin"):
        TrainConfig(backbone="resnet18").resolved_margin()


def test_the_defaults_are_the_reference_command():
    # These are not choices made here, and a drift would be a silent divergence from the
    # recipe this branch exists to reproduce.
    config = TrainConfig()
    assert (config.image_size, config.neg_num) == (362, 5)
    assert (config.query_size, config.pool_size) == (2000, 22000)
    assert (config.lr, config.weight_decay, config.batch_size) == (5e-7, 1e-6, 5)


def test_a_checkpoint_we_write_is_one_the_eval_path_can_read(tmp_path):
    # The point of matching the reference's layout: a network trained here must be
    # loadable by the same code that loads the published checkpoints, or the two tiers
    # cannot be compared under one evaluation.
    from cbir.descriptors.cnn.finetuned import _rename

    model = RetrievalNet("vgg16")
    path = tmp_path / "check.pth"
    _save(path, model, TrainConfig(backbone="vgg16"), epoch=3, metrics={"mean_average_precision": 0.5})

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) == {"state_dict", "meta"}
    assert payload["meta"]["architecture"] == "vgg16"
    assert payload["meta"]["pooling"] == "gem"

    # The two things finetuned.load reads out of a state dict.
    assert "pool.p" in payload["state_dict"]
    renamed = {_rename(k, "vgg16") for k in payload["state_dict"] if k.startswith("features.")}
    assert renamed == set(RetrievalNet("vgg16").features.state_dict())


def test_the_optimizer_is_saved_only_for_the_resumable_checkpoint(tmp_path):
    # `best.pth` is for evaluation and `last.pth` for resuming; carrying Adam's state in
    # both would double the size of every published checkpoint for nothing.
    model = RetrievalNet("vgg16")
    optimizer = torch.optim.Adam(model.parameters())

    _save(tmp_path / "best.pth", model, TrainConfig(), 1, {})
    _save(tmp_path / "last.pth", model, TrainConfig(), 1, {}, optimizer)

    assert "optimizer" not in torch.load(tmp_path / "best.pth", map_location="cpu", weights_only=False)
    assert "optimizer" in torch.load(tmp_path / "last.pth", map_location="cpu", weights_only=False)


def test_validation_reports_a_retrieval_metric_not_a_loss():
    # Validation loss falls as mined negatives get harder as well as when the model
    # improves, so it is not comparable across epochs. mAP is.
    assert "mean_average_precision" in VAL_METRICS
    assert "precision_at_1" in VAL_METRICS


def test_validation_uses_exact_search_not_faiss():
    # AccuracyCalculator defaults to FaissKNN. AGENTS.md keeps search exact and in torch,
    # and faiss is not a dependency of this project.
    import inspect

    from cbir.training import loop

    source = inspect.getsource(loop.validate)
    assert "CustomKNN" in source
    assert "faiss" not in source.lower() or "not the FaissKNN default" in source


def test_training_starts_from_the_weights_the_reference_starts_from():
    # The reference fills the architecture from its Caffe-converted ImageNet weights
    # whenever it has them -- true for all three trainable backbones -- so the published
    # numbers were reached from a Caffe init. Defaulting to torchvision would diverge
    # from the recipe before the first step, and the frozen tier measured that gap at
    # 0.347 against 0.461 for resnet101 GeM.
    from cbir.descriptors.cnn.weights import CAFFE_FILES

    assert TrainConfig().weights == "caffe"
    for backbone in ("vgg16", "resnet50", "resnet101"):
        assert backbone in CAFFE_FILES, f"{backbone} has no caffe weights to start from"


def test_the_weight_source_reaches_the_network(monkeypatch):
    # `weights` had no CLI flag at all for a while, and TrainConfig carried a default
    # that never matched the recipe. Parameterized knobs that never reach the model are
    # this project's recurring bug.
    seen = {}

    def _record(backbone, **kwargs):
        seen.update(kwargs | {"backbone": backbone})
        raise RuntimeError("stop before the GPU")

    from cbir.training import loop

    monkeypatch.setattr(loop, "RetrievalNet", _record)
    with pytest.raises(RuntimeError, match="stop before"):
        loop.train(TrainConfig(backbone="resnet101", weights="torchvision"), device="cpu", log=lambda *_: None)

    assert seen["backbone"] == "resnet101"
    assert seen["weights"] == "torchvision"


def test_a_finetuned_checkpoint_cannot_be_a_starting_point():
    # sfm120k weights are already the output of this process; training from them would
    # be a second fine-tune wearing the first one's name.
    with pytest.raises(ValueError, match="already fine-tuned"):
        RetrievalNet("vgg16", weights="sfm120k")


# --- heartbeat, which the container healthcheck reads -------------------------------


def test_the_heartbeat_records_every_log_line(tmp_path):
    from cbir.training.loop import _with_heartbeat

    seen = []
    logged = _with_heartbeat(seen.append, tmp_path / "heartbeat")

    logged("epoch 3/30: loss 0.4")

    assert seen == ["epoch 3/30: loss 0.4"]  # the original log still happens
    written = (tmp_path / "heartbeat").read_text()
    assert "epoch 3/30" in written
    assert float(written.split()[0]) > 0  # a timestamp the probe can age


def test_the_heartbeat_is_replaced_atomically(tmp_path):
    # A probe reading mid-write would otherwise see a truncated file and, if it parsed
    # the contents, call a healthy run dead.
    from cbir.training.loop import _with_heartbeat

    path = tmp_path / "heartbeat"
    logged = _with_heartbeat(lambda _: None, path)
    logged("first")
    logged("second")

    assert "second" in path.read_text()
    assert not list(tmp_path.glob("*.tmp"))  # nothing left behind


def test_an_unwritable_heartbeat_does_not_kill_the_run(tmp_path, monkeypatch):
    # A read-only mount or a full disk must not end a run that is otherwise healthy --
    # the heartbeat is diagnostics, not the work.
    from cbir.training.loop import _with_heartbeat

    seen = []
    logged = _with_heartbeat(seen.append, tmp_path / "nope" / "heartbeat")
    monkeypatch.setattr("pathlib.Path.mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))

    logged("still training")

    assert seen == ["still training"]
