"""The fine-tuned checkpoints, and the three separate things each one carries.

The checkpoint files are a manual download, so every test that needs the real bytes is
skipped without them. What can be tested unconditionally is the part that would fail
silently: a config that asks for a trained network *and* a hand-picked pooling, or for
the learned projection without the checkpoint that holds it, would otherwise run for an
hour and record a row describing an experiment that never happened.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbir.configs.cnn import PooledConfig, RMACConfig
from cbir.descriptors.cnn import finetuned
from cbir.descriptors.cnn.pooling import MULTI_SCALE, PooledCNN

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _available(backbone: str) -> bool:
    return (finetuned.root() / finetuned.FILES[backbone]).exists()


needs_checkpoint = pytest.mark.skipif(not _available("vgg16"), reason="fine-tuned vgg16 checkpoint not downloaded")


# --- Configs that describe no experiment ------------------------------------------


def test_trained_weights_require_the_trained_exponent():
    # Running fine-tuned weights at p=3.0 evaluates the network at a pooling it was
    # never trained for -- a plausible-looking wrong number, which is the failure mode
    # worth spending a raise on.
    with pytest.raises(ValueError, match="go together"):
        PooledConfig(backbone="vgg16", weights="sfm120k", p=3.0, last_pool=False)


def test_the_trained_exponent_requires_trained_weights():
    with pytest.raises(ValueError, match="go together"):
        PooledConfig(backbone="vgg16", p="learned")


def test_learned_whitening_requires_the_checkpoint_that_carries_it():
    with pytest.raises(ValueError, match="fine-tuned checkpoint"):
        PooledConfig(backbone="vgg16", whiten=True, whiten_source="learned")


def test_learned_whitening_with_whitening_off_is_a_contradiction():
    with pytest.raises(ValueError, match="fit nothing and apply nothing"):
        PooledConfig(backbone="vgg16", weights="sfm120k", p="learned", last_pool=False, whiten_source="learned")


def test_shrinkage_is_meaningless_against_a_projection_that_is_not_fitted():
    with pytest.raises(ValueError, match="not fitted here"):
        PooledConfig(
            backbone="vgg16",
            weights="sfm120k",
            p="learned",
            last_pool=False,
            whiten=True,
            whiten_source="learned",
            shrinkage=0.1,
        )


def test_rmac_has_no_fine_tuned_checkpoint():
    # Only GeM was published. Caught at the config so the message names the reason
    # rather than surfacing as a missing file.
    with pytest.raises(ValueError, match="GeM only"):
        RMACConfig(backbone="vgg16", weights="sfm120k")
    with pytest.raises(ValueError, match="GeM only"):
        RMACConfig(backbone="vgg16", whiten=True, whiten_source="learned")


def test_the_valid_combination_is_accepted():
    config = PooledConfig(
        backbone="vgg16", weights="sfm120k", p="learned", last_pool=False, whiten=True, whiten_source="learned"
    )
    assert config.p == "learned"


def test_no_checkpoint_for_a_backbone_says_so():
    with pytest.raises(ValueError, match="no fine-tuned checkpoint"):
        finetuned.load("alexnet")


# --- The real files ----------------------------------------------------------------


@needs_checkpoint
def test_the_checkpoint_supplies_an_exponent_the_config_did_not():
    # The published number rests on this: p is differentiable, so it was trained, and it
    # is not the 3.0 anyone would have picked.
    checkpoint = finetuned.load("vgg16")

    assert checkpoint.p == pytest.approx(2.9208, abs=1e-3)
    assert checkpoint.p != 3.0


@needs_checkpoint
def test_conv_weights_land_in_the_architecture_without_remapping():
    # Silent failure here is a half-initialized network: torchvision weights in the
    # layers whose names did not match, fine-tuned ones in the rest.
    model = PooledCNN("vgg16", weights="sfm120k", p="learned", last_pool=False)

    assert model.finetuned is not None
    assert model.p == pytest.approx(2.9208, abs=1e-3)


@needs_checkpoint
def test_the_scale_variant_follows_the_run_rather_than_being_a_knob():
    # `ss` and `ms` are fitted for single- and multi-scale extraction and are not
    # interchangeable; deriving the choice from `scales` makes the two impossible to
    # contradict.
    checkpoint = finetuned.load("vgg16")

    single = checkpoint.whitening((1.0,)).components
    multi = checkpoint.whitening(MULTI_SCALE).components

    assert single.shape == multi.shape == (512, 512)
    assert not np.allclose(single, multi)


@needs_checkpoint
def test_learned_whitening_normalizes_what_it_transforms():
    checkpoint = finetuned.load("vgg16")
    descriptors = np.random.default_rng(0).normal(size=(4, 512)).astype(np.float32)

    out = checkpoint.whitening((1.0,)).transform(descriptors)

    assert out.shape == (4, 512)
    assert np.linalg.norm(out, axis=1) == pytest.approx(1.0, abs=1e-5)


@needs_checkpoint
def test_a_shorter_width_truncates_the_projection_rather_than_refitting():
    checkpoint = finetuned.load("vgg16")

    full = checkpoint.whitening((1.0,)).components
    short = checkpoint.whitening((1.0,), dim=128).components

    assert short.shape == (128, 512)
    assert np.array_equal(short, full[:128])


@needs_checkpoint
def test_keeping_vggs_trailing_pool_is_refused():
    # The checkpoint was trained on `features[:-1]`. Keeping the pool quarters the conv
    # map the fine-tuned weights see, and nothing about the result would look wrong.
    with pytest.raises(ValueError, match="last_pool=False"):
        PooledCNN("vgg16", weights="sfm120k", p="learned", last_pool=True)


# --- local checkpoints, i.e. our own training output -------------------------------


@pytest.fixture
def local_checkpoint(tmp_path):
    """A checkpoint in the shape `cbir train` writes, with an identifiable `p`."""
    import torch

    from cbir.training.loop import TrainConfig, _save
    from cbir.training.net import RetrievalNet

    net = RetrievalNet("vgg16", weights="torchvision")
    with torch.no_grad():
        net.pool.p.fill_(2.5)
    path = tmp_path / "best.pth"
    _save(path, net, TrainConfig(backbone="vgg16", weights="torchvision"), epoch=7, metrics={})
    return path, net


def test_a_trained_checkpoint_loads_through_the_published_loader(local_checkpoint):
    # The reason training writes the reference's layout: one loader, not two that drift.
    path, net = local_checkpoint

    loaded = finetuned.load("vgg16", str(path))

    assert loaded.p == pytest.approx(2.5)
    assert set(loaded.state) == set(net.features.state_dict())


def test_the_trained_exponent_survives_the_round_trip(local_checkpoint):
    # `p` is trained, so evaluating at the config's 3.0 instead would score the network
    # at a pooling it was never trained for -- the whole reason `p="learned"` exists.
    path, _ = local_checkpoint

    model = PooledCNN("vgg16", p="learned", weights="sfm120k", last_pool=False, checkpoint=str(path))

    assert model.p == pytest.approx(2.5)


def test_a_trained_checkpoint_has_no_learned_whitening_and_says_so(local_checkpoint):
    # The published checkpoints ship a supervised projection fitted after training; ours
    # does not, because learning it is a step this project has not built. The message has
    # to name the way out, or the run just fails.
    path, _ = local_checkpoint

    loaded = finetuned.load("vgg16", str(path))

    assert loaded._whitening is None
    with pytest.raises(ValueError, match="held_out"):
        loaded.whitening((1.0,))


def test_a_checkpoint_path_requires_the_finetuned_weight_source():
    with pytest.raises(ValueError, match="needs weights='sfm120k'"):
        PooledConfig(backbone="vgg16", checkpoint="somewhere.pth")


def test_a_missing_checkpoint_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        finetuned.load("vgg16", str(tmp_path / "absent.pth"))


def test_a_checkpoint_for_another_architecture_is_refused(local_checkpoint):
    # meta['architecture'] is checked, so a vgg16 checkpoint cannot be loaded into a
    # resnet and half-fill it.
    path, _ = local_checkpoint

    with pytest.raises(ValueError, match="is a vgg16 checkpoint"):
        finetuned.load("resnet101", str(path))


def test_rmac_refuses_a_checkpoint_too():
    # RMACConfig has no `checkpoint` field, so that misuse cannot be spelled. The model
    # takes **kwargs, though, and would forward one straight into PooledCNN.
    from cbir.descriptors.cnn.rmac import RMAC

    assert "checkpoint" not in RMACConfig.__dataclass_fields__
    with pytest.raises(ValueError, match="GeM only"):
        RMAC("vgg16", checkpoint="x.pth")
