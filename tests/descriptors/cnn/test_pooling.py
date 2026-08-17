import numpy as np
import pytest
import torch
from PIL import Image

from cbir.descriptors.cnn.pooling import CHANNELS, PooledCNN


@pytest.fixture(scope="module")
def model() -> PooledCNN:
    return PooledCNN("alexnet", p=3.0, device="cpu")


def _pooler(p):
    return PooledCNN.__new__(PooledCNN).__class__.pool.__get__(type("_", (), {"p": p})(), PooledCNN)


def test_p_of_one_is_average_pooling():
    # SPoC is not a separate method -- it is this module at p=1. If that identity broke,
    # we would silently be reporting something that is neither SPoC nor GeM.
    features = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) + 1
    pooled = _pooler(1.0)(features)
    torch.testing.assert_close(pooled, features.mean(dim=(2, 3)), rtol=1e-4, atol=1e-4)


def test_p_none_is_max_pooling():
    # MAC. Implemented as a true max rather than a large finite p, which overflows
    # float32 long before it converges on the maximum.
    features = torch.rand(2, 5, 3, 3)
    torch.testing.assert_close(_pooler(None)(features), features.amax(dim=(2, 3)))


def test_larger_p_moves_towards_the_max():
    # The generalized mean interpolates between average and max; that ordering is the
    # entire reason p is a knob.
    features = torch.rand(1, 4, 6, 6)
    mean = _pooler(1.0)(features)
    gem = _pooler(3.0)(features)
    mx = _pooler(None)(features)
    assert torch.all(mean <= gem + 1e-6)
    assert torch.all(gem <= mx + 1e-6)


def test_zero_activations_do_not_produce_nan():
    # Conv maps are post-ReLU, so exact zeros are common; 0 ** (1/p) is where a missing
    # clamp shows up as NaN rather than as an error.
    pooled = _pooler(3.0)(torch.zeros(1, 3, 4, 4))
    assert torch.isfinite(pooled).all()


def test_preprocess_caps_the_long_side_without_changing_aspect(model):
    tensor = model.preprocess(Image.new("RGB", (2048, 1024)))
    assert tensor.shape[-2:] == (512, 1024)  # halved, ratio preserved


def test_preprocess_never_enlarges_a_small_crop(model):
    # Query crops go down to 131 px on this benchmark. Upsampling them 8x adds no
    # information and only costs compute -- and a conv stack does not require it.
    tensor = model.preprocess(Image.new("RGB", (131, 90)))
    assert tensor.shape[-2:] == (90, 131)


def test_extract_handles_differently_sized_images(model):
    # The point of dropping the fully-connected layer: no fixed input size. These three
    # cannot be stacked into a batch, which is why extraction is one image per pass.
    sizes = [(300, 200), (64, 512), (131, 90)]
    out = model.extract([Image.new("RGB", s, "gray") for s in sizes])

    assert out.shape == (3, CHANNELS["alexnet"])
    assert np.linalg.norm(out, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_descriptor_is_far_smaller_than_a_neural_code(model):
    # 256-D natively against fc6's 4096-D, with no PCA involved.
    assert CHANNELS["alexnet"] == 256


def test_extract_of_nothing_is_empty(model):
    assert model.extract([]).shape == (0, CHANNELS["alexnet"])


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="unknown backbone"):
        PooledCNN("inception", device="cpu")  # type: ignore[arg-type]
