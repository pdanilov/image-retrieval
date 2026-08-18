import numpy as np
import pytest
import torch
from PIL import Image

from cbir.descriptors.cnn.pooling import CHANNELS, MIN_SIDE, MULTI_SCALE, PooledCNN


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


def test_multi_scale_is_off_by_default(model):
    assert model.scales == (1.0,)


def test_scale_shrinks_the_input_below_the_cap(model):
    tensor = model.preprocess(Image.new("RGB", (400, 200)), scale=0.5)
    assert tensor.shape[-2:] == (100, 200)


def test_scale_composes_with_the_cap(model):
    # The cap applies first, then the scale factor multiplies what survived it -- so a
    # huge image at scale 1/2 is half of max_side, not half of its original size.
    tensor = model.preprocess(Image.new("RGB", (4096, 2048)), scale=0.5)
    assert tensor.shape[-2:] == (256, 512)


def test_scale_does_not_shrink_below_what_the_backbone_accepts(model):
    # A narrow query crop at scale 1/2 goes under AlexNet's 63-px floor and the forward
    # pass raises inside a max-pool. The floor keeps the aspect ratio and lifts both
    # sides together rather than padding.
    tensor = model.preprocess(Image.new("RGB", (131, 90)), scale=0.5)
    assert min(tensor.shape[-2:]) == MIN_SIDE["alexnet"]
    assert tensor.shape[-2:] == (63, 92)  # 131/90 preserved


def test_a_narrow_crop_survives_every_published_scale():
    # The actual crash: this ran fine single-scale, so nothing caught it until a
    # fractional scale was requested.
    multi = PooledCNN("alexnet", p=None, scales=MULTI_SCALE, device="cpu")
    assert multi.extract([Image.new("RGB", (131, 40), "gray")]).shape == (1, CHANNELS["alexnet"])


def test_the_floor_leaves_full_scale_alone(model):
    # It must only ever trigger below scale 1, or the single-scale rows already in
    # runs.jsonl would stop reproducing.
    tensor = model.preprocess(Image.new("RGB", (131, 90)))
    assert tensor.shape[-2:] == (90, 131)


def test_multi_scale_descriptor_keeps_the_native_width(model):
    # Averaging across scales, not concatenating: three passes still yield one C-vector,
    # so a multi-scale row stays comparable with a single-scale one at the same width.
    multi = PooledCNN("alexnet", p=3.0, scales=MULTI_SCALE, device="cpu")
    out = multi.extract([Image.new("RGB", (300, 200), "gray")])

    assert out.shape == (1, CHANNELS["alexnet"])
    assert np.linalg.norm(out, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_single_scale_multi_scale_agrees_with_plain_extraction(model):
    # scales=(1.0,) must be exactly the old path, or every descriptor already recorded
    # would silently stop being reproducible.
    image = Image.new("RGB", (200, 150))
    image.putpixel((10, 10), (255, 0, 0))
    explicit = PooledCNN("alexnet", p=3.0, scales=(1.0,), device="cpu")
    np.testing.assert_allclose(model.extract([image]), explicit.extract([image]), atol=1e-6)


def test_multi_scale_changes_the_descriptor(model):
    # Guards the wiring: if `scales` were accepted but ignored, every test above would
    # still pass while the run reported single-scale numbers under a multi-scale name.
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (150, 200, 3), dtype=np.uint8))
    multi = PooledCNN("alexnet", p=3.0, scales=MULTI_SCALE, device="cpu")

    similarity = float((model.extract([image]) @ multi.extract([image]).T).item())
    assert similarity < 0.999


def _fake_multi_scale(p, vectors):
    """A PooledCNN whose forward pass is bypassed, yielding `vectors` one per scale."""
    supply = iter(vectors)
    model = PooledCNN.__new__(PooledCNN)
    model.scales, model.device, model.max_side, model.p, model.backbone = (1.0, 0.5), "cpu", 1024, p, "alexnet"
    model._model = lambda tensor: tensor
    model.pool = lambda features: torch.from_numpy(next(supply))
    return model


def test_mac_combines_scales_by_plain_averaging():
    # Radenovic et al. average across scales for every method except GeM. MAC is one of
    # the "every other", and it is the row we reproduce to within 0.7 mAP.
    a = np.array([[1.0, 0.0]], np.float32)
    b = np.array([[0.0, 1.0]], np.float32)
    out = _fake_multi_scale(None, [a, b]).describe(Image.new("RGB", (8, 8)))
    np.testing.assert_allclose(out, [[0.5, 0.5]], atol=1e-6)


def test_gem_combines_scales_by_its_own_generalized_mean():
    # The exception in the same paragraph: GeM reuses its exponent across scales too.
    # Averaging instead is invisible for MAC and costs GeM several mAP.
    a = np.array([[1.0, 0.25]], np.float32)
    b = np.array([[0.5, 0.75]], np.float32)
    out = _fake_multi_scale(3.0, [a, b]).describe(Image.new("RGB", (8, 8)))

    # describe() L2-normalizes each scale before combining, so the expectation must too.
    a, b = (v / np.linalg.norm(v) for v in (a, b))
    expected = ((a**3 + b**3) / 2) ** (1 / 3)
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    # And it is genuinely not the arithmetic mean, or the test would prove nothing.
    assert not np.allclose(out, (a + b) / 2, atol=1e-3)


def test_one_scale_is_untouched_by_the_combination_rule():
    # Single-scale rows are already recorded; the generalized mean of one element is an
    # identity only up to float error, so that path is short-circuited.
    a = np.array([[0.6, 0.8]], np.float32)
    model = _fake_multi_scale(3.0, [a])
    model.scales = (1.0,)
    np.testing.assert_array_equal(model.describe(Image.new("RGB", (8, 8))), a)


def test_scales_are_l2_normalized_before_combining():
    # Otherwise the full-resolution pass dominates: it pools over the most positions,
    # so its raw magnitude is the largest and the combination is it plus a nudge. Shown
    # on MAC, whose plain averaging makes the 3:1 imbalance easiest to read.
    vectors = [np.array([[3.0, 0.0]], np.float32), np.array([[0.0, 1.0]], np.float32)]

    # Equal weight after normalization -> the mean of two unit vectors, not 3:1.
    out = _fake_multi_scale(None, vectors).describe(Image.new("RGB", (8, 8)))
    np.testing.assert_allclose(out, [[0.5, 0.5]], atol=1e-6)


@pytest.mark.parametrize("scales", [(), (1.0, 0.0), (1.0, -0.5)])
def test_impossible_scale_sets_are_rejected(scales):
    with pytest.raises(ValueError, match="scales must"):
        PooledCNN("alexnet", scales=scales, device="cpu")


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="unknown backbone"):
        PooledCNN("inception", device="cpu")  # type: ignore[arg-type]


@pytest.mark.parametrize("backbone", ["vgg19", "resnet18", "resnet34", "resnet50"])
def test_resnet_backbones_produce_their_declared_width(backbone):
    # CHANNELS is what every caller sizes its PCA and empty-result array by, so a wrong
    # entry surfaces as a shape error deep in the run rather than here.
    model = PooledCNN(backbone, p=3.0, device="cpu")
    out = model.extract([Image.new("RGB", (200, 150), "gray")])

    assert out.shape == (1, CHANNELS[backbone])
    assert np.linalg.norm(out, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_same_width_backbones_are_not_the_same_network():
    # vgg19, resnet18 and resnet34 all produce 512-D descriptors, which is the whole
    # point of comparing them -- width held fixed, architecture and depth varying. An
    # aliasing slip in the shared match arms would make that comparison meaningless.
    sizes = {
        name: sum(p.numel() for p in PooledCNN(name, device="cpu")._model.parameters())
        for name in ("vgg16", "vgg19", "resnet18", "resnet34")
    }
    assert len(set(sizes.values())) == 4
    assert sizes["vgg19"] > sizes["vgg16"]
    assert sizes["resnet34"] > sizes["resnet18"]


def test_resnet_depths_are_distinguished_not_aliased():
    # The shared match arm builds all three from the backbone name; a slip there would
    # silently run resnet18 weights under a resnet50 label.
    assert CHANNELS["resnet18"] == 512
    assert CHANNELS["resnet50"] == CHANNELS["resnet101"] == 2048

    shallow = sum(p.numel() for p in PooledCNN("resnet18", device="cpu")._model.parameters())
    deep = sum(p.numel() for p in PooledCNN("resnet50", device="cpu")._model.parameters())
    assert deep > shallow * 2


def test_every_backbone_has_a_minimum_side():
    # A missing MIN_SIDE entry is a KeyError inside preprocess, on the first scaled
    # image of a multi-hour run.
    from typing import get_args

    from cbir.descriptors.cnn.pooling import Backbone as PoolBackbone

    for name in get_args(PoolBackbone):
        assert name in MIN_SIDE and name in CHANNELS
