import numpy as np
import pytest
import torch
from PIL import Image

from cbir.descriptors.cnn.neural_codes import IMAGENET_MEAN, IMAGENET_STD
from cbir.descriptors.transformer.vit import IMAGE_SIZE, SPECS, FrozenViT, _unexpectedly_missing

SMALL = "dinov2-vits14"


def _images(count: int, size: tuple[int, int] = (64, 48)) -> list[Image.Image]:
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8)) for _ in range(count)]


def _normalized(value: float, channel: int = 0) -> float:
    """What a raw pixel intensity in [0, 1] becomes after ImageNet normalization."""
    return (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]


def _vit(**kwargs) -> FrozenViT:
    """A real instance in a non-default mode; the small model is cached after first use."""
    return FrozenViT(SMALL, device="cpu", **kwargs)


@pytest.fixture(scope="module")
def model() -> FrozenViT:
    """One instance for the module — building it downloads the smallest DINOv2."""
    return FrozenViT(SMALL, device="cpu", batch_size=2)


def test_preprocess_produces_the_square_protocol_input(model):
    tensor = model.preprocess(Image.new("RGB", (640, 100)))
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_pads_the_short_side_rather_than_cropping_it(model):
    # A 2:1 image has its long side resized to 224, which leaves the short side at 112.
    # The protocol centres that on a black square: 56 rows of padding above and below,
    # and the whole image kept. Cropping instead would cut the frame's edges -- which
    # on this benchmark can remove the landmark.
    tensor = model.preprocess(Image.new("RGB", (448, 224), "white"))

    assert float(tensor[0, 0, IMAGE_SIZE // 2]) == pytest.approx(_normalized(0.0), abs=1e-4)
    assert float(tensor[0, IMAGE_SIZE - 1, IMAGE_SIZE // 2]) == pytest.approx(_normalized(0.0), abs=1e-4)
    assert float(tensor[0, IMAGE_SIZE // 2, 0]) == pytest.approx(_normalized(1.0), abs=1e-4)
    assert float(tensor[0, IMAGE_SIZE // 2, IMAGE_SIZE - 1]) == pytest.approx(_normalized(1.0), abs=1e-4)


def test_preprocess_enlarges_small_images_to_the_protocol_size(model):
    # The protocol fixes the long side at 224 rather than capping it, so a small query
    # crop is enlarged -- unlike the CNN tier, which never upscales.
    tensor = model.preprocess(Image.new("RGB", (50, 100), "white"))

    column = tensor[0, :, IMAGE_SIZE // 2]
    assert column == pytest.approx(torch.full_like(column, _normalized(1.0)), abs=1e-4)


def test_preprocess_applies_imagenet_normalization(model):
    tensor = model.preprocess(Image.new("RGB", (32, 32), color=(128, 128, 128)))
    assert not np.isclose(float(tensor.mean()), 0.5, atol=0.05)


def test_aspect_keeps_the_ratio_without_padding():
    # A 2:1 image becomes 224 x 112: both sides whole patches, no black bars spent.
    vit = _vit(resize="aspect")
    tensor = vit.preprocess(Image.new("RGB", (448, 224), "white"))

    assert tensor.shape == (3, 112, IMAGE_SIZE)
    assert float(tensor[0].min()) == pytest.approx(_normalized(1.0), abs=1e-4)


def test_aspect_rounds_each_side_to_whole_patches():
    vit = _vit(resize="aspect")
    tensor = vit.preprocess(Image.new("RGB", (300, 100)))

    assert tensor.shape[1] % 14 == 0 and tensor.shape[2] % 14 == 0
    assert tensor.shape[2] == IMAGE_SIZE


def test_crop_is_dinov2s_eval_transform_and_cuts_the_edges():
    # Shorter side to 256, centre crop to 224: on a 2:1 image the left and right
    # borders are cut away, which is the property that makes it a risky protocol here.
    image = Image.new("RGB", (400, 200), "white")
    for y in range(200):
        for x in range(40):
            image.putpixel((x, y), (255, 0, 0))

    tensor = _vit(resize="crop").preprocess(image)

    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    green = tensor[1]
    assert float(green.min()) == pytest.approx(_normalized(1.0, channel=1), abs=1e-4)


def test_bilinear_and_bicubic_differ():
    image = _images(1, size=(97, 61))[0]
    bicubic = _vit().preprocess(image)
    bilinear = _vit(interpolation="bilinear").preprocess(image)
    assert not torch.allclose(bicubic, bilinear)


def test_unknown_resize_is_rejected_before_loading():
    with pytest.raises(ValueError, match="unknown resize"):
        FrozenViT(SMALL, resize="stretch", device="cpu")  # type: ignore[arg-type]


def test_aspect_mode_batches_mixed_shapes_without_reordering():
    # Differently shaped inputs cannot be stacked; a shape change must end the batch
    # without permuting descriptors against their images.
    vit = FrozenViT(SMALL, resize="aspect", device="cpu", batch_size=4)
    images = [*_images(2, size=(64, 48)), *_images(1, size=(48, 64)), *_images(2, size=(64, 48))]

    batched = vit.extract(images)
    one_at_a_time = np.concatenate([vit.extract([image]) for image in images])
    np.testing.assert_allclose(batched, one_at_a_time, atol=1e-4)


def test_image_size_must_be_a_multiple_of_the_patch():
    # Not an error to the patch embedding, which would silently drop the remainder.
    with pytest.raises(ValueError, match="multiple of the patch"):
        FrozenViT(SMALL, image_size=230, device="cpu")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        FrozenViT("dinov1-vits16", device="cpu")  # type: ignore[arg-type]


def test_patch_size_is_per_model():
    # DINOv3 uses 16-px patches: 224 is fine, 518 (DINOv2's 37 x 14) is not.
    with pytest.raises(ValueError, match="multiple of the patch size 16"):
        FrozenViT("dinov3-vitl16", image_size=518, device="cpu")


def test_fixed_resolution_checkpoints_take_only_their_native_size():
    # Each SigLIP 2 checkpoint learned position embeddings for one grid; another size
    # would interpolate them, which is not the released network.
    with pytest.raises(ValueError, match="fixed-resolution"):
        FrozenViT("siglip2-vitl16-256", image_size=224, device="cpu")
    with pytest.raises(ValueError, match="fixed-resolution"):
        FrozenViT("siglip2-vitl16-256", image_size=256, resize="aspect", device="cpu")


def test_every_family_has_its_own_normalization():
    # SigLIP is trained on [-1, 1] inputs, DINO on ImageNet statistics. Feeding one the
    # other's does not fail; it shifts every activation.
    assert SPECS["siglip2-vitl16-256"].mean == (0.5, 0.5, 0.5)
    assert SPECS["dinov3-vitl16"].mean == SPECS["dinov2-vitl14"].mean


def test_extract_returns_l2_normalized_descriptors_of_the_model_width(model):
    vectors = model.extract(_images(3))

    assert vectors.shape == (3, SPECS[SMALL].width)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_the_descriptor_is_the_normalized_cls_token(model):
    # The published rows are measured on CLS after the final LayerNorm. A readout that
    # drifted to mean-pooled patches, or to CLS before the norm, would still return
    # plausible unit vectors -- this pins which token, at which point.
    image = _images(1)[0]
    with torch.inference_mode():
        hidden = model._model(pixel_values=model.preprocess(image).unsqueeze(0)).last_hidden_state
    expected = hidden[:, 0].numpy()
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)

    np.testing.assert_allclose(model.extract([image]), expected, atol=1e-5)


def test_extract_is_deterministic(model):
    images = _images(2)
    np.testing.assert_allclose(model.extract(images), model.extract(images), rtol=1e-5)


def test_extract_spans_batches_without_reordering(model):
    # batch_size=2 with 5 images exercises a ragged final batch; a bug in the batching
    # would silently permute descriptors against their ground-truth indices.
    images = _images(5)
    batched = model.extract(images)
    one_at_a_time = np.concatenate([model.extract([image]) for image in images])
    np.testing.assert_allclose(batched, one_at_a_time, atol=1e-4)


def test_extract_of_nothing_is_an_empty_array_not_an_error(model):
    assert model.extract([]).shape == (0, SPECS[SMALL].width)


def test_different_images_give_different_descriptors(model):
    vectors = model.extract([Image.new("RGB", (64, 64), c) for c in ("black", "white", "red")])
    off_diagonal = (vectors @ vectors.T)[~np.eye(3, dtype=bool)]
    assert off_diagonal.max() < 0.99


def test_only_unused_heads_may_be_missing_at_load():
    # DeiT's classification checkpoints never trained the generic `pooler` head that
    # Hugging Face builds, so it loads random -- harmless, since nothing reads it. A
    # missing encoder weight is not harmless: part of the network would be random and
    # still return unit vectors.
    assert _unexpectedly_missing({"pooler.dense.weight", "pooler.dense.bias"}) == set()
    assert _unexpectedly_missing({"pooler.dense.weight", "encoder.layer.0.attention.query.weight"}) == {
        "encoder.layer.0.attention.query.weight"
    }


def test_deit_is_read_out_by_cls_not_by_its_untrained_pooler():
    # The trap this pins: `pooler_output` for a DeiT checkpoint is tanh(dense(CLS)) with
    # random dense weights. Reading it would give plausible unit vectors and a
    # meaningless mAP.
    vit = FrozenViT("deit-vits16-dist-224", image_size=224, device="cpu", batch_size=2)
    image = _images(1)[0]
    with torch.inference_mode():
        output = vit._model(pixel_values=vit.preprocess(image).unsqueeze(0))
    cls = output.last_hidden_state[:, 0].numpy()
    cls /= np.linalg.norm(cls, axis=1, keepdims=True)

    descriptor = vit.extract([image])
    np.testing.assert_allclose(descriptor, cls, atol=1e-5)
    assert descriptor.shape == (1, SPECS["deit-vits16-dist-224"].width)
