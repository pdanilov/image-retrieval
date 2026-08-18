import numpy as np
import pytest
import torch
from PIL import Image

from cbir.descriptors.cnn.neural_codes import DIM, INPUT_SIZE, NeuralCodes


def _images(count: int, size: tuple[int, int] = (64, 48)) -> list[Image.Image]:
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8)) for _ in range(count)]


@pytest.fixture(scope="module")
def model() -> NeuralCodes:
    """One instance for the module — building it downloads and loads ImageNet weights."""
    return NeuralCodes("alexnet", device="cpu", batch_size=2)


def test_preprocess_forces_the_fixed_fc_input_size():
    # The fully-connected layer cannot accept anything else; this constraint is the
    # whole reason the pooling methods that follow exist.
    tensor = NeuralCodes.preprocess(Image.new("RGB", (640, 100)))
    assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)


def test_preprocess_resizes_rather_than_centre_crops():
    # A centre crop would discard the edges of the frame, which on this benchmark can
    # remove the landmark outright and would undo the bbx crop queries are given.
    # Content at a corner must survive.
    image = Image.new("RGB", (200, 200), color="black")
    image.putpixel((199, 199), (255, 255, 255))

    tensor = NeuralCodes.preprocess(image)

    corner = tensor[:, -1, -1]
    centre = tensor[:, INPUT_SIZE // 2, INPUT_SIZE // 2]
    assert not torch.allclose(corner, centre)


def test_preprocess_applies_imagenet_normalization():
    # Skipping normalization does not fail anywhere; it just shifts every activation
    # and quietly costs mAP. A mid-grey image must not come back as 0.5.
    tensor = NeuralCodes.preprocess(Image.new("RGB", (32, 32), color=(128, 128, 128)))
    assert not np.isclose(float(tensor.mean()), 0.5, atol=0.05)


def test_extract_returns_l2_normalized_codes_in_order(model):
    codes = model.extract(_images(3))

    assert codes.shape == (3, DIM)
    assert codes.dtype == np.float32
    # exact_search reads a dot product as cosine similarity, so anything reaching it
    # un-normalized is scored partly on magnitude.
    assert np.linalg.norm(codes, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_extract_is_deterministic(model):
    # AlexNet's classifier contains dropout. Without eval() and inference_mode, every
    # call would return different vectors and every mAP would be unreproducible.
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
    codes = model.extract([])
    assert codes.shape == (0, DIM)


def test_different_images_give_different_codes(model):
    # A model truncated at the wrong layer, or fed constant tensors, can return near
    # identical vectors for everything -- which still normalizes and still searches.
    codes = model.extract([Image.new("RGB", (64, 64), c) for c in ("black", "white", "red")])
    similarities = codes @ codes.T
    off_diagonal = similarities[~np.eye(3, dtype=bool)]
    assert off_diagonal.max() < 0.99


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="unknown backbone"):
        NeuralCodes("resnet101", device="cpu")  # type: ignore[arg-type]


def test_vgg16_truncates_to_fc6_not_fc7():
    # The two classifiers do not start alike -- AlexNet opens with a Dropout, VGG16 with
    # the Linear -- so the same cut is a different index. Too far gives fc7, too short
    # gives the pre-ReLU signed projection; both return plausible-looking vectors.
    layers = list(NeuralCodes._build("vgg16").classifier.children())
    assert [type(layer).__name__ for layer in layers] == ["Linear", "ReLU"]
    assert layers[0].out_features == DIM


def test_alexnet_truncation_is_unchanged():
    layers = list(NeuralCodes._build("alexnet").classifier.children())
    assert [type(layer).__name__ for layer in layers] == ["Dropout", "Linear", "ReLU"]


def test_vgg16_codes_are_post_relu_and_the_canonical_width():
    # Post-ReLU means non-negative. A signed code here would mean fc6 was taken before
    # its activation, which is a different feature than the paper's.
    codes = NeuralCodes("vgg16", device="cpu", batch_size=2).extract([Image.new("RGB", (300, 200), "gray")])

    assert codes.shape == (1, DIM)
    assert (codes >= 0).all()
