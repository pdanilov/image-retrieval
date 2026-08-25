import pytest
import torch

from cbir.descriptors.cnn import weights
from cbir.descriptors.cnn.pooling import PooledCNN


def test_default_is_torchvision():
    # Every recorded row so far used torchvision weights; a changed default would make
    # them silently irreproducible.
    assert PooledCNN("alexnet", device="cpu").weights == "torchvision"


def test_unpublished_backbone_is_rejected():
    with pytest.raises(ValueError, match="no caffe weights published"):
        weights.load_caffe(torch.nn.Linear(1, 1), "alexnet")


def test_missing_file_names_the_download(monkeypatch, tmp_path):
    monkeypatch.setenv("CBIR_WEIGHTS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="download it from"):
        weights.load_caffe(torch.nn.Linear(1, 1), "resnet101")


def test_a_mismatched_file_is_rejected_rather_than_half_loaded(monkeypatch, tmp_path):
    # load_state_dict(strict=False) silently accepts a file describing another network,
    # leaving a partly random model that produces plausible-looking descriptors.
    monkeypatch.setenv("CBIR_WEIGHTS_ROOT", str(tmp_path))
    path = tmp_path / weights.CAFFE_FILES["resnet50"]
    torch.save({"nonsense.weight": torch.zeros(2, 2)}, path)

    with pytest.raises(ValueError, match="does not match resnet50"):
        weights.load_caffe(PooledCNN._build("resnet50")[0], "resnet50")


def test_only_num_batches_tracked_may_be_absent(monkeypatch, tmp_path):
    # The published files omit exactly those buffers, which matter only while training.
    monkeypatch.setenv("CBIR_WEIGHTS_ROOT", str(tmp_path))
    model, _ = PooledCNN._build("resnet50")
    state = {k: v for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    torch.save(state, tmp_path / weights.CAFFE_FILES["resnet50"])

    assert weights.load_caffe(model, "resnet50") is model
