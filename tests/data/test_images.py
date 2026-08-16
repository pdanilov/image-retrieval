import pytest
from PIL import Image

import cbir.data.images as images_module
from cbir.data.images import image_paths, iter_images


class _FakeDataset:
    """Stands in for an HF Dataset whose `image` column is an `Image(decode=True)`.

    Whole-column access returns decoded images (the expensive path `image_paths` must
    avoid); only after `cast_column` to a decode=False feature does it hand back the
    undecoded `{"path": ..., "bytes": ...}` records.
    """

    def __init__(self, paths, *, decoded=True):
        self._paths = paths
        self._decoded = decoded

    def cast_column(self, name, feature):
        assert name == "image"
        assert feature.decode is False
        return _FakeDataset(self._paths, decoded=False)

    def __getitem__(self, key):
        assert key == "image"
        if self._decoded:
            raise AssertionError("image_paths must not read the decoded image column")
        return [{"path": path, "bytes": None} for path in self._paths]


def test_image_paths_reads_undecoded_column():
    # _FakeDataset raises if the decoded column is touched, so this passing is the
    # assertion that decoding was avoided -- the whole point of the decode=False cast.
    assert image_paths(_FakeDataset(["a.jpg", "b.jpg"])) == ["a.jpg", "b.jpg"]


def test_iter_images_opens_each_path_and_preserves_order(tmp_path):
    sizes = [(4, 3), (6, 5), (8, 7)]
    paths = []
    for i, size in enumerate(sizes):
        path = tmp_path / f"img{i}.png"
        Image.new("RGB", size).save(path)
        paths.append(str(path))

    assert [image.size for image in iter_images(paths)] == sizes


def test_iter_images_closes_each_image_before_opening_the_next(monkeypatch):
    # The contract that keeps peak memory at one decoded image: image N-1 must be
    # closed before image N is opened. Decoding everything up front is what OOMed.
    events = []

    class FakeImage:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            events.append(("open", self.path))
            return self

        def __exit__(self, *exc_info):
            events.append(("close", self.path))
            return False

    monkeypatch.setattr(images_module.Image, "open", FakeImage)

    assert [image.path for image in iter_images(["a.jpg", "b.jpg"])] == ["a.jpg", "b.jpg"]
    assert events == [
        ("open", "a.jpg"),
        ("close", "a.jpg"),
        ("open", "b.jpg"),
        ("close", "b.jpg"),
    ]


def test_iter_images_is_lazy(tmp_path):
    # Creating the generator must open nothing -- a cache hit never consumes it, and
    # must not pay for (or fail on) any image as a result.
    missing = str(tmp_path / "does-not-exist.png")
    generator = iter_images([missing])

    with pytest.raises(FileNotFoundError):
        next(iter(generator))
