"""The project's weights on the Hugging Face Hub, as `.safetensors`, one repository per model.

The repositories are private to the project's owner, so fetching needs a logged-in token
(`hf auth login`, or `HF_TOKEN` in CI) that can read them. Two kinds of file live there.
The Caffe-converted ImageNet weights and Radenović et al.'s fine-tuned GeM checkpoints are
mirrored from `cmp.felk.cvut.cz`, where they are a manual download — and the Caffe ones
are not a detail: on ResNet101 they are worth 0.347 → 0.461 roxford5k Medium over
torchvision's, so a reproduction depends on them. The checkpoints this project trained
are kept there because nowhere else has them; IRT in particular never released its own.

A file is named `<repository>/<file>`, without the owner. The mirrored originals keep the
reference's own names (`imagenet-caffe-resnet101`, `retrieval-sfm120k-resnet101-gem`), so
the repository says which network it holds. Every file is pinned to a SHA-256 checked
after each download, so one that changed under the same name is refused rather than
silently evaluated. A local `.pth` from the original host still takes precedence where
the loaders look for one — the mirror is a fallback, not a replacement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWNER = "pdanilov"

TRAINED: dict[str, str] = {
    "vgg16-gem-": "gem-vgg16-sfm120k-reproduced",
    "resnet101-gem-": "gem-resnet101-sfm120k-reproduced",
    "deit-vits16-dist-224-": "irt-deit-small-distilled-224",
    "deit-vitb16-dist-224-": "irt-deit-base-distilled-224",
    "dinov2-": "dinov2-sfm120k-finetune",
}
"""`cbir train` run-name prefix to the repository its checkpoints are kept in."""

PREFIX = "mirror:"
"""Marks a `checkpoint` argument as a mirror path rather than a local one, e.g.
`mirror:irt-deit-small-distilled-224/deit-vits16-dist-224-cls-margin0.85-lr5e-07-seed0.safetensors`. The
recorded run config keeps the string, so a row says where its weights came from."""

SHA256: dict[str, str] = json.loads(Path(__file__).with_name("mirror_sha256.json").read_text())
"""Mirror path to the SHA-256 of the file published there. Written by
`scripts/export_safetensors.py`, which converts and verifies every file first."""


def reference(filename: str) -> str:
    """Mirror name of one of the reference's downloads: `imagenet-caffe-resnet101-features-
    10a101d.pth` lives in `imagenet-caffe-resnet101`, `retrievalSfM120k-vgg16-gem-b4dcdc6.pth`
    in `retrieval-sfm120k-vgg16-gem`."""
    stem = Path(filename).stem
    repository = stem.split("-features-")[0] if "-features-" in stem else stem.rsplit("-", 1)[0]
    return f"{repository.replace('retrievalSfM120k', 'retrieval-sfm120k')}/{stem}.safetensors"


def trained(run: str) -> str:
    """Mirror name of a checkpoint `cbir train` wrote for `run`."""
    for prefix, repository in TRAINED.items():
        if run.startswith(prefix):
            return f"{repository}/{run}.safetensors"
    raise ValueError(f"no repository holds {run!r} runs; have {sorted(TRAINED)}")


def fetch(name: str) -> Path:
    """Download `name` (or reuse the Hub cache) and verify its hash."""
    if name not in SHA256:
        raise ValueError(f"{name!r} is not a mirrored file; have {sorted(SHA256)}")
    from huggingface_hub import hf_hub_download

    repository, filename = name.split("/", 1)
    path = Path(hf_hub_download(f"{OWNER}/{repository}", filename))
    verify(path, SHA256[name])
    return path


def resolve(checkpoint: str) -> Path:
    """A `checkpoint` argument as a local path, fetching it first if it names the mirror."""
    if checkpoint.startswith(PREFIX):
        return fetch(checkpoint.removeprefix(PREFIX))
    return Path(checkpoint)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path.name} has SHA-256 {actual}, expected {expected}: refusing a changed file")
