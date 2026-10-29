"""Convert every checkpoint the mirror publishes from `.pth` to `.safetensors`, and prove it.

Outside the harness — it writes nothing to `results/`. For each file it converts, reads
the conversion back and compares it with the original exactly: every tensor bit for bit
and in its own dtype, every whitening array likewise, and the rest of `meta` field by
field. One mismatch aborts the export. It then prints each file's SHA-256, the values
`cbir.descriptors.mirror.SHA256` pins.

Serialization is deterministic, so running this on the reference's original downloads
reproduces the mirror's hashes — the way to check the mirror instead of trusting it.

    uv run python scripts/export_safetensors.py data/mirror
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cbir.descriptors import mirror
from cbir.descriptors.cnn import finetuned, weights
from cbir.descriptors.mirror import sha256
from cbir.descriptors.serialization import read, write

RUNS = Path("data/runs")
MANIFEST = Path(mirror.__file__).with_name("mirror_sha256.json")

EXCLUDED = ("dinov3-",)
"""Not mirrored. DINOv3 is gated behind its own license, and the one LoRA run on it scored
at the noise floor, so there is nothing worth keeping."""


def sources() -> dict[str, Path]:
    """Mirror path -> local `.pth`, for everything the mirror carries."""
    out = {}
    for filename in [*weights.CAFFE_FILES.values(), *finetuned.FILES.values()]:
        out[mirror.reference(filename)] = weights.root() / filename
    for run in sorted(RUNS.iterdir()):
        best = run / "best.pth"
        if best.exists() and not run.name.startswith(EXCLUDED):
            out[mirror.trained(run.name)] = best
    return out


def same(original: tuple, converted: tuple, name: str) -> None:
    (state, meta), (state2, meta2) = original, converted
    if state.keys() != state2.keys():
        raise AssertionError(f"{name}: state keys differ")
    for key, tensor in state.items():
        if tensor.dtype != state2[key].dtype or not torch.equal(tensor, state2[key]):
            raise AssertionError(f"{name}: tensor {key} differs")
    if (meta is None) != (meta2 is None):
        raise AssertionError(f"{name}: meta present in one only")
    if meta is None:
        return
    whitening, whitening2 = meta.get("Lw", {}), meta2.get("Lw", {})
    for corpus, variants in whitening.items():
        for variant, arrays in variants.items():
            for array_name, array in arrays.items():
                other = whitening2[corpus][variant][array_name]
                if other.dtype != array.dtype or not np.array_equal(other, array):
                    raise AssertionError(f"{name}: Lw/{corpus}/{variant}/{array_name} differs")
    rest = {key: value for key, value in meta.items() if key != "Lw"}
    rest2 = {key: value for key, value in meta2.items() if key != "Lw"}
    # Through JSON on both sides: a tuple in the original comes back a list, by design.
    if json.loads(json.dumps(rest, sort_keys=True)) != rest2:
        raise AssertionError(f"{name}: meta differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    manifest = {}
    for name, source in sources().items():
        if not source.exists():
            raise FileNotFoundError(f"{source} is missing; download it before exporting")
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original = read(source)
        write(target, *original)
        same(original, read(target), name)
        manifest[name] = sha256(target)
        print(f"{manifest[name]}  {target.stat().st_size / 2**20:7.1f} MB  {name}")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"pinned in {MANIFEST}")
    print(f"\n{len(manifest)} files exported and verified against their .pth")


if __name__ == "__main__":
    main()
