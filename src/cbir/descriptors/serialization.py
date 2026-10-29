"""Checkpoints as files: the pickled `.pth` this project reads, the `.safetensors` it publishes.

Every checkpoint here is one of two shapes. The Caffe-converted ImageNet weights are a bare
state dict. Everything fine-tuned — Radenović et al.'s GeM checkpoints and the ones
`cbir train` writes — is `{"state_dict": ..., "meta": ...}`, where `meta` describes the
network and may carry a supervised whitening `meta["Lw"]` as numpy arrays.

`.pth` is read through pickle, which executes whatever the file says to. That is
acceptable for a file downloaded once from the reference's own host, and not for a public
mirror. `.safetensors` holds tensors and string metadata only, so the mirror publishes
that, laid out so the round trip is exact:

* state-dict tensors under `state/<name>`;
* each whitening array under `Lw/<corpus>/<variant>/<name>`, in its own dtype (the
  published `P` is float64, `m` float32 — converting either would change the numbers);
* the rest of `meta` as JSON, inside the one header field `cbir`. A bare state dict has
  none.

One field, not several: safetensors writes its metadata map in no fixed order, so two
fields would let the same checkpoint serialize to different bytes. With one, a file is a
function of its contents — anyone can convert the reference's original `.pth` and get the
SHA-256 the mirror pins, which is how the mirror can be checked rather than trusted.

`read` accepts either format and returns the same `(state, meta)` pair, so loaders are
indifferent to which one is on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT = "cbir-checkpoint/1"
"""Written into every header, so a reader can refuse a file laid out some other way."""

FIELD = "cbir"
"""The single safetensors metadata field holding `{"format": ..., "meta": ...}`."""

STATE = "state/"
WHITENING = "Lw/"


def read(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    """`(state_dict, meta)` from either format; `meta` is `None` for a bare state dict."""
    path = Path(path)
    if path.suffix == ".safetensors":
        return _read_safetensors(path)
    # `weights_only=False`: `meta` holds numpy arrays. Reached only for local `.pth`
    # files — the reference's downloads and this project's own training output.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return dict(payload["state_dict"]), payload["meta"]
    return dict(payload), None


def write(path: str | Path, state: dict[str, torch.Tensor], meta: dict[str, Any] | None = None) -> None:
    """Write `(state, meta)` as `.safetensors`, in the layout `read` inverts exactly."""
    from safetensors.torch import save_file

    tensors = {f"{STATE}{key}": _own(value) for key, value in state.items()}
    header: dict[str, Any] = {"format": FORMAT}
    if meta is not None:
        rest = {key: value for key, value in meta.items() if key != "Lw"}
        for corpus, variants in meta.get("Lw", {}).items():
            for variant, arrays in variants.items():
                for name, array in arrays.items():
                    tensors[f"{WHITENING}{corpus}/{variant}/{name}"] = torch.from_numpy(np.ascontiguousarray(array))
        header["meta"] = rest
    # Sorted, so the same checkpoint always serializes to the same bytes and SHA-256.
    save_file(tensors, str(path), metadata={FIELD: json.dumps(header, sort_keys=True)})


def _read_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        header = json.loads((handle.metadata() or {}).get(FIELD, "{}"))
        if header.get("format") != FORMAT:
            raise ValueError(f"{path.name} is not a {FORMAT} file (format {header.get('format')!r})")
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118 - not a dict

    state = {key.removeprefix(STATE): value for key, value in tensors.items() if key.startswith(STATE)}
    if "meta" not in header:
        return state, None
    meta = header["meta"]
    for key, value in tensors.items():
        if key.startswith(WHITENING):
            corpus, variant, name = key.removeprefix(WHITENING).split("/")
            meta.setdefault("Lw", {}).setdefault(corpus, {}).setdefault(variant, {})[name] = value.numpy()
    return state, meta


def _own(tensor: torch.Tensor) -> torch.Tensor:
    """A contiguous tensor with storage of its own: safetensors refuses shared or strided
    storage, and copying is exact."""
    return tensor.detach().contiguous().clone()
