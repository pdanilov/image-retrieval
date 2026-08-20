"""The retrieval-SfM landmark images, used as a whitening-fitting corpus.

A deliberate exception to the two-dataset rule in AGENTS.md, taken because the reference
numbers depend on it: Radenović et al. fit PCA whitening for every off-the-shelf row on
a landmark set of this kind, not on the sibling benchmark. Our rparis6k pool is 6322
images against their ~117k, and whitening is the single largest post-processing effect
in this tier, so a comparison that holds the fitting corpus different is not really a
comparison of the same method.

There is no leakage: these are Flickr landmark photographs from SfM reconstructions, and
the released sets exclude Oxford and Paris by construction. Runs record `whiten_source`
so a row always says which corpus produced its projection.

Not downloaded automatically -- it is 37.7 GB. See the README for the fetch.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Literal

WhitenSource = Literal["held_out", "sfm30k", "sfm120k"]
"""Where a run's whitening projection is fitted. `held_out` is the sibling benchmark."""

PKL = {"sfm30k": "retrieval-SfM-30k-whiten.pkl", "sfm120k": "retrieval-SfM-120k-whiten.pkl"}


def root() -> Path:
    """Where the corpus lives; `CBIR_SFM_ROOT` overrides the default cache location."""
    return Path(os.environ.get("CBIR_SFM_ROOT", Path.home() / ".cache" / "cbir" / "sfm120k"))


def image_path(cid: str) -> Path:
    """`cids` are MD5 hashes and the archive nests by the *last* three byte-pairs.

    A hash ending `...8f4646` lives at `46/46/8f/<hash>` — reversed, not the leading
    bytes, which is the natural guess and silently finds nothing.
    """
    return root() / "ims" / cid[-2:] / cid[-4:-2] / cid[-6:-4] / cid


def whitening_paths(source: WhitenSource) -> list[str]:
    """Image paths for one whitening corpus, in the order the release lists them."""
    if source == "held_out":
        raise ValueError("held_out draws its paths from the eval split, not from here")

    manifest = root() / PKL[source]
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found — the retrieval-SfM corpus is a manual download (see README). "
            f"Set CBIR_SFM_ROOT if it lives elsewhere."
        )
    with manifest.open("rb") as f:
        cids = pickle.load(f)["cids"]

    paths = [str(image_path(cid)) for cid in cids]
    missing = next((p for p in paths if not Path(p).exists()), None)
    if missing is not None:
        raise FileNotFoundError(f"{manifest} lists images that are not extracted, e.g. {missing}")
    return paths
