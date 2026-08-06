"""Disk cache for trained `GaussianMixture` (EM) models, used by Fisher vectors.

Fitting a GMM via EM on tens of millions of held-out RootSIFT descriptors is the other
expensive step in the classic pipeline, alongside RootSIFT extraction itself (cached by
`cache/rootsift.py`). Unlike `Vocabulary` (shared by BoW/VLAD, see `cache/vocabulary.py`),
a `GaussianMixture` is only ever consumed by Fisher -- but re-running Fisher at a `k`
already tried (or across the two eval directions when one reuses the other's held-out
dataset) should still skip EM refitting.

Cached under this repo's `data/gmm/` (gitignored — see `.gitignore`'s
`/data/` rule), keyed by `(dataset, k, seed)`. `dataset` is the *held-out* dataset name
(`ClassicDescriptorInputs.held_out_dataset`), not the eval dataset -- see
`cache/vocabulary.py`'s docstring for why that distinction matters.

A cache hit/miss is decided by a row lookup in the `gmm` table of the shared SQLite
index (`db.py`), not by a file existing at a conventionally-named path -- see that
module's docstring for why — it also owns where blobs get written, so this module holds
no paths of its own, only its table name. Keeping all of that here rather than on
`GaussianMixture` itself is what lets `codebook/gaussian_mixture.py` stay pure EM.
"""

import numpy as np

from cbir.descriptors.classic.cache import db
from cbir.descriptors.classic.codebook.gaussian_mixture import GaussianMixture

TABLE = "gmm"


class GaussianMixtureCache:
    """Cached diagonal-covariance GMM fitting, keyed by `(dataset, k, seed)`."""

    @staticmethod
    def train(dataset: str, descriptors: np.ndarray, k: int, seed: int) -> GaussianMixture:
        """`GaussianMixture` for `(dataset, k, seed)`: fit fresh, or loaded from disk on a hit."""
        with db.connect() as conn:
            row = conn.execute(
                f"SELECT filepath FROM {TABLE} WHERE dataset = ? AND k = ? AND seed = ?", (dataset, k, seed)
            ).fetchone()

        if row is not None:
            with np.load(row[0]) as data:
                return GaussianMixture(weights=data["weights"], means=data["means"], variances=data["variances"])

        model = GaussianMixture.train(descriptors, k=k, seed=seed)
        path = db.reserve_blob_path(TABLE, f"{dataset}_k{k}_seed{seed}")
        np.savez(path, weights=model.weights, means=model.means, variances=model.variances)

        with db.connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {TABLE} (dataset, k, seed, filepath) VALUES (?, ?, ?, ?)",
                (dataset, k, seed, str(path)),
            )

        return model
