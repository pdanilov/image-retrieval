"""Exact similarity search via a single torch matmul.

roxford5k/rparis6k are small enough (5-6k database images) that exact search on GPU
is both correct and fast, and no ANN index (FAISS, HNSW, IVF-PQ) is needed — at this
scale one would add approximation error while being slower than a single matmul.
"""

import numpy as np
import torch


def exact_search(queries: np.ndarray, database: np.ndarray, device: str | None = None) -> np.ndarray:
    """Rank database indices by descending cosine similarity for each query.

    `queries` (Nq, D) and `database` (Nd, D) must already be float32, L2-normalized,
    so cosine similarity is a plain dot product. Returns an (Nq, Nd) int64 array;
    row i is a full permutation of database indices, most similar first. Full
    rankings (not top-k) are required — eval needs the rank of every positive after
    dropping ignored images, and a positive can be arbitrarily far down the list.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    queries_t = torch.from_numpy(np.asarray(queries, dtype=np.float32)).to(device)
    database_t = torch.from_numpy(np.asarray(database, dtype=np.float32)).to(device)

    similarities = queries_t @ database_t.T
    rankings = torch.argsort(similarities, dim=1, descending=True)
    return rankings.cpu().numpy()
