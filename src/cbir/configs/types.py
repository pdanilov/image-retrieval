"""Vocabulary shared by every descriptor tier's config.

A leaf module on purpose: `classic.py` and `cnn.py` both need `Encoded`, and `run.py`
imports both to build the union, so anything they share has to live somewhere neither
of them imports back.
"""

import numpy as np

Encoded = tuple[np.ndarray, np.ndarray]
"""`(database_vectors, query_vectors)`, both `(N, D)` float32 and L2-normalized.

Normalized because `search.exact_search` takes a plain dot product as cosine
similarity — anything reaching it un-normalized is silently scored on magnitudes.
"""
