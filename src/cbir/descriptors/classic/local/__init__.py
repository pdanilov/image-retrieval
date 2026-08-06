"""Local descriptors: per-keypoint features extracted from a single image.

The first stage of the classic pipeline. What comes out is a ragged `(n_i, 128)` array
per image, which `codebook/` clusters and `aggregate/` folds into one vector per image.
"""
