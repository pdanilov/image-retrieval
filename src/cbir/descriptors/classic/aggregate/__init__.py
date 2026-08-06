"""Aggregators: fold an image's ragged local descriptors into one fixed-length vector.

`BagOfWords`, `VLAD`, and `FisherVector` all expose the same two-step shape — build one
with `train(inputs, k, seed)`, then call `encode(images)` as many times as needed — so
database and query encodings go through an identical call regardless of technique.
"""
