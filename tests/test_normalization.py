import numpy as np
import pytest

from cbir.descriptors.classic.normalization import safe_l2_normalize


def test_nonzero_row_becomes_unit_norm():
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    out = safe_l2_normalize(x, axis=1)
    assert out[0] == pytest.approx([0.6, 0.8], abs=1e-6)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)


def test_zero_row_stays_zero_not_nan():
    x = np.zeros((1, 4), dtype=np.float32)
    out = safe_l2_normalize(x, axis=1)
    assert not np.isnan(out).any()
    assert np.linalg.norm(out[0]) == 0.0


def test_mixed_rows_normalize_independently():
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = safe_l2_normalize(x, axis=1)
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(out[1], [0.0, 0.0])
    np.testing.assert_allclose(out[2], [1.0, 0.0], atol=1e-6)


def test_normalizes_along_arbitrary_axis_for_blocked_input():
    # (N=1, k=2 blocks, d=2) - VLAD's intra-normalization shape, axis=2 normalizes
    # each block independently rather than the whole row.
    x = np.array([[[3.0, 4.0], [0.0, 5.0]]], dtype=np.float32)
    out = safe_l2_normalize(x, axis=2)
    np.testing.assert_allclose(out[0, 0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(out[0, 1], [0.0, 1.0], atol=1e-6)


def test_output_dtype_is_float32():
    x = np.array([[3.0, 4.0]], dtype=np.float64)
    out = safe_l2_normalize(x, axis=1)
    assert out.dtype == np.float32
