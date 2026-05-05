"""Tests for ``stability_monitor.smoothing``."""

from __future__ import annotations

import numpy as np
import pytest

from stability_monitor.smoothing import StreamingSavGol, sg_smooth_batch


def test_polynomial_passes_through_unchanged() -> None:
    """SG with polyorder=3 should reproduce any cubic polynomial exactly."""
    T, J = 50, 6
    t = np.arange(T)
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal((4, J))  # cubic in t
    poly = (
        coeffs[0]
        + coeffs[1] * t[:, None]
        + coeffs[2] * (t[:, None] ** 2)
        + coeffs[3] * (t[:, None] ** 3)
    )
    smoothed = sg_smooth_batch(poly, window=7, polyorder=3)
    np.testing.assert_allclose(smoothed, poly, rtol=1e-10, atol=1e-10)


def test_streaming_matches_batch_interior() -> None:
    """Streaming SG output equals batch SG on the strictly-symmetric interior."""
    T, J, window, polyorder = 60, 4, 7, 3
    rng = np.random.default_rng(1)
    x = rng.standard_normal((T, J))
    batch_smoothed = sg_smooth_batch(x, window, polyorder)

    sm = StreamingSavGol(window, polyorder, J)
    stream_outs: list[np.ndarray | None] = []
    for k in range(T):
        stream_outs.append(sm.update(x[k]))

    lag = (window - 1) // 2  # = 3
    # When the k-th sample arrives (0-indexed), if k >= window - 1 = 6,
    # the centre of the buffer is index k - lag = k - 3.
    for k in range(window - 1, T):
        m = k - lag
        # Boundary frames (the first/last `lag` samples of batch) come from
        # scipy's polynomial-fit interp mode; only require equality on the
        # symmetric interior.
        if m < lag or m > T - 1 - lag:
            continue
        out = stream_outs[k]
        assert out is not None
        np.testing.assert_allclose(out, batch_smoothed[m], rtol=1e-12, atol=1e-12)


def test_streaming_warmup_returns_none() -> None:
    sm = StreamingSavGol(window=7, polyorder=3, n_channels=2)
    rng = np.random.default_rng(2)
    for k in range(6):
        assert sm.update(rng.standard_normal(2)) is None
    out = sm.update(rng.standard_normal(2))
    assert out is not None and out.shape == (2,)


def test_streaming_reject_even_window() -> None:
    with pytest.raises(ValueError):
        StreamingSavGol(window=6, polyorder=3, n_channels=1)


def test_streaming_input_shape_validated() -> None:
    sm = StreamingSavGol(window=5, polyorder=2, n_channels=3)
    with pytest.raises(ValueError):
        sm.update(np.zeros(2))


def test_streaming_reset_clears_state() -> None:
    sm = StreamingSavGol(window=5, polyorder=2, n_channels=1)
    rng = np.random.default_rng(3)
    for _ in range(20):
        sm.update(rng.standard_normal(1))
    sm.reset()
    for k in range(4):
        assert sm.update(rng.standard_normal(1)) is None
