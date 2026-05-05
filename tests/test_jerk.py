"""Tests for ``stability_monitor.metrics.jerk``."""

from __future__ import annotations

import numpy as np
import pytest

from stability_monitor.config import Config
from stability_monitor.metrics.jerk import (
    JerkResult,
    StreamingRMSJerk,
    batch,
)


def _cfg(N: int = 30, J: int = 6) -> Config:
    return Config(N=N, n_joints=J)


def _make_action(state: np.ndarray) -> np.ndarray:
    """Jerk metric ignores action; produce a same-shape dummy."""
    return state.copy()


def test_pure_quadratic_position_has_zero_jerk() -> None:
    """A polynomial of order <= 3 has analytic jerk == constant; for order
    <= 2, jerk == 0. SG with polyorder=3 reproduces it exactly, so RMS jerk
    is 0 within numerical tolerance.
    """
    cfg = _cfg(N=10, J=2)
    T = 80
    t = np.arange(T) * cfg.Ts
    rng = np.random.default_rng(0)
    c0 = rng.standard_normal(2)
    v = rng.standard_normal(2)
    a = rng.standard_normal(2)
    state = c0 + v * t[:, None] + 0.5 * a * (t[:, None] ** 2)
    action = _make_action(state)

    res = batch(action.astype(np.float64), state.astype(np.float64), cfg)
    valid = ~np.isnan(res.agg)
    assert valid.any()
    np.testing.assert_allclose(res.agg[valid], 0.0, atol=1e-6)
    np.testing.assert_allclose(res.per_joint[valid], 0.0, atol=1e-6)


def test_pure_cubic_position_has_constant_jerk() -> None:
    """For q(t) = (1/6) j t^3, dddot{q} == j. SG polyorder=3 fits cubics
    exactly, so RMS jerk should equal |j| element-wise (after warmup).
    """
    cfg = _cfg(N=10, J=2)
    T = 80
    t = np.arange(T) * cfg.Ts
    j_true = np.array([2.5, -1.7])
    state = (j_true[None, :] / 6.0) * (t[:, None] ** 3)
    action = _make_action(state)

    res = batch(action.astype(np.float64), state.astype(np.float64), cfg)
    valid = ~np.isnan(res.agg)
    expected_pj = np.broadcast_to(np.abs(j_true), res.per_joint[valid].shape)
    np.testing.assert_allclose(
        res.per_joint[valid], expected_pj, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        res.agg[valid],
        np.full(valid.sum(), np.linalg.norm(j_true)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_noisy_signal_has_higher_jerk_than_smooth() -> None:
    """Adding broadband noise to a smooth trajectory must raise RMS jerk."""
    cfg = _cfg(N=20, J=1)
    T = 200
    t = np.arange(T) * cfg.Ts
    smooth = np.sin(2 * np.pi * 0.5 * t)[:, None]  # 0.5 Hz sine
    rng = np.random.default_rng(2)
    noisy = smooth + 0.01 * rng.standard_normal((T, 1))
    action = smooth  # any same-shape dummy

    j_smooth = batch(action.astype(np.float64), smooth.astype(np.float64), cfg)
    j_noisy = batch(action.astype(np.float64), noisy.astype(np.float64), cfg)

    valid = ~np.isnan(j_smooth.agg)
    assert np.nanmean(j_noisy.agg[valid]) > 2.0 * np.nanmean(j_smooth.agg[valid])


def test_warmup_boundaries() -> None:
    """First valid frame is k = N + 1 (one extra frame because the first
    valid jerk lives at j = 2). Last valid frame is k = T - 3 (jerk
    undefined at j = T - 2 and j = T - 1).
    """
    cfg = _cfg(N=8, J=3)
    T = 40
    rng = np.random.default_rng(3)
    state = rng.standard_normal((T, 3)).astype(np.float64)
    action = _make_action(state)

    res = batch(action, state, cfg)

    # k = 0..N..N+1 — frames whose window includes any nan jerk
    assert np.all(np.isnan(res.agg[: cfg.N + 1]))
    # last 2 frames have undefined jerk -> their windows contain a nan
    assert np.all(np.isnan(res.agg[T - 2 :]))
    # interior should be finite
    assert np.all(np.isfinite(res.agg[cfg.N + 1 : T - 2]))


def test_streaming_first_calls_return_nan() -> None:
    cfg = _cfg(N=4, J=2)
    sm = StreamingRMSJerk(cfg)
    rng = np.random.default_rng(4)
    # Need (sg_window - 1) + 4 + (N - 1) = 6 + 4 + 3 = 13 updates before
    # the first finite output for default sg_window=7, N=4.
    for _ in range(12):
        out = sm.update(rng.standard_normal(2), rng.standard_normal(2))
        assert np.isnan(out.agg)
        assert np.all(np.isnan(out.per_joint))


def test_streaming_matches_batch_on_interior() -> None:
    """Non-negotiable consistency check on the strictly-symmetric interior.

    Streaming output at update step ``k`` corresponds to time index
    ``m = k - lag`` (lag = 5 with default cfg). Batch and streaming must
    agree numerically for every ``k`` whose ``m`` falls inside the
    symmetric SG region AND whose RMS window is fully populated.
    """
    cfg = _cfg(N=12, J=4)
    T = 200
    rng = np.random.default_rng(5)
    state = rng.standard_normal((T, 4)).astype(np.float64)
    action = _make_action(state)

    res = batch(action, state, cfg)

    sm = StreamingRMSJerk(cfg)
    lag = sm.lag  # 5
    sg_lag = (cfg.sg_window - 1) // 2  # 3
    agg_stream = np.full(T, np.nan, dtype=np.float64)
    pj_stream = np.full((T, 4), np.nan, dtype=np.float64)
    for k in range(T):
        out = sm.update(action[k], state[k])
        agg_stream[k] = out.agg
        pj_stream[k] = out.per_joint

    # Agreement zone in batch indices m. Streaming throws out jerk[2..4]
    # because q_tilde[0..2] would be SG-boundary-interpolated rather than
    # the symmetric SG output streaming computes; its first jerk is at
    # j = sg_lag + 2 = 5. The RMS window [m-N+1, m] must therefore lie
    # entirely in [5, T-1-sg_lag-2] = [5, T-6].
    m_lo = cfg.N + sg_lag + 1   # = N + 4 with sg_lag = 3
    m_hi = T - 1 - sg_lag - 2   # = T - 6 with sg_lag = 3
    assert m_hi > m_lo

    for m in range(m_lo, m_hi + 1):
        k = m + lag
        assert not np.isnan(res.agg[m]), f"batch should be valid at m={m}"
        assert not np.isnan(agg_stream[k]), f"stream should be valid at k={k}"
        np.testing.assert_allclose(
            agg_stream[k], res.agg[m], rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            pj_stream[k], res.per_joint[m], rtol=1e-10, atol=1e-10
        )


def test_streaming_reset_is_clean() -> None:
    cfg = _cfg(N=5, J=2)
    rng = np.random.default_rng(6)
    state = rng.standard_normal((40, 2))
    action = _make_action(state)

    sm = StreamingRMSJerk(cfg)
    for k in range(40):
        sm.update(action[k], state[k])
    sm.reset()

    sm2 = StreamingRMSJerk(cfg)
    for k in range(40):
        out = sm.update(action[k], state[k])
        ref = sm2.update(action[k], state[k])
        # Both should produce identical (possibly nan) outputs.
        if np.isnan(ref.agg):
            assert np.isnan(out.agg)
        else:
            assert out.agg == pytest.approx(ref.agg, rel=1e-12, abs=1e-12)


def test_input_shape_validation() -> None:
    cfg = _cfg(N=5, J=3)
    with pytest.raises(ValueError):
        batch(np.zeros((10, 3)), np.zeros((10, 2)), cfg)
    with pytest.raises(ValueError):
        batch(np.zeros((10,)), np.zeros((10,)), cfg)
    sm = StreamingRMSJerk(cfg)
    with pytest.raises(ValueError):
        sm.update(np.zeros(2), np.zeros(3))
