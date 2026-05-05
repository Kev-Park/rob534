"""Tests for ``stability_monitor.metrics.tracking``.

Synthetic-signal correctness checks plus the streaming/batch consistency
contract that the spec calls out as non-negotiable.
"""

from __future__ import annotations

import numpy as np
import pytest

from stability_monitor.config import Config
from stability_monitor.metrics.tracking import (
    StreamingTrackingError,
    batch,
)


def _cfg(N: int = 30, J: int = 6) -> Config:
    """Config override that lets tests vary N and J."""
    # Frozen dataclass — recreate via ``__class__`` kwargs.
    return Config(N=N, n_joints=J)


def test_constant_offset_matches_offset_norm() -> None:
    """Constant a-q offset c => E_i^RMS = |c_i|, E^RMS = ||c||_2."""
    N, J, T = 30, 6, 200
    rng = np.random.default_rng(0)
    c = rng.standard_normal(J).astype(np.float64) * 2.0
    state = rng.standard_normal((T, J)).astype(np.float64)
    # action[k] = state[k+1] + c so that e[k] = action[k-1] - state[k] = c
    # i.e. for k>=1: state[k] = action[k-1] - c.
    action = np.empty_like(state)
    # Build action so that action[k-1] - state[k] == c for k >= 1.
    action[:-1] = state[1:] + c
    action[-1] = state[-1] + c  # arbitrary final command, ignored by metric
    cfg = _cfg(N=N, J=J)

    res = batch(action, state, cfg)
    assert res.agg.shape == (T,)
    assert res.per_joint.shape == (T, J)

    # Warmup nans.
    assert np.all(np.isnan(res.agg[:N]))
    assert np.all(np.isnan(res.per_joint[:N]))

    expected_pj = np.broadcast_to(np.abs(c), res.per_joint[N:].shape)
    np.testing.assert_allclose(res.per_joint[N:], expected_pj, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        res.agg[N:],
        np.full(T - N, np.linalg.norm(c)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_perfect_tracking_is_zero() -> None:
    """If a[k-1] == q[k] then E^RMS == 0 after warmup."""
    N, J, T = 10, 3, 50
    rng = np.random.default_rng(1)
    state = rng.standard_normal((T, J))
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    cfg = _cfg(N=N, J=J)

    res = batch(action, state, cfg)
    np.testing.assert_allclose(res.agg[N:], 0.0, atol=1e-15)
    np.testing.assert_allclose(res.per_joint[N:], 0.0, atol=1e-15)


def test_shape_and_warmup_boundary() -> None:
    """First valid frame is k=N exactly (window fully populated)."""
    N, J, T = 5, 2, 12
    rng = np.random.default_rng(2)
    action = rng.standard_normal((T, J))
    state = rng.standard_normal((T, J))
    cfg = _cfg(N=N, J=J)
    res = batch(action, state, cfg)

    # k = 0..N-1 -> nan ; k = N..T-1 -> finite (errors are random, not zero)
    assert np.all(np.isnan(res.agg[:N]))
    assert np.all(np.isfinite(res.agg[N:]))


def test_aggregate_equals_norm_of_per_joint() -> None:
    """E^RMS[k] should equal ||(E_i^RMS[k])_i||_2."""
    N, J, T = 8, 4, 40
    rng = np.random.default_rng(3)
    action = rng.standard_normal((T, J))
    state = rng.standard_normal((T, J))
    cfg = _cfg(N=N, J=J)
    res = batch(action, state, cfg)
    valid = ~np.isnan(res.agg)
    expected = np.linalg.norm(res.per_joint[valid], axis=1)
    np.testing.assert_allclose(res.agg[valid], expected, rtol=1e-12, atol=1e-12)


def test_too_short_episode_is_all_nan() -> None:
    """If T < N every output frame is nan, and shapes still match."""
    N, J, T = 30, 6, 10
    rng = np.random.default_rng(4)
    action = rng.standard_normal((T, J))
    state = rng.standard_normal((T, J))
    cfg = _cfg(N=N, J=J)
    res = batch(action, state, cfg)
    assert res.agg.shape == (T,)
    assert res.per_joint.shape == (T, J)
    assert np.all(np.isnan(res.agg))
    assert np.all(np.isnan(res.per_joint))


def test_streaming_first_call_returns_nan() -> None:
    """The first update() call has no previous action, so e is undefined."""
    cfg = _cfg(N=4, J=3)
    sm = StreamingTrackingError(cfg)
    a0 = np.zeros(3)
    q0 = np.zeros(3)
    out = sm.update(a0, q0)
    assert np.isnan(out.agg)
    assert np.all(np.isnan(out.per_joint))


def test_streaming_matches_batch_random() -> None:
    """The non-negotiable consistency check across the full episode."""
    N, J, T = 17, 6, 300
    rng = np.random.default_rng(5)
    action = rng.standard_normal((T, J)).astype(np.float64)
    state = rng.standard_normal((T, J)).astype(np.float64)
    cfg = _cfg(N=N, J=J)

    res = batch(action, state, cfg)

    sm = StreamingTrackingError(cfg)
    agg_stream = np.empty(T, dtype=np.float64)
    pj_stream = np.empty((T, J), dtype=np.float64)
    for k in range(T):
        out = sm.update(action[k], state[k])
        agg_stream[k] = out.agg
        pj_stream[k] = out.per_joint

    # Both must produce nans at the same positions.
    np.testing.assert_array_equal(np.isnan(res.agg), np.isnan(agg_stream))
    np.testing.assert_array_equal(np.isnan(res.per_joint), np.isnan(pj_stream))

    valid = ~np.isnan(res.agg)
    np.testing.assert_allclose(
        agg_stream[valid], res.agg[valid], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        pj_stream[valid], res.per_joint[valid], rtol=1e-12, atol=1e-12
    )


def test_streaming_reset_is_clean() -> None:
    """After reset() the streaming object is indistinguishable from fresh."""
    cfg = _cfg(N=5, J=2)
    rng = np.random.default_rng(6)
    a = rng.standard_normal((20, 2))
    q = rng.standard_normal((20, 2))

    sm = StreamingTrackingError(cfg)
    for k in range(20):
        sm.update(a[k], q[k])
    sm.reset()

    sm2 = StreamingTrackingError(cfg)
    for k in range(20):
        out_reset = sm.update(a[k], q[k])
        out_fresh = sm2.update(a[k], q[k])
        if np.isnan(out_fresh.agg):
            assert np.isnan(out_reset.agg)
        else:
            assert out_reset.agg == pytest.approx(out_fresh.agg, rel=1e-12, abs=1e-12)
        np.testing.assert_array_equal(
            np.isnan(out_reset.per_joint), np.isnan(out_fresh.per_joint)
        )
        valid = ~np.isnan(out_fresh.per_joint)
        np.testing.assert_allclose(
            out_reset.per_joint[valid],
            out_fresh.per_joint[valid],
            rtol=1e-12,
            atol=1e-12,
        )


def test_input_shape_validation() -> None:
    cfg = _cfg(N=5, J=6)
    with pytest.raises(ValueError):
        batch(np.zeros((10, 6)), np.zeros((10, 5)), cfg)
    with pytest.raises(ValueError):
        batch(np.zeros((10,)), np.zeros((10,)), cfg)
    sm = StreamingTrackingError(cfg)
    with pytest.raises(ValueError):
        sm.update(np.zeros(5), np.zeros(6))
