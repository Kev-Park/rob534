"""Tests for ``stability_monitor.metrics.stall`` and ``Thresholds``."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from stability_monitor.calibration import Thresholds
from stability_monitor.config import Config
from stability_monitor.metrics.stall import (
    StreamingStall,
    batch,
)


def _cfg(N: int = 10, M: int = 5, J: int = 2) -> Config:
    return Config(N=N, M=M, n_joints=J)


def _thresholds(
    J: int,
    delta_a: float = 1.0,
    delta_q: float = 0.1,
    a_min: float = -100.0,
    a_max: float = 100.0,
) -> Thresholds:
    return Thresholds(
        theta=np.zeros(5),
        delta_a=np.full(J, delta_a, dtype=np.float64),
        delta_q=np.full(J, delta_q, dtype=np.float64),
        a_min=np.full(J, a_min, dtype=np.float64),
        a_max=np.full(J, a_max, dtype=np.float64),
    )


def test_matched_ramp_does_not_stall() -> None:
    """When command ramps and state ramps to match, neither stall nor clip
    fires, so sigma_bar == 0."""
    cfg = _cfg(N=10, M=5, J=2)
    # Wide limits so the ramp never approaches the clip band.
    thr = _thresholds(J=2, delta_a=1.0, delta_q=0.1, a_min=-10000.0, a_max=10000.0)
    T = 50
    ramp = np.arange(T, dtype=np.float64)
    action = np.stack([ramp * 2.0, ramp * 1.5], axis=1)
    state = action.copy()  # perfect tracking

    res = batch(action, state, cfg, thr)
    valid = ~np.isnan(res.agg)
    np.testing.assert_allclose(res.agg[valid], 0.0)
    np.testing.assert_allclose(res.per_joint[valid], 0.0)


def test_action_ramps_state_static_triggers_stall() -> None:
    """Action ramps fast enough that |a[k]-a[k-M]| > delta_a, but state
    holds at zero (|q[k]-q[k-M]| < delta_q). Stall must fire."""
    cfg = _cfg(N=10, M=5, J=2)
    thr = _thresholds(J=2, delta_a=1.0, delta_q=0.1, a_min=-100.0, a_max=100.0)
    T = 30
    action = np.stack([np.arange(T, dtype=np.float64), np.zeros(T)], axis=1)
    state = np.zeros_like(action)

    res = batch(action, state, cfg, thr)
    # First valid window ends at k = N - 1 = 9. After k = M (=5) the stall
    # is true on joint 0; after the window fully covers post-M frames,
    # sigma_bar should saturate at 1.0.
    assert res.agg[cfg.N + cfg.M] == pytest.approx(1.0)
    # Joint 0 carries the stall; joint 1 has no command motion -> no stall.
    assert res.per_joint[cfg.N + cfg.M, 0] == pytest.approx(1.0)
    assert res.per_joint[cfg.N + cfg.M, 1] == pytest.approx(0.0)


def test_action_at_joint_limit_triggers_clip() -> None:
    """Action saturates at a_max -> sigma_clip == 1 even with no motion."""
    cfg = _cfg(N=10, M=5, J=2)
    a_max = 90.0
    thr = _thresholds(J=2, delta_a=1.0, delta_q=0.1, a_min=-90.0, a_max=a_max)
    T = 30
    # eps = 0.02 * (a_max - a_min) = 0.02 * 180 = 3.6 -> a_max - eps = 86.4
    # action = 88 -> within eps of a_max -> clip fires.
    action = np.full((T, 2), 88.0)
    state = action.copy()

    res = batch(action, state, cfg, thr)
    valid = ~np.isnan(res.agg)
    np.testing.assert_allclose(res.agg[valid], 1.0)


def test_warmup_boundary() -> None:
    """sigma_bar is nan for k < N-1 and finite from k = N-1 on."""
    cfg = _cfg(N=8, M=3, J=2)
    thr = _thresholds(J=2)
    rng = np.random.default_rng(0)
    T = 30
    action = rng.standard_normal((T, 2))
    state = rng.standard_normal((T, 2))
    res = batch(action, state, cfg, thr)
    assert np.all(np.isnan(res.agg[: cfg.N - 1]))
    assert np.all(np.isfinite(res.agg[cfg.N - 1 :]))


def test_too_short_episode_is_all_nan() -> None:
    cfg = _cfg(N=20, J=3)
    thr = _thresholds(J=3)
    res = batch(np.zeros((10, 3)), np.zeros((10, 3)), cfg, thr)
    assert np.all(np.isnan(res.agg))


def test_streaming_matches_batch() -> None:
    """Non-negotiable streaming/batch consistency. lag = 0."""
    cfg = _cfg(N=15, M=5, J=4)
    rng = np.random.default_rng(1)
    T = 200
    action = rng.standard_normal((T, 4)) * 5.0
    state = rng.standard_normal((T, 4)) * 5.0
    thr = Thresholds(
        theta=np.zeros(5),
        delta_a=np.full(4, 2.0),
        delta_q=np.full(4, 1.0),
        a_min=np.full(4, -10.0),
        a_max=np.full(4, 10.0),
    )

    res = batch(action, state, cfg, thr)

    sm = StreamingStall(cfg, thr)
    agg_s = np.full(T, np.nan)
    pj_s = np.full((T, 4), np.nan)
    for k in range(T):
        out = sm.update(action[k], state[k])
        agg_s[k] = out.agg
        pj_s[k] = out.per_joint

    np.testing.assert_array_equal(np.isnan(res.agg), np.isnan(agg_s))
    valid = ~np.isnan(res.agg)
    np.testing.assert_allclose(
        agg_s[valid], res.agg[valid], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        pj_s[valid], res.per_joint[valid], rtol=1e-12, atol=1e-12
    )


def test_streaming_reset_clears_buffers() -> None:
    cfg = _cfg(N=5, M=3, J=2)
    thr = _thresholds(J=2)
    rng = np.random.default_rng(2)
    state = rng.standard_normal((30, 2))
    action = rng.standard_normal((30, 2))
    sm = StreamingStall(cfg, thr)
    for k in range(30):
        sm.update(action[k], state[k])
    sm.reset()
    sm2 = StreamingStall(cfg, thr)
    for k in range(30):
        a = sm.update(action[k], state[k])
        b = sm2.update(action[k], state[k])
        if np.isnan(b.agg):
            assert np.isnan(a.agg)
        else:
            assert a.agg == pytest.approx(b.agg, rel=1e-12, abs=1e-12)


def test_input_shape_validation() -> None:
    cfg = _cfg(J=3)
    thr = _thresholds(J=3)
    with pytest.raises(ValueError):
        batch(np.zeros((20, 3)), np.zeros((20, 2)), cfg, thr)
    with pytest.raises(ValueError):
        batch(np.zeros((20, 2)), np.zeros((20, 2)), cfg, thr)  # J mismatch
    sm = StreamingStall(cfg, thr)
    with pytest.raises(ValueError):
        sm.update(np.zeros(2), np.zeros(3))


def test_thresholds_save_load_roundtrip() -> None:
    thr = Thresholds(
        theta=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        delta_a=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        delta_q=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
        a_min=np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]),
        a_max=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "thr.json"
        thr.save(p)
        loaded = Thresholds.load(p)
        np.testing.assert_array_equal(thr.theta, loaded.theta)
        np.testing.assert_array_equal(thr.delta_a, loaded.delta_a)
        np.testing.assert_array_equal(thr.delta_q, loaded.delta_q)
        np.testing.assert_array_equal(thr.a_min, loaded.a_min)
        np.testing.assert_array_equal(thr.a_max, loaded.a_max)
        # File is plain JSON
        payload = json.loads(p.read_text())
        assert set(payload.keys()) == {"theta", "delta_a", "delta_q", "a_min", "a_max"}


def test_thresholds_placeholder_does_not_fire_stall() -> None:
    """The placeholder factory uses delta_a=inf and delta_q=0 so the stall
    branch can never fire; combined with a_min=-inf, a_max=inf, the clip
    branch never fires either."""
    cfg = _cfg(N=10, M=5, J=2)
    thr = Thresholds.placeholder(n_joints=2)
    rng = np.random.default_rng(3)
    T = 60
    action = rng.standard_normal((T, 2)) * 100.0  # huge motion
    state = rng.standard_normal((T, 2)) * 100.0
    res = batch(action, state, cfg, thr)
    valid = ~np.isnan(res.agg)
    np.testing.assert_allclose(res.agg[valid], 0.0)
