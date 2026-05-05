"""Tests for ``stability_monitor.metrics.hf_power``."""

from __future__ import annotations

import numpy as np
import pytest

from stability_monitor.config import Config
from stability_monitor.metrics.hf_power import (
    StreamingHFPower,
    batch,
)


def _cfg(N: int = 30, J: int = 6) -> Config:
    return Config(N=N, n_joints=J)


def _make_state_action_for_error(e_signal: np.ndarray, J: int) -> tuple[np.ndarray, np.ndarray]:
    """Build action and state so that e_i[k] = a_i[k-1] - q_i[k] equals
    ``e_signal[k]`` on joint 0 and zero on other joints.

    ``state[k] = -e_signal[k]`` so that with ``action == 0`` we get
    ``e[k] = 0 - (-e_signal[k]) = e_signal[k]``.
    """
    T = len(e_signal)
    action = np.zeros((T, J))
    state = np.zeros((T, J))
    state[:, 0] = -e_signal
    return action, state


def test_8hz_sine_in_error_gives_ratio_near_one() -> None:
    """A pure 8 Hz error sits inside [5, 15] Hz so the ratio should be ~1."""
    cfg = _cfg(N=60, J=3)
    fs = cfg.fs
    T = 200
    t = np.arange(T) / fs
    e = np.sin(2.0 * np.pi * 8.0 * t)
    action, state = _make_state_action_for_error(e, cfg.n_joints)

    res = batch(action, state, cfg)
    valid = ~np.isnan(res.agg)
    assert valid.any()
    # Joint 0 carries all the chatter; aggregate = max across joints = ~1.0
    np.testing.assert_allclose(res.agg[valid], 1.0, atol=1e-6)


def test_1hz_sine_in_error_gives_ratio_near_zero() -> None:
    """A pure 1 Hz error has all power below the 5 Hz crossover."""
    cfg = _cfg(N=60, J=3)
    fs = cfg.fs
    T = 200
    t = np.arange(T) / fs
    e = np.sin(2.0 * np.pi * 1.0 * t)
    action, state = _make_state_action_for_error(e, cfg.n_joints)

    res = batch(action, state, cfg)
    valid = ~np.isnan(res.agg)
    assert valid.any()
    # Below crossover: should be much closer to 0 than to 1.
    assert np.all(res.agg[valid] < 0.05), f"max agg = {res.agg[valid].max():.4f}"


def test_aggregate_is_max_across_joints() -> None:
    """The aggregate should equal np.max of the per-joint vector."""
    cfg = _cfg(N=30, J=4)
    rng = np.random.default_rng(1)
    T = 100
    state = rng.standard_normal((T, 4))
    action = rng.standard_normal((T, 4))
    res = batch(action, state, cfg)
    valid = ~np.isnan(res.agg)
    np.testing.assert_allclose(
        res.agg[valid], res.per_joint[valid].max(axis=1), rtol=1e-12
    )


def test_warmup_boundary() -> None:
    """First valid output is exactly at k = N (window e[1..N] populated)."""
    cfg = _cfg(N=12, J=2)
    rng = np.random.default_rng(2)
    T = 50
    state = rng.standard_normal((T, 2))
    action = rng.standard_normal((T, 2))
    res = batch(action, state, cfg)
    assert np.all(np.isnan(res.agg[: cfg.N]))
    assert np.all(np.isfinite(res.agg[cfg.N :]))


def test_too_short_episode_is_all_nan() -> None:
    cfg = _cfg(N=30, J=6)
    res = batch(np.zeros((10, 6)), np.zeros((10, 6)), cfg)
    assert res.agg.shape == (10,)
    assert np.all(np.isnan(res.agg))


def test_streaming_first_call_returns_nan() -> None:
    cfg = _cfg(N=4, J=2)
    sm = StreamingHFPower(cfg)
    out = sm.update(np.zeros(2), np.zeros(2))
    assert np.isnan(out.agg)


def test_streaming_matches_batch() -> None:
    """Non-negotiable streaming/batch consistency. lag = 0 so streaming[k]
    must equal batch[k] for every k >= N."""
    cfg = _cfg(N=20, J=4)
    T = 200
    rng = np.random.default_rng(3)
    state = rng.standard_normal((T, 4))
    action = rng.standard_normal((T, 4))

    res = batch(action, state, cfg)

    sm = StreamingHFPower(cfg)
    agg_stream = np.full(T, np.nan)
    pj_stream = np.full((T, 4), np.nan)
    for k in range(T):
        out = sm.update(action[k], state[k])
        agg_stream[k] = out.agg
        pj_stream[k] = out.per_joint

    np.testing.assert_array_equal(np.isnan(res.agg), np.isnan(agg_stream))
    valid = ~np.isnan(res.agg)
    np.testing.assert_allclose(
        agg_stream[valid], res.agg[valid], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        pj_stream[valid], res.per_joint[valid], rtol=1e-12, atol=1e-12
    )


def test_streaming_reset_is_clean() -> None:
    cfg = _cfg(N=6, J=2)
    rng = np.random.default_rng(4)
    state = rng.standard_normal((30, 2))
    action = rng.standard_normal((30, 2))
    sm = StreamingHFPower(cfg)
    for k in range(30):
        sm.update(action[k], state[k])
    sm.reset()
    sm2 = StreamingHFPower(cfg)
    for k in range(30):
        a = sm.update(action[k], state[k])
        b = sm2.update(action[k], state[k])
        if np.isnan(b.agg):
            assert np.isnan(a.agg)
        else:
            assert a.agg == pytest.approx(b.agg, rel=1e-12, abs=1e-12)


def test_input_shape_validation() -> None:
    cfg = _cfg(N=10, J=3)
    with pytest.raises(ValueError):
        batch(np.zeros((20, 3)), np.zeros((20, 2)), cfg)
    with pytest.raises(ValueError):
        batch(np.zeros((20,)), np.zeros((20,)), cfg)
    sm = StreamingHFPower(cfg)
    with pytest.raises(ValueError):
        sm.update(np.zeros(2), np.zeros(3))
