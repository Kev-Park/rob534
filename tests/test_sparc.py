"""Tests for ``stability_monitor.metrics.sparc``."""

from __future__ import annotations

import numpy as np
import pytest

from stability_monitor.config import Config
from stability_monitor.metrics.sparc import (
    StreamingSPARC,
    _sparc_window,
    batch,
)


def _cfg(N: int = 30, J: int = 6) -> Config:
    return Config(N=N, n_joints=J)


def _min_jerk_speed(T: int) -> np.ndarray:
    """30 * tau^2 * (1-tau)^2 — speed profile of a min-jerk reach."""
    tau = np.arange(T) / (T - 1)
    return 30.0 * tau**2 * (1.0 - tau) ** 2


def test_min_jerk_reach_is_smooth() -> None:
    """SPARC of a min-jerk speed profile must lie in a smooth-trajectory
    range. The exact value depends on padding/cutoff conventions; we use
    generous bounds rather than pin to Balasubramanian's reference -1.6
    because the methods doc fixes ``zero-pad to next power of two`` (much
    less padding than Balasubramanian's ``padlevel=4`` reference).
    """
    cfg = _cfg(N=30, J=1)
    v = _min_jerk_speed(cfg.N)
    s = _sparc_window(v, cfg.fs, cfg.omega_c_max, cfg.V_bar)
    assert -2.0 < s < -1.0, f"min-jerk SPARC {s:.3f} outside smooth-range bounds"


def test_noise_makes_sparc_more_negative() -> None:
    cfg = _cfg(N=60, J=1)
    rng = np.random.default_rng(0)
    smooth = _min_jerk_speed(cfg.N)
    noisy = smooth + 0.5 * rng.standard_normal(cfg.N)
    s_smooth = _sparc_window(smooth, cfg.fs, cfg.omega_c_max, cfg.V_bar)
    s_noisy = _sparc_window(noisy, cfg.fs, cfg.omega_c_max, cfg.V_bar)
    assert s_noisy < s_smooth, (
        f"expected noisy ({s_noisy:.3f}) more negative than smooth "
        f"({s_smooth:.3f})"
    )


def test_zero_speed_returns_nan() -> None:
    cfg = _cfg(N=30, J=1)
    s = _sparc_window(np.zeros(cfg.N), cfg.fs, cfg.omega_c_max, cfg.V_bar)
    assert np.isnan(s)


def test_batch_warmup_and_shape() -> None:
    cfg = _cfg(N=15, J=3)
    T = 80
    rng = np.random.default_rng(1)
    state = np.cumsum(rng.standard_normal((T, 3)) * 0.1, axis=0)  # smooth-ish
    action = state.copy()
    out = batch(action, state, cfg)
    assert out.shape == (T,)
    assert np.all(np.isnan(out[: cfg.N - 1]))
    assert np.any(np.isfinite(out[cfg.N - 1 :]))


def test_too_short_episode_is_all_nan() -> None:
    cfg = _cfg(N=30, J=2)
    out = batch(np.zeros((10, 2)), np.zeros((10, 2)), cfg)
    assert out.shape == (10,)
    assert np.all(np.isnan(out))


def test_streaming_warmup_returns_nan() -> None:
    cfg = _cfg(N=10, J=2)
    sm = StreamingSPARC(cfg)
    rng = np.random.default_rng(2)
    # SG lag 3 + N - 1 = 12 updates before first finite output is even possible
    for _ in range(12):
        out = sm.update(rng.standard_normal(2), rng.standard_normal(2))
        assert np.isnan(out)


def test_streaming_matches_batch_on_interior() -> None:
    """Non-negotiable consistency check on the symmetric-SG interior.

    Streaming uses the centred SG-with-deriv=1, so its first-valid speed
    sample lives at ``m = sg_lag``; batch's symmetric SG region is
    ``m in [sg_lag, T - 1 - sg_lag]``. SPARC over window ``[m-N+1, m]``
    therefore agrees in the zone ``m in [N + sg_lag - 1, T - 1 - sg_lag]``.
    """
    cfg = _cfg(N=12, J=4)
    T = 200
    rng = np.random.default_rng(3)
    # Use a smooth-ish trajectory so SPARC values are finite throughout.
    t = np.arange(T) / cfg.fs
    state = (
        np.sin(2.0 * np.pi * 0.3 * t)[:, None]
        * rng.standard_normal((1, 4))
        + 0.1 * rng.standard_normal((T, 4))
    )
    action = state.copy()

    res_batch = batch(action, state, cfg)

    sm = StreamingSPARC(cfg)
    sg_lag = sm.lag  # = 3
    out_stream = np.full(T, np.nan, dtype=np.float64)
    for k in range(T):
        out_stream[k] = sm.update(action[k], state[k])

    m_lo = cfg.N + sg_lag - 1
    m_hi = T - 1 - sg_lag
    assert m_hi > m_lo
    # Both should be finite or both nan (if the spectrum is degenerate at
    # this window). Check finite frames agree numerically, and that nan
    # patterns coincide.
    for m in range(m_lo, m_hi + 1):
        k = m + sg_lag
        b, s = res_batch[m], out_stream[k]
        if np.isnan(b):
            assert np.isnan(s)
        else:
            assert not np.isnan(s)
            assert s == pytest.approx(b, rel=1e-10, abs=1e-10)


def test_streaming_reset_is_clean() -> None:
    cfg = _cfg(N=8, J=2)
    rng = np.random.default_rng(4)
    state = np.cumsum(rng.standard_normal((40, 2)) * 0.1, axis=0)
    action = state.copy()
    sm = StreamingSPARC(cfg)
    for k in range(40):
        sm.update(action[k], state[k])
    sm.reset()
    sm2 = StreamingSPARC(cfg)
    for k in range(40):
        a = sm.update(action[k], state[k])
        b = sm2.update(action[k], state[k])
        if np.isnan(b):
            assert np.isnan(a)
        else:
            assert a == pytest.approx(b, rel=1e-12, abs=1e-12)


def test_input_shape_validation() -> None:
    cfg = _cfg(N=10, J=3)
    with pytest.raises(ValueError):
        batch(np.zeros((20, 3)), np.zeros((20, 2)), cfg)
    with pytest.raises(ValueError):
        batch(np.zeros((20,)), np.zeros((20,)), cfg)
    sm = StreamingSPARC(cfg)
    with pytest.raises(ValueError):
        sm.update(np.zeros(2), np.zeros(3))
