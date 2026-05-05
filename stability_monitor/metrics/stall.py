"""Stall and joint-limit clipping indicator.

Implements the metric defined in ``stability_monitoring_methods.tex``,
subsection *Stall Indicator and Joint-Limit Clipping*:

    sigma_i^stall[k] = 1[ |a_i[k] - a_i[k-M]| > delta_a^(i)
                          AND |q_i[k] - q_i[k-M]| < delta_q^(i) ]
    sigma_i^clip[k]  = 1[ a_i[k] <= a_i^min + eps_i
                          OR   a_i[k] >= a_i^max - eps_i ]
                       where eps_i = cfg.eps_clip_frac * (a_i^max - a_i^min)
    sigma_i[k]       = sigma_i^stall[k] OR sigma_i^clip[k]
    sigma_bar[k]     = (1/N) * sum_{j in W[k]} 1[ max_i sigma_i[j] = 1 ]

Convention: for ``k < M`` the look-back is incomplete and we set
``sigma_i^stall[k] = 0``. ``sigma_i^clip`` is always defined, so
``sigma_i[k]`` is defined for every ``k``. The windowed aggregate
``sigma_bar[k]`` therefore has the same warmup as the other metrics
(``k >= N - 1``).

This metric needs the calibrated thresholds — see
:mod:`stability_monitor.calibration`. Unlike the four signal-only metrics,
its API takes a ``Thresholds`` argument in addition to ``Config``.

Causal lag (streaming): ``0`` — the indicator at time ``k`` looks only
backwards.

Per-joint diagnostic ``sigma_i[k]`` is the *windowed mean of the
instantaneous flag* (so it is comparable across joints and
co-aggregateable into ``sigma_bar``). Note that
``sigma_bar != max_i (per-joint window-mean)`` in general — the aggregate
is the window-mean of the any-joint flag, not the max of per-joint
window-means.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from stability_monitor.calibration import Thresholds
from stability_monitor.config import Config


class StallResult(NamedTuple):
    """Aggregate ``sigma_bar`` and per-joint windowed flag rate."""

    agg: np.ndarray | float
    per_joint: np.ndarray


def _sigma_per_joint(
    action: np.ndarray,
    state: np.ndarray,
    cfg: Config,
    thresholds: Thresholds,
) -> np.ndarray:
    """Compute the boolean per-joint indicator ``sigma_i[k]`` for every k.

    Returns shape ``(T, J)`` of bools; for ``k < M`` only the clip
    component is checked (stall component is treated as 0).
    """
    T, J = action.shape
    M = cfg.M

    sigma_clip = (
        (action <= thresholds.a_min + cfg.eps_clip_frac
         * (thresholds.a_max - thresholds.a_min))
        | (action >= thresholds.a_max - cfg.eps_clip_frac
           * (thresholds.a_max - thresholds.a_min))
    )

    sigma_stall = np.zeros((T, J), dtype=bool)
    if T > M:
        d_a = np.abs(action[M:] - action[:-M])
        d_q = np.abs(state[M:] - state[:-M])
        sigma_stall[M:] = (d_a > thresholds.delta_a) & (d_q < thresholds.delta_q)

    return sigma_clip | sigma_stall


def batch(
    action: np.ndarray,
    state: np.ndarray,
    cfg: Config,
    thresholds: Thresholds,
) -> StallResult:
    """Episode-level windowed stall+clip rate ``sigma_bar``."""
    action = np.asarray(action, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if action.shape != state.shape:
        raise ValueError(
            f"action and state shapes differ: {action.shape} vs {state.shape}"
        )
    if action.ndim != 2:
        raise ValueError(f"expected 2-D action/state, got ndim={action.ndim}")

    T, J = action.shape
    if J != thresholds.delta_a.shape[0]:
        raise ValueError(
            f"thresholds.delta_a shape {thresholds.delta_a.shape} does not "
            f"match action joint count {J}"
        )

    sigma_i = _sigma_per_joint(action, state, cfg, thresholds)        # (T, J)
    any_flag = sigma_i.any(axis=1).astype(np.float64)                  # (T,)

    N = cfg.N
    agg = np.full(T, np.nan, dtype=np.float64)
    per_joint = np.full((T, J), np.nan, dtype=np.float64)
    if T < N:
        return StallResult(agg=agg, per_joint=per_joint)

    # Vectorised sliding-window mean via cumulative sum.
    csum_any = np.concatenate(([0.0], np.cumsum(any_flag)))            # (T+1,)
    csum_pj = np.concatenate(
        [np.zeros((1, J), dtype=np.float64),
         np.cumsum(sigma_i.astype(np.float64), axis=0)],
        axis=0,
    )                                                                  # (T+1, J)
    ks = np.arange(N - 1, T)
    starts = ks - N + 1
    ends = ks + 1
    agg[ks] = (csum_any[ends] - csum_any[starts]) / N
    per_joint[ks] = (csum_pj[ends] - csum_pj[starts]) / N

    return StallResult(agg=agg, per_joint=per_joint)


class StreamingStall:
    """Online ``sigma_bar`` with zero causal lag."""

    def __init__(self, cfg: Config, thresholds: Thresholds) -> None:
        if thresholds.delta_a.shape[0] != cfg.n_joints:
            raise ValueError(
                f"thresholds.delta_a shape {thresholds.delta_a.shape} does "
                f"not match cfg.n_joints={cfg.n_joints}"
            )
        self._cfg = cfg
        self._thr = thresholds
        self._N = cfg.N
        self._M = cfg.M
        self._J = cfg.n_joints
        # Look-back buffers: keep the last M+1 actions and states so that
        # the oldest entry is a[k-M] / q[k-M] when index k is the newest.
        self._a_buf: deque[np.ndarray] = deque()
        self._q_buf: deque[np.ndarray] = deque()
        # Window buffers: per-joint flag history and any-flag history.
        self._sigma_i_buf: deque[np.ndarray] = deque()
        self._any_flag_buf: deque[bool] = deque()
        self.lag = 0

    def reset(self) -> None:
        self._a_buf.clear()
        self._q_buf.clear()
        self._sigma_i_buf.clear()
        self._any_flag_buf.clear()

    def update(self, a_k: np.ndarray, q_k: np.ndarray) -> StallResult:
        a_k_arr = np.asarray(a_k, dtype=np.float64)
        q_k_arr = np.asarray(q_k, dtype=np.float64)
        if a_k_arr.shape != (self._J,) or q_k_arr.shape != (self._J,):
            raise ValueError(
                f"expected per-step shape ({self._J},), got "
                f"action={a_k_arr.shape}, state={q_k_arr.shape}"
            )

        self._a_buf.append(a_k_arr)
        self._q_buf.append(q_k_arr)
        if len(self._a_buf) > self._M + 1:
            self._a_buf.popleft()
            self._q_buf.popleft()

        eps = self._cfg.eps_clip_frac * (self._thr.a_max - self._thr.a_min)
        sigma_clip = (
            (a_k_arr <= self._thr.a_min + eps)
            | (a_k_arr >= self._thr.a_max - eps)
        )
        if len(self._a_buf) >= self._M + 1:
            d_a = np.abs(a_k_arr - self._a_buf[0])
            d_q = np.abs(q_k_arr - self._q_buf[0])
            sigma_stall = (d_a > self._thr.delta_a) & (d_q < self._thr.delta_q)
        else:
            sigma_stall = np.zeros(self._J, dtype=bool)
        sigma_i = sigma_clip | sigma_stall

        self._sigma_i_buf.append(sigma_i.astype(np.float64))
        self._any_flag_buf.append(bool(np.any(sigma_i)))
        if len(self._any_flag_buf) > self._N:
            self._sigma_i_buf.popleft()
            self._any_flag_buf.popleft()

        nan_pj = np.full(self._J, np.nan, dtype=np.float64)
        if len(self._any_flag_buf) < self._N:
            return StallResult(agg=float("nan"), per_joint=nan_pj)

        agg = float(sum(self._any_flag_buf)) / self._N
        per_joint = np.mean(
            np.stack(list(self._sigma_i_buf), axis=0), axis=0
        )
        return StallResult(agg=agg, per_joint=per_joint)
