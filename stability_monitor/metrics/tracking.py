"""Position-tracking error.

Implements the metric defined in ``stability_monitoring_methods.tex``,
subsection *Position Tracking Error*:

    e_i[k]            = a_i[k - 1] - q_i[k]
    E_i^RMS[k]        = sqrt( (1/N) * sum_{j in W[k]} e_i[j]^2 )
    E^RMS[k]          = sqrt( sum_i (E_i^RMS[k])^2 )

where ``W[k] = {k - N + 1, ..., k}`` is the trailing window of length
``N``.

By convention ``e_i[0]`` is undefined (no ``a[-1]``); it is reported as
``nan``. As a consequence, ``E^RMS[k]`` and ``E_i^RMS[k]`` are ``nan`` for
``k < N`` (the first window that contains only valid errors is
``W[N] = {1, ..., N}``).
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from stability_monitor.config import Config


class TrackingResult(NamedTuple):
    """Aggregate scalar (or vector) and per-joint vector (or matrix).

    For batch mode: ``agg`` shape ``(T,)``, ``per_joint`` shape ``(T, J)``.
    For streaming mode: ``agg`` is a Python ``float``, ``per_joint`` shape
    ``(J,)``.
    """

    agg: np.ndarray | float
    per_joint: np.ndarray


def batch(action: np.ndarray, state: np.ndarray, cfg: Config) -> TrackingResult:
    """Per-frame windowed RMS tracking error over an entire episode.

    Parameters
    ----------
    action
        Commanded joint positions, shape ``(T, J)``.
    state
        Measured joint positions, shape ``(T, J)``.
    cfg
        Monitor configuration; only ``cfg.N`` is consulted here.

    Returns
    -------
    TrackingResult
        ``agg`` of shape ``(T,)`` and ``per_joint`` of shape ``(T, J)``.
        Frames ``k < N`` are ``nan``.
    """
    action = np.asarray(action, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if action.shape != state.shape:
        raise ValueError(
            f"action and state shapes differ: {action.shape} vs {state.shape}"
        )
    if action.ndim != 2:
        raise ValueError(f"expected 2-D action/state, got ndim={action.ndim}")

    T, J = action.shape
    N = cfg.N
    if N <= 0:
        raise ValueError(f"cfg.N must be positive, got {N}")

    # e[k] = a[k-1] - q[k] for k >= 1; e[0] is undefined.
    e = np.full((T, J), np.nan, dtype=np.float64)
    if T >= 2:
        e[1:] = action[:-1] - state[1:]
    e_sq = e * e  # nan-preserving

    per_joint = np.full((T, J), np.nan, dtype=np.float64)
    agg = np.full(T, np.nan, dtype=np.float64)

    if T < N:
        return TrackingResult(agg=agg, per_joint=per_joint)

    # Vectorised sliding-window mean via cumulative sum, with explicit
    # nan-window suppression so warmup remains nan rather than wrapping zero
    # into the running sum.
    e_sq_filled = np.where(np.isnan(e_sq), 0.0, e_sq)
    isnan_int = np.isnan(e_sq).astype(np.int64)

    csum = np.empty((T + 1, J), dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(e_sq_filled, axis=0, out=csum[1:])

    nan_csum = np.empty((T + 1, J), dtype=np.int64)
    nan_csum[0] = 0
    np.cumsum(isnan_int, axis=0, out=nan_csum[1:])

    ks = np.arange(N - 1, T)
    starts = ks - N + 1
    ends = ks + 1

    window_sum = csum[ends] - csum[starts]              # (len(ks), J)
    window_nan = nan_csum[ends] - nan_csum[starts]      # (len(ks), J)
    has_nan = window_nan.sum(axis=1) > 0

    ms = window_sum / N
    pj = np.sqrt(ms)
    pj[has_nan] = np.nan
    per_joint[ks] = pj

    agg_vals = np.sqrt(ms.sum(axis=1))
    agg_vals[has_nan] = np.nan
    agg[ks] = agg_vals

    return TrackingResult(agg=agg, per_joint=per_joint)


class StreamingTrackingError:
    """Online ``E^RMS`` with an ``O(1)`` running sum-of-squares window."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._N = cfg.N
        self._J = cfg.n_joints
        self._prev_action: np.ndarray | None = None
        self._sq_buffer: deque[np.ndarray] = deque()
        self._sum_sq = np.zeros(self._J, dtype=np.float64)

    def reset(self) -> None:
        self._prev_action = None
        self._sq_buffer.clear()
        self._sum_sq[:] = 0.0

    def update(self, a_k: np.ndarray, q_k: np.ndarray) -> TrackingResult:
        a_k_arr = np.asarray(a_k, dtype=np.float64)
        q_k_arr = np.asarray(q_k, dtype=np.float64)
        if a_k_arr.shape != (self._J,) or q_k_arr.shape != (self._J,):
            raise ValueError(
                f"expected per-step shape ({self._J},), got "
                f"action={a_k_arr.shape}, state={q_k_arr.shape}"
            )

        nan_pj = np.full(self._J, np.nan, dtype=np.float64)

        if self._prev_action is None:
            self._prev_action = a_k_arr.copy()
            return TrackingResult(agg=float("nan"), per_joint=nan_pj)

        e = self._prev_action - q_k_arr
        self._prev_action = a_k_arr.copy()

        e_sq = e * e
        self._sq_buffer.append(e_sq)
        self._sum_sq += e_sq
        if len(self._sq_buffer) > self._N:
            self._sum_sq -= self._sq_buffer.popleft()

        if len(self._sq_buffer) < self._N:
            return TrackingResult(agg=float("nan"), per_joint=nan_pj)

        ms = self._sum_sq / self._N
        per_joint = np.sqrt(ms)
        agg = float(np.sqrt(ms.sum()))
        return TrackingResult(agg=agg, per_joint=per_joint)
