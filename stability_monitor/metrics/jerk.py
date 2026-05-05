"""RMS jerk on the Savitzky-Golay-smoothed measured trajectory.

Implements the metric defined in ``stability_monitoring_methods.tex``,
subsection *Jerk-Based Smoothness*:

    dddot{q_tilde}_i[k] approx
        ( q_tilde[k+2] - 2 q_tilde[k+1] + 2 q_tilde[k-1] - q_tilde[k-2] )
        / (2 T_s^3)

    J_i^RMS[k] = sqrt( (1/N) * sum_{j in W[k]} dddot{q_tilde}_i[j]^2 )
    J^RMS[k]   = sqrt( sum_i (J_i^RMS[k])^2 )

Smoothing uses a symmetric SG (window 7, polyorder 3) — see
:mod:`stability_monitor.smoothing`.

Causal-lag convention (streaming).
The five-point stencil reads ``j+2`` ahead, so ``dddot{q}[j]`` cannot be
emitted until two more samples have arrived. Combined with the SG centring
delay of ``(window - 1) // 2 = 3`` samples, the streaming class therefore
emits ``J^RMS[k - 5]`` when sample ``k`` arrives. The :attr:`lag` attribute
makes this explicit so the top-level monitor can align signals.

Boundary frames where ``sg_smooth_batch`` falls back to its polynomial
interpolation diverge from the streaming output by construction; the
streaming/batch numerical-equality contract holds on the strictly-symmetric
interior (``m in [N+4, T-6]``).
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from stability_monitor.config import Config
from stability_monitor.smoothing import StreamingSavGol, sg_smooth_batch


class JerkResult(NamedTuple):
    """Aggregate scalar (or vector) and per-joint vector (or matrix)."""

    agg: np.ndarray | float
    per_joint: np.ndarray


def _jerk_from_qtilde(q_tilde: np.ndarray, Ts: float) -> np.ndarray:
    """Five-point central third-derivative on a smoothed trajectory.

    Parameters
    ----------
    q_tilde
        Shape ``(T, J)`` smoothed positions.
    Ts
        Sample period in seconds.

    Returns
    -------
    np.ndarray
        Shape ``(T, J)`` jerk array; the first two and last two rows are
        ``nan`` because the stencil is undefined there.
    """
    T, J = q_tilde.shape
    out = np.full((T, J), np.nan, dtype=np.float64)
    if T < 5:
        return out
    coeff = 1.0 / (2.0 * Ts**3)
    out[2 : T - 2] = coeff * (
        q_tilde[4:T] - 2.0 * q_tilde[3 : T - 1]
        + 2.0 * q_tilde[1 : T - 3] - q_tilde[: T - 4]
    )
    return out


def _windowed_rms_aggregate(
    sq: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window mean over an ``(T, J)`` array, with explicit nan guard.

    Returns ``(agg, per_joint)`` with shapes ``(T,)`` and ``(T, J)``;
    frames whose window contains any nan are themselves nan.
    """
    T, J = sq.shape
    per_joint = np.full((T, J), np.nan, dtype=np.float64)
    agg = np.full(T, np.nan, dtype=np.float64)
    if T < N:
        return agg, per_joint

    sq_filled = np.where(np.isnan(sq), 0.0, sq)
    isnan_int = np.isnan(sq).astype(np.int64)

    csum = np.empty((T + 1, J), dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(sq_filled, axis=0, out=csum[1:])

    nan_csum = np.empty((T + 1, J), dtype=np.int64)
    nan_csum[0] = 0
    np.cumsum(isnan_int, axis=0, out=nan_csum[1:])

    ks = np.arange(N - 1, T)
    starts = ks - N + 1
    ends = ks + 1
    window_sum = csum[ends] - csum[starts]
    window_nan = nan_csum[ends] - nan_csum[starts]
    has_nan = window_nan.sum(axis=1) > 0

    ms = window_sum / N
    pj = np.sqrt(ms)
    pj[has_nan] = np.nan
    per_joint[ks] = pj

    agg_vals = np.sqrt(ms.sum(axis=1))
    agg_vals[has_nan] = np.nan
    agg[ks] = agg_vals

    return agg, per_joint


def batch(action: np.ndarray, state: np.ndarray, cfg: Config) -> JerkResult:
    """Episode-level windowed RMS jerk on the SG-smoothed state.

    The ``action`` argument is unused — kept for the uniform metric API —
    but its shape is validated against ``state``.
    """
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if action.shape != state.shape:
        raise ValueError(
            f"action and state shapes differ: {action.shape} vs {state.shape}"
        )
    if state.ndim != 2:
        raise ValueError(f"expected 2-D state, got ndim={state.ndim}")

    q_tilde = sg_smooth_batch(state, cfg.sg_window, cfg.sg_polyorder)
    j = _jerk_from_qtilde(q_tilde, cfg.Ts)
    agg, per_joint = _windowed_rms_aggregate(j * j, cfg.N)
    return JerkResult(agg=agg, per_joint=per_joint)


class StreamingRMSJerk:
    """Online windowed RMS jerk with explicit causal lag.

    The output of :meth:`update` corresponds to the time index ``k - lag``
    where ``k`` is the count of samples that have been ingested and
    ``lag = (sg_window - 1) // 2 + 2 = 5`` for the default configuration.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._N = cfg.N
        self._J = cfg.n_joints
        self._Ts = cfg.Ts
        self._sg = StreamingSavGol(cfg.sg_window, cfg.sg_polyorder, cfg.n_joints)
        self._qtilde_buf: deque[np.ndarray] = deque()  # last 5 q_tilde samples
        self._jerk_sq_buf: deque[np.ndarray] = deque()  # last N jerk^2 vectors
        self.lag = self._sg.lag + 2  # SG centring + stencil forward reach

    def reset(self) -> None:
        self._sg.reset()
        self._qtilde_buf.clear()
        self._jerk_sq_buf.clear()

    def update(self, a_k: np.ndarray, q_k: np.ndarray) -> JerkResult:
        # ``a_k`` is unused but validated for the uniform interface.
        a_k_arr = np.asarray(a_k, dtype=np.float64)
        q_k_arr = np.asarray(q_k, dtype=np.float64)
        if a_k_arr.shape != (self._J,) or q_k_arr.shape != (self._J,):
            raise ValueError(
                f"expected per-step shape ({self._J},), got "
                f"action={a_k_arr.shape}, state={q_k_arr.shape}"
            )

        nan_pj = np.full(self._J, np.nan, dtype=np.float64)
        nan_result = JerkResult(agg=float("nan"), per_joint=nan_pj)

        q_tilde_k = self._sg.update(q_k_arr)
        if q_tilde_k is None:
            return nan_result
        self._qtilde_buf.append(q_tilde_k)
        if len(self._qtilde_buf) > 5:
            self._qtilde_buf.popleft()
        if len(self._qtilde_buf) < 5:
            return nan_result

        # qt[0..4] correspond to q_tilde at [m-2, m-1, m, m+1, m+2] where
        # m is the time index 5 samples back from the current input.
        qt = self._qtilde_buf
        jerk_m = (
            (qt[4] - 2.0 * qt[3] + 2.0 * qt[1] - qt[0])
            / (2.0 * self._Ts**3)
        )
        jerk_sq = jerk_m * jerk_m
        self._jerk_sq_buf.append(jerk_sq)
        if len(self._jerk_sq_buf) > self._N:
            self._jerk_sq_buf.popleft()

        if len(self._jerk_sq_buf) < self._N:
            return nan_result

        # Recompute the window mean from scratch each step. Spec §5
        # explicitly permits this for jerk because N is small; it avoids
        # the floating-point cancellation that dogs a long-running sum.
        ms = np.mean(np.stack(list(self._jerk_sq_buf), axis=0), axis=0)
        per_joint = np.sqrt(ms)
        agg = float(np.sqrt(ms.sum()))
        return JerkResult(agg=agg, per_joint=per_joint)
