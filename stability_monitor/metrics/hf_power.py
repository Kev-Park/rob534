"""High-frequency power ratio of the tracking error.

Implements the metric defined in ``stability_monitoring_methods.tex``,
subsection *High-Frequency Power Ratio of the Tracking Error*:

    rho_i^HF[k] = integral_{omega_h}^{omega_Nyq} S_{e_i}(omega, k) d omega
                  / integral_{0}^{omega_Nyq}      S_{e_i}(omega, k) d omega
    rho^HF[k]    = max_i rho_i^HF[k]

where ``S_{e_i}`` is the Welch PSD of the per-joint tracking error
``e_i[k] = a_i[k-1] - q_i[k]`` over the trailing window of length ``cfg.N``
samples. We use ``scipy.signal.welch`` with ``nperseg = N`` and
``noverlap = N//2`` (per spec §5); with a window of exactly ``N`` samples
this reduces to a single Hann-tapered periodogram.

Causal lag (streaming): ``0`` — the error itself is causal up to the
one-step action-state alignment, and the Welch periodogram needs no
forward samples. The first valid output is at ``k = N`` (window
``[1, N]`` of ``e``-values).

Per-joint guard: when the denominator integral is exactly zero (a joint
with no error variance — e.g. a synthetic zero-error channel), the ratio
is reported as ``0`` ("no power, no chatter") rather than ``nan``. The
aggregate uses ``np.max`` rather than ``np.nanmax`` because all per-joint
values are finite by construction.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np
from scipy.signal import welch

from stability_monitor.config import Config


class HFPowerResult(NamedTuple):
    """Aggregate and per-joint HF power ratios."""

    agg: np.ndarray | float
    per_joint: np.ndarray


def _hf_ratio_window(
    e_window: np.ndarray, fs: float, omega_h: float, nperseg: int
) -> np.ndarray:
    """Per-joint HF power ratio for a single ``(N, J)`` error window."""
    if e_window.ndim != 2:
        raise ValueError(f"expected 2-D error window, got ndim={e_window.ndim}")
    n, J = e_window.shape
    f, Sxx = welch(
        e_window,
        fs=fs,
        nperseg=min(nperseg, n),
        noverlap=min(nperseg // 2, n // 2),
        axis=0,
    )
    # Sxx has shape (n_freqs, J).
    nyq = fs / 2.0
    num_mask = (f >= omega_h) & (f <= nyq)
    den_mask = (f >= 0.0) & (f <= nyq)
    f_num = f[num_mask]
    f_den = f[den_mask]
    if f_num.size < 2 or f_den.size < 2:
        return np.full(J, np.nan, dtype=np.float64)
    num = np.trapezoid(Sxx[num_mask, :], f_num, axis=0)
    den = np.trapezoid(Sxx[den_mask, :], f_den, axis=0)
    return np.where(den > 0.0, num / np.where(den == 0.0, 1.0, den), 0.0)


def batch(
    action: np.ndarray, state: np.ndarray, cfg: Config
) -> HFPowerResult:
    """Episode-level windowed HF power ratio of the tracking error."""
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
    per_joint = np.full((T, J), np.nan, dtype=np.float64)
    agg = np.full(T, np.nan, dtype=np.float64)

    # e[k] = a[k-1] - q[k]; e[0] undefined.
    if T < N + 1:
        return HFPowerResult(agg=agg, per_joint=per_joint)

    e = np.full((T, J), np.nan, dtype=np.float64)
    e[1:] = action[:-1] - state[1:]

    # First fully-populated, all-finite window ends at k = N (uses e[1..N]).
    for k in range(N, T):
        win = e[k - N + 1 : k + 1]
        if np.any(np.isnan(win)):
            continue
        rho_per_joint = _hf_ratio_window(win, cfg.fs, cfg.omega_h, N)
        per_joint[k] = rho_per_joint
        agg[k] = float(np.max(rho_per_joint))

    return HFPowerResult(agg=agg, per_joint=per_joint)


class StreamingHFPower:
    """Online HF power ratio with zero causal lag (output at time ``k``)."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._N = cfg.N
        self._J = cfg.n_joints
        self._prev_action: np.ndarray | None = None
        self._e_buf: deque[np.ndarray] = deque()
        self.lag = 0

    def reset(self) -> None:
        self._prev_action = None
        self._e_buf.clear()

    def update(self, a_k: np.ndarray, q_k: np.ndarray) -> HFPowerResult:
        a_k_arr = np.asarray(a_k, dtype=np.float64)
        q_k_arr = np.asarray(q_k, dtype=np.float64)
        if a_k_arr.shape != (self._J,) or q_k_arr.shape != (self._J,):
            raise ValueError(
                f"expected per-step shape ({self._J},), got "
                f"action={a_k_arr.shape}, state={q_k_arr.shape}"
            )

        nan_pj = np.full(self._J, np.nan, dtype=np.float64)
        nan_result = HFPowerResult(agg=float("nan"), per_joint=nan_pj)

        if self._prev_action is None:
            self._prev_action = a_k_arr.copy()
            return nan_result

        e = self._prev_action - q_k_arr
        self._prev_action = a_k_arr.copy()

        self._e_buf.append(e)
        if len(self._e_buf) > self._N:
            self._e_buf.popleft()
        if len(self._e_buf) < self._N:
            return nan_result

        win = np.stack(list(self._e_buf), axis=0)
        rho_per_joint = _hf_ratio_window(
            win, self._cfg.fs, self._cfg.omega_h, self._N
        )
        agg = float(np.max(rho_per_joint))
        return HFPowerResult(agg=agg, per_joint=rho_per_joint)
