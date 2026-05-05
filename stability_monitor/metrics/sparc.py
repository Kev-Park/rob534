"""Spectral Arc Length (SPARC) of the speed magnitude.

Implements the metric defined in ``stability_monitoring_methods.tex``,
subsection *Spectral Arc Length (SPARC)*:

    v[k]                = || dot{q_tilde}[k] ||_2
    V_hat(omega)        = | DFT( hann . v_window )(omega) | / | DFT(...)(0) |
    omega_c             = min( omega_c_max,
                               max{ omega : V_hat(omega) > V_bar } )
    eta_SPARC[k]        = - integral_{0}^{omega_c}
                              sqrt( (1/omega_c)^2 + (d V_hat / d omega)^2 )

The arc-length integral is computed in its canonical discrete form (the
piecewise-linear arc length of the spectrum, i.e. the form used by
Balasubramanian et al., *J. NeuroEng. Rehab.* 2015):

    SPARC = - sum_n sqrt( (df_n / omega_c)^2 + (d V_hat_n)^2 ).

This is mathematically equivalent to ``np.trapz`` on the slope-based
integrand to first order; the piecewise-linear arc length is preferred
because it is exactly the path length of the discrete spectrum.

The smoothed first derivative is taken in a single pass via
``scipy.signal.savgol_filter(deriv=1, delta=T_s)`` so the lag of the
streaming SPARC is just the SG centring lag (``(window-1)//2 = 3``).

Sign-aligned signal for the monitor's ``m`` vector is ``-eta_SPARC`` so
that "larger means worse" — see the methods doc.

Numerical guards:
  - if ``V[0] == 0`` (stationary arm) -> nan,
  - if no frequency bin above DC exceeds ``V_bar`` -> nan,
  - episodes shorter than ``cfg.N`` -> all nan.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from stability_monitor.config import Config
from stability_monitor.smoothing import StreamingSavGol, sg_smooth_batch


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _sparc_window(
    v_window: np.ndarray, fs: float, omega_c_max: float, V_bar: float
) -> float:
    """Single SPARC value for one 1-D speed-magnitude window.

    Returns ``nan`` for the degenerate cases described in the module
    docstring.
    """
    n = len(v_window)
    if n < 2:
        return float("nan")

    # Hann taper, zero-pad to next power of two, take the magnitude rfft.
    hann = np.hanning(n)
    tapered = v_window * hann
    nfft = _next_pow2(n)
    spectrum = np.abs(np.fft.rfft(tapered, n=nfft))
    if spectrum[0] == 0.0 or not np.isfinite(spectrum[0]):
        return float("nan")

    V_hat = spectrum / spectrum[0]
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Restrict to f <= omega_c_max BEFORE picking the adaptive cutoff.
    cap_mask = freqs <= omega_c_max
    f_cap = freqs[cap_mask]
    V_cap = V_hat[cap_mask]

    # Adaptive cutoff: largest index in [0, end] where V_hat exceeds V_bar.
    # f[0] = 0 and V_hat[0] = 1 always exceeds V_bar = 0.05, so this set is
    # never empty. ``cutoff_idx == 0`` means the spectrum collapses to DC --
    # the arc length is then degenerate (omega_c == 0); we return nan.
    above = np.flatnonzero(V_cap > V_bar)
    cutoff_idx = int(above[-1])
    if cutoff_idx == 0:
        return float("nan")

    f_sel = f_cap[: cutoff_idx + 1]
    V_sel = V_cap[: cutoff_idx + 1]
    omega_c = f_sel[-1] - f_sel[0]   # = f_sel[-1] (DC bin is at 0)
    if omega_c <= 0:
        return float("nan")

    df = np.diff(f_sel) / omega_c
    dV = np.diff(V_sel)
    arc = float(np.sum(np.sqrt(df * df + dV * dV)))
    return -arc


def batch(action: np.ndarray, state: np.ndarray, cfg: Config) -> np.ndarray:
    """Episode-level windowed SPARC of the SG-smoothed speed magnitude.

    ``action`` is unused (kept for the uniform metric API) but its shape is
    validated against ``state``. Returns shape ``(T,)``; warmup frames are
    ``nan`` (see the agreement-zone discussion in the test).
    """
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if action.shape != state.shape:
        raise ValueError(
            f"action and state shapes differ: {action.shape} vs {state.shape}"
        )
    if state.ndim != 2:
        raise ValueError(f"expected 2-D state, got ndim={state.ndim}")

    T = state.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    if T < cfg.N:
        return out

    q_dot = sg_smooth_batch(
        state, cfg.sg_window, cfg.sg_polyorder, deriv=1, delta=cfg.Ts
    )
    speed = np.linalg.norm(q_dot, axis=1)  # shape (T,)

    for k in range(cfg.N - 1, T):
        win = speed[k - cfg.N + 1 : k + 1]
        out[k] = _sparc_window(win, cfg.fs, cfg.omega_c_max, cfg.V_bar)
    return out


class StreamingSPARC:
    """Online SPARC with a 3-sample causal lag.

    Returns the SPARC value at time index ``k - lag`` where ``k`` is the
    number of samples ingested so far and ``lag = (sg_window - 1) // 2``.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._N = cfg.N
        self._J = cfg.n_joints
        self._sg = StreamingSavGol(
            cfg.sg_window,
            cfg.sg_polyorder,
            cfg.n_joints,
            deriv=1,
            delta=cfg.Ts,
        )
        self._speed_buf: deque[float] = deque()
        self.lag = self._sg.lag  # = 3 with default sg_window = 7

    def reset(self) -> None:
        self._sg.reset()
        self._speed_buf.clear()

    def update(self, a_k: np.ndarray, q_k: np.ndarray) -> float:
        a_k_arr = np.asarray(a_k, dtype=np.float64)
        q_k_arr = np.asarray(q_k, dtype=np.float64)
        if a_k_arr.shape != (self._J,) or q_k_arr.shape != (self._J,):
            raise ValueError(
                f"expected per-step shape ({self._J},), got "
                f"action={a_k_arr.shape}, state={q_k_arr.shape}"
            )

        v_vec = self._sg.update(q_k_arr)
        if v_vec is None:
            return float("nan")
        speed = float(np.linalg.norm(v_vec))
        self._speed_buf.append(speed)
        if len(self._speed_buf) > self._N:
            self._speed_buf.popleft()
        if len(self._speed_buf) < self._N:
            return float("nan")

        win = np.asarray(self._speed_buf, dtype=np.float64)
        return _sparc_window(
            win, self._cfg.fs, self._cfg.omega_c_max, self._cfg.V_bar
        )
