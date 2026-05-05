"""Savitzky-Golay smoothing primitives shared by jerk and SPARC.

Two flavours are exposed:

- :func:`sg_smooth_batch` is a thin wrapper over ``scipy.signal.savgol_filter``
  with ``mode='interp'`` and ``axis=0`` so a ``(T, J)`` matrix is smoothed
  along time. Interior samples (``window <= 2 * lag + 1`` away from either
  edge) get the symmetric SG estimate; boundary samples use a polynomial fit
  that scipy provides automatically.

- :class:`StreamingSavGol` returns the symmetric SG estimate for the centre
  of the most recent ``window`` samples, with ``lag = (window - 1) // 2``
  steps of intrinsic delay. It is numerically identical to the interior of
  ``sg_smooth_batch`` (boundary frames diverge by construction — there is no
  causal way to recover the symmetric output for the first or last ``lag``
  frames).

The streaming class uses pre-computed centred SG coefficients
(`scipy.signal.savgol_coeffs`) and a ``deque`` of length ``window``.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import savgol_coeffs, savgol_filter


def sg_smooth_batch(
    x: np.ndarray,
    window: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
) -> np.ndarray:
    """Apply a symmetric Savitzky-Golay filter along ``axis=0``.

    Parameters
    ----------
    x
        Shape ``(T, J)`` (or 1-D); samples are along the first axis.
    window
        Filter length in samples; must be odd and ``> polyorder``.
    polyorder
        Polynomial order of the local fit.
    deriv
        Order of the derivative to compute. ``0`` = smoothing only.
    delta
        Sample spacing (used for derivative scaling). Pass ``T_s`` to get
        derivatives in physical units when ``deriv > 0``.
    """
    if window <= polyorder:
        raise ValueError(
            f"window ({window}) must exceed polyorder ({polyorder})"
        )
    if window % 2 == 0:
        raise ValueError(f"window ({window}) must be odd")
    if x.ndim == 1:
        return savgol_filter(
            x, window, polyorder, deriv=deriv, delta=delta, mode="interp"
        )
    return savgol_filter(
        x, window, polyorder, deriv=deriv, delta=delta, axis=0, mode="interp"
    )


class StreamingSavGol:
    """Online symmetric Savitzky-Golay smoother with ``(window-1)/2`` lag.

    Each :meth:`update` call ingests one ``(n_channels,)`` sample and returns
    either the smoothed value at the centre of the most recent ``window``
    samples, or ``None`` while the buffer is still warming up.
    """

    def __init__(
        self,
        window: int,
        polyorder: int,
        n_channels: int,
        deriv: int = 0,
        delta: float = 1.0,
    ) -> None:
        if window % 2 == 0:
            raise ValueError(f"window must be odd, got {window}")
        if window <= polyorder:
            raise ValueError(
                f"window ({window}) must exceed polyorder ({polyorder})"
            )
        self.window = window
        self.polyorder = polyorder
        self.n_channels = n_channels
        self.deriv = deriv
        self.delta = delta
        self.lag = (window - 1) // 2
        self._coeffs = np.asarray(
            savgol_coeffs(
                window, polyorder, deriv=deriv, delta=delta, pos=self.lag
            ),
            dtype=np.float64,
        )
        self._buf: deque[np.ndarray] = deque()

    def reset(self) -> None:
        self._buf.clear()

    def update(self, x: np.ndarray) -> np.ndarray | None:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != (self.n_channels,):
            raise ValueError(
                f"expected shape ({self.n_channels},), got {x_arr.shape}"
            )
        self._buf.append(x_arr)
        if len(self._buf) > self.window:
            self._buf.popleft()
        if len(self._buf) < self.window:
            return None
        # buf holds the last `window` samples in chronological order.
        # Stack into (window, n_channels) and apply centred SG coeffs.
        stacked = np.stack(list(self._buf), axis=0)
        return self._coeffs @ stacked
