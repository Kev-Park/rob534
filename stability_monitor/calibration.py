"""Calibrated thresholds for the stability monitor.

Exposes:

- :class:`Thresholds` — dataclass that carries the calibration outputs.
- :func:`calibrate` — pools episodes from a teacher corpus and fits the
  thresholds per ``stability_monitoring_methods.tex``.

The two stages of calibration:

1. Per-joint calibration of the stall/clip parameters from raw action and
   state arrays (no metric outputs needed):

   - ``delta_a^(i) = P95 over the corpus of |a_i[k] - a_i[k-M]|``
   - ``delta_q^(i) = P95 over the corpus of |q_i[k] - q_i[k-M]|``
   - ``a_min^(i), a_max^(i)`` from the empirical action range plus a
     symmetric margin of ``margin_frac * (a_max - a_min)``.

2. Aggregate-channel calibration of ``theta`` (5,) by running all five
   metrics in batch on the corpus with a *placeholder* :class:`Thresholds`
   — ``theta_l = P95`` of the pooled non-NaN values of channel ``l``.
   Spec §6 explicitly asks for placeholder thresholds at this stage so
   ``sigma_bar`` is identically zero on the success corpus and any
   stall/clip event at runtime triggers intervention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from stability_monitor.config import Config


@dataclass(frozen=True)
class Thresholds:
    """Calibrated 95th-percentile thresholds.

    Attributes
    ----------
    theta
        Shape ``(5,)``; per-channel intervention threshold for the
        ``m`` vector ``[E^RMS, J^RMS, -SPARC, rho^HF, sigma_bar]``. Filled
        in by :func:`scripts.calibrate.calibrate`. Until then a zero
        placeholder is acceptable for the stall metric, which only reads
        the ``delta_*`` and ``a_*`` fields.
    delta_a
        Shape ``(J,)``; per-joint command-magnitude threshold used in
        ``sigma_i^stall`` — 95th percentile of ``|a_i[k] - a_i[k-M]|``
        on the teacher corpus.
    delta_q
        Shape ``(J,)``; per-joint state-motion threshold used in
        ``sigma_i^stall`` — 95th percentile of ``|q_i[k] - q_i[k-M]|``.
    a_min, a_max
        Shape ``(J,)``; per-joint commanded-position range used in
        ``sigma_i^clip`` (empirical min/max plus a small margin).
    """

    theta: np.ndarray
    delta_a: np.ndarray
    delta_q: np.ndarray
    a_min: np.ndarray
    a_max: np.ndarray

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "theta": self.theta.tolist(),
            "delta_a": self.delta_a.tolist(),
            "delta_q": self.delta_q.tolist(),
            "a_min": self.a_min.tolist(),
            "a_max": self.a_max.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        payload = json.loads(Path(path).read_text())
        return cls(
            theta=np.asarray(payload["theta"], dtype=np.float64),
            delta_a=np.asarray(payload["delta_a"], dtype=np.float64),
            delta_q=np.asarray(payload["delta_q"], dtype=np.float64),
            a_min=np.asarray(payload["a_min"], dtype=np.float64),
            a_max=np.asarray(payload["a_max"], dtype=np.float64),
        )

    @classmethod
    def placeholder(cls, n_joints: int) -> "Thresholds":
        """A clearly-marked placeholder for development before calibration.

        Useful in unit tests and as the default during the initial
        plumb-through. ``theta`` is zero (no intervention threshold yet);
        ``delta_*`` and ``a_*`` are filled with finite-but-huge values
        chosen so the stall and clip predicates never fire under normal
        inputs. Finite values are used instead of ``inf`` so that
        ``a_max - a_min`` does not produce ``nan``.
        """
        big = 1e18
        return cls(
            theta=np.zeros(5, dtype=np.float64),
            delta_a=np.full(n_joints, big, dtype=np.float64),
            delta_q=np.full(n_joints, 0.0, dtype=np.float64),
            a_min=np.full(n_joints, -big, dtype=np.float64),
            a_max=np.full(n_joints, big, dtype=np.float64),
        )


# The five aggregate channels of the monitor vector ``m``, in order.
# Sign-aligned per the methods doc so "larger means worse" for each.
M_CHANNELS: tuple[str, ...] = (
    "E_RMS",     # tracking error aggregate
    "J_RMS",     # jerk aggregate
    "neg_SPARC", # -eta_SPARC (so smoother trajectories give smaller m)
    "rho_HF",    # HF power ratio aggregate (max across joints)
    "sigma_bar", # stall + clip windowed rate
)


def _per_channel_batch(
    action: np.ndarray, state: np.ndarray, cfg: Config, thresholds: "Thresholds"
) -> dict[str, np.ndarray]:
    """Compute the (T,) aggregate for each of the five m-vector channels.

    Imported lazily to avoid a circular import with the metric modules.
    """
    from stability_monitor.metrics import (
        hf_power as _hf,
        jerk as _jerk,
        sparc as _sparc,
        stall as _stall,
        tracking as _tracking,
    )

    return {
        "E_RMS":     _tracking.batch(action, state, cfg).agg,
        "J_RMS":     _jerk.batch(action, state, cfg).agg,
        "neg_SPARC": -_sparc.batch(action, state, cfg),
        "rho_HF":    _hf.batch(action, state, cfg).agg,
        "sigma_bar": _stall.batch(action, state, cfg, thresholds).agg,
    }


def calibrate(
    episodes: Iterable[tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    margin_frac: float = 0.05,
    percentile: float = 95.0,
) -> tuple[Thresholds, dict[str, dict[str, float]]]:
    """Fit :class:`Thresholds` from a corpus of (action, state) episode pairs.

    Parameters
    ----------
    episodes
        Iterable of ``(action, state)`` tuples; each is a ``(T_i, J)``
        array. The iterable is consumed *twice* (once for the stall/clip
        per-joint calibration, once for the per-channel ``theta``), so
        callers should pass a list — not a one-shot generator.
    cfg
        Monitor config.
    margin_frac
        Symmetric margin added to the empirical action range when fitting
        ``[a_min, a_max]``.
    percentile
        Percentile rule (default 95). Same value used for ``delta_*`` and
        ``theta_*``.

    Returns
    -------
    Thresholds
        Fitted thresholds.
    dict
        Per-channel summary statistics: median, P95, P99, max for each of
        the five channels (handy for sanity-checking before adopting).
    """
    episodes_list = list(episodes)
    if not episodes_list:
        raise ValueError("calibrate(): no episodes supplied")

    # Pass 1: stall/clip per-joint thresholds from raw action/state.
    M = cfg.M
    da_chunks: list[np.ndarray] = []
    dq_chunks: list[np.ndarray] = []
    a_chunks: list[np.ndarray] = []
    for action, state in episodes_list:
        if action.shape != state.shape or action.ndim != 2:
            raise ValueError(
                f"unexpected episode shapes: action {action.shape}, "
                f"state {state.shape}"
            )
        if action.shape[0] > M:
            da_chunks.append(np.abs(action[M:] - action[:-M]))
            dq_chunks.append(np.abs(state[M:] - state[:-M]))
        a_chunks.append(action)

    if not da_chunks:
        raise ValueError(
            f"all corpus episodes shorter than M={M} samples; cannot "
            f"calibrate stall thresholds"
        )

    da_pool = np.concatenate(da_chunks, axis=0)
    dq_pool = np.concatenate(dq_chunks, axis=0)
    a_pool = np.concatenate(a_chunks, axis=0)

    delta_a = np.percentile(da_pool, percentile, axis=0).astype(np.float64)
    delta_q = np.percentile(dq_pool, percentile, axis=0).astype(np.float64)
    a_lo = a_pool.min(axis=0).astype(np.float64)
    a_hi = a_pool.max(axis=0).astype(np.float64)
    rng = a_hi - a_lo
    a_min = a_lo - margin_frac * rng
    a_max = a_hi + margin_frac * rng

    # Pass 2: theta from the five m-vector channels with placeholder stall
    # thresholds (per spec §6: sigma_bar is identically zero on the success
    # corpus, so any runtime stall/clip event triggers intervention).
    placeholder = Thresholds.placeholder(cfg.n_joints)
    pooled: dict[str, list[np.ndarray]] = {ch: [] for ch in M_CHANNELS}
    for action, state in episodes_list:
        ch_vals = _per_channel_batch(action, state, cfg, placeholder)
        for ch, arr in ch_vals.items():
            pooled[ch].append(np.asarray(arr, dtype=np.float64))

    theta = np.zeros(len(M_CHANNELS), dtype=np.float64)
    summary: dict[str, dict[str, float]] = {}
    for i, ch in enumerate(M_CHANNELS):
        vals = np.concatenate(pooled[ch])
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            theta[i] = 0.0
            summary[ch] = {
                "n_finite": 0,
                "median": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }
            continue
        theta[i] = float(np.percentile(finite, percentile))
        summary[ch] = {
            "n_finite": int(finite.size),
            "median": float(np.median(finite)),
            "p95": float(np.percentile(finite, 95)),
            "p99": float(np.percentile(finite, 99)),
            "max": float(finite.max()),
        }

    thresholds = Thresholds(
        theta=theta,
        delta_a=delta_a,
        delta_q=delta_q,
        a_min=a_min,
        a_max=a_max,
    )
    return thresholds, summary
