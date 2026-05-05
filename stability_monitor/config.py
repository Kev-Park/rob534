"""Static configuration for the stability monitor.

All values come from ``stability_monitoring_methods.tex``. The defaults match
the SO-101 setup at f_s = 30 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Sampling and windowing parameters for all five metrics.

    Attributes
    ----------
    fs
        Sampling rate in Hz.
    N
        Sliding-window length in samples (``Delta = N * T_s``).
    M
        Stall-indicator look-back in samples (~167 ms at 30 Hz).
    P
        Persistence-filter length in samples (~100 ms at 30 Hz).
    sg_window, sg_polyorder
        Savitzky-Golay smoothing applied to ``observation.state`` before
        differentiation.
    omega_h
        Crossover frequency (Hz) for the HF-power-ratio numerator integration.
    omega_c_max
        Cap (Hz) for the SPARC adaptive cutoff omega_c.
    V_bar
        Threshold on the normalised speed spectrum used to pick omega_c.
    eps_clip_frac
        Per-joint clip tolerance as a fraction of the calibrated
        ``[a_min, a_max]`` range.
    n_joints, joint_names
        Joint count and labels in the SO-101 order.
    """

    fs:            float = 30.0
    N:             int   = 30
    M:             int   = 5
    P:             int   = 3
    sg_window:     int   = 7
    sg_polyorder: int    = 3
    omega_h:       float = 5.0
    omega_c_max:   float = 12.0
    V_bar:         float = 0.05
    eps_clip_frac: float = 0.02
    n_joints:      int   = 6
    joint_names: tuple[str, ...] = field(
        default_factory=lambda: (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
    )

    @property
    def Ts(self) -> float:
        """Sample period (s)."""
        return 1.0 / self.fs
