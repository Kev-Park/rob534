"""
Gantt-style timeline figure for the first N episodes of do_ab_eval().

Usage
-----
    from ab_eval_timeline import AbEvalTimeline

    tl = AbEvalTimeline(n_episodes=10)
    do_ab_eval(..., timeline=tl)
    # figure is saved automatically after episode N, or call manually:
    tl.plot("my_run.png", show=True)
"""

from __future__ import annotations

import contextlib
import threading
import time as _time
from dataclasses import dataclass, field

# ── colour palette ─────────────────────────────────────────────────────────────

SPAN_COLORS: dict[str, str] = {
    "go_home":           "#4477aa",
    "save_wait":         "#aa6622",
    "record_A":          "#00aacc",
    "record_B":          "#cc44cc",
    "warmup":            "#334488",
    "dwell":             "#cccc00",
    "vote_inflight":     "#cc2222",
    "motion_active":     "#00bb66",
    "vote_wait":         "#ff8833",
    "monitor_active":    "#1a5c2a",   # dark green — judges ON and scoring
    "monitor_suppress":  "#5c1a1a",   # dark red   — judges OFF (B intervention)
}

# label → (color, matplotlib marker)
MARK_STYLES: dict[str, tuple[str, str]] = {
    "ep_start":          ("#666666",  "|"),
    "metrics_start":     ("#44aaff",  "^"),
    "first_action":      ("#00ff88",  "|"),
    "motion_fire":       ("#ff6600",  "x"),
    "vote_result_Y":     ("#ff3333",  "D"),
    "vote_result_N":     ("#33cc33",  "D"),
    "auto_switch":       ("#ffff00",  "*"),
    "monitor_reset":     ("#aaaaaa",  "s"),
    "monitor_suppress":  ("#ff4444",  "v"),
    "monitor_resume":    ("#44ff44",  "h"),
}

# Which swim-lane a mark is anchored to (None → stagger area above all lanes).
# Lane-anchored marks are drawn as a short vertical tick AT that lane's
# y-position, making the connection obvious.  Float marks appear in the
# rotating stagger rows above the plot.
MARK_LANES: dict[str, str | None] = {
    "ep_start":          None,          # episode-wide → stagger
    "metrics_start":     "monitor",     # at monitor lane
    "first_action":      "motion",      # at timer lane
    "motion_fire":       "motion",      # at timer lane
    "vote_result_Y":     "vote",        # at vote lane
    "vote_result_N":     "vote",        # at vote lane
    "auto_switch":       None,          # episode-wide → stagger
    "monitor_reset":     "monitor",     # at monitor lane
    "monitor_suppress":  "monitor",
    "monitor_resume":    "monitor",
}

# swim-lane internal name → y-index (0 = bottom)
LANE_Y: dict[str, int] = {
    "save":    0,
    "vote":    1,
    "monitor": 2,
    "motion":  3,
    "phase":   4,
}
# Display labels on the y-axis (rename "motion" → "timer" for clarity)
_LANE_DISPLAY: dict[str, str] = {
    "save":    "save",
    "vote":    "vote",
    "monitor": "monitor",
    "motion":  "timer",
    "phase":   "phase",
}
_LANE_NAMES   = {v: k for k, v in LANE_Y.items()}
N_LANES       = len(LANE_Y)
LANE_H        = 0.30   # bar height in data units

# Float marks (None lane) live above the top lane in rotating rows.
_MARK_BASE_OFFSET = 0.18
_MARK_ROW_STEP    = 0.55
_N_MARK_ROWS      = 3
_MARK_MIN_GAP_S   = 2.0


# ── data containers ────────────────────────────────────────────────────────────

@dataclass
class _Span:
    label:   str
    color:   str
    lane:    int
    t_start: float
    t_end:   float = 0.0
    epoch_s: float = 0.0   # absolute perf_counter at episode start (epoch-safe span_end)


@dataclass
class _Mark:
    label:  str
    color:  str
    marker: str
    t:      float
    lane_y: int | None = None   # None → float into stagger area


@dataclass
class _EpLog:
    ep_num:     int
    t_ep_start: float
    spans: list[_Span] = field(default_factory=list)
    marks: list[_Mark] = field(default_factory=list)


# ── main class ─────────────────────────────────────────────────────────────────

class AbEvalTimeline:
    """Thread-safe event recorder → Gantt timeline figure."""

    def __init__(
        self,
        n_episodes: int = 10,
        out_path:   str  = "ab_eval_timeline.png",
        auto_plot:  bool = True,
    ):
        self._n        = n_episodes
        self._out      = out_path
        self._autoplot = auto_plot
        self._lock     = threading.Lock()
        self._logs:    list[_EpLog] = []
        self._cur:     _EpLog | None = None
        self.done      = threading.Event()

    # ── episode lifecycle ──────────────────────────────────────────────────────

    def episode_start(self, ep_num: int) -> bool:
        if ep_num > self._n:
            return False
        log = _EpLog(ep_num=ep_num, t_ep_start=_time.perf_counter())
        with self._lock:
            self._logs.append(log)
            self._cur = log
        self._put_mark(log, "ep_start", *MARK_STYLES["ep_start"])
        return True

    def episode_end(self, ep_num: int):
        if ep_num >= self._n:
            self._cur = None
            self.done.set()
            if self._autoplot:
                threading.Thread(
                    target=self.plot, args=(self._out,), daemon=True
                ).start()

    # ── spans ──────────────────────────────────────────────────────────────────

    def span_start(self, lane: str, label: str, color: str) -> _Span | None:
        log = self._cur
        if log is None:
            return None
        now = _time.perf_counter()
        s = _Span(
            label=label, color=color,
            lane=LANE_Y.get(lane, 0),
            t_start=now - log.t_ep_start,
            epoch_s=log.t_ep_start,
        )
        with self._lock:
            log.spans.append(s)
        return s

    def span_end(self, span: _Span | None, log: _EpLog | None = None):
        """Close an open span.  Uses the span's own epoch — safe to call after
        _cur has advanced to the next episode."""
        if span is None:
            return
        if span.epoch_s:
            span.t_end = _time.perf_counter() - span.epoch_s
        else:
            ref = self._cur
            if ref is None:
                return
            span.t_end = _time.perf_counter() - ref.t_ep_start

    @contextlib.contextmanager
    def span(self, lane: str, label: str, color: str):
        s = self.span_start(lane, label, color)
        try:
            yield s
        finally:
            self.span_end(s)

    # ── marks ──────────────────────────────────────────────────────────────────

    def _put_mark(self, log: _EpLog, label: str, color: str, marker: str):
        lane_name = MARK_LANES.get(label)
        lane_y    = LANE_Y.get(lane_name) if lane_name else None
        m = _Mark(
            label=label, color=color, marker=marker,
            t=_time.perf_counter() - log.t_ep_start,
            lane_y=lane_y,
        )
        with self._lock:
            log.marks.append(m)

    def mark(self, label: str, color: str = "#aaaaaa", marker: str = "^"):
        log = self._cur
        if log is None:
            return
        self._put_mark(log, label, color, marker)

    # ── named shortcuts ────────────────────────────────────────────────────────

    def mark_metrics_start(self):
        self.mark("metrics_start",    *MARK_STYLES["metrics_start"])

    def mark_first_action(self):
        self.mark("first_action",     *MARK_STYLES["first_action"])

    def mark_motion_fire(self):
        self.mark("motion_fire",      *MARK_STYLES["motion_fire"])

    def mark_vote_result(self, struggling: bool):
        k = "vote_result_Y" if struggling else "vote_result_N"
        self.mark(k, *MARK_STYLES[k])

    def mark_auto_switch(self):
        self.mark("auto_switch",      *MARK_STYLES["auto_switch"])

    def mark_monitor_reset(self):
        self.mark("monitor_reset",    *MARK_STYLES["monitor_reset"])

    def mark_monitor_suppress(self):
        self.mark("monitor_suppress", *MARK_STYLES["monitor_suppress"])

    def mark_monitor_resume(self):
        self.mark("monitor_resume",   *MARK_STYLES["monitor_resume"])

    # ── figure ─────────────────────────────────────────────────────────────────

    def plot(self, path: str | None = None, show: bool = False):
        """Render and save the Gantt timeline figure."""
        import matplotlib
        matplotlib.use("TkAgg" if show else "Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines   import Line2D

        path = path or self._out
        with self._lock:
            logs = list(self._logs)
        if not logs:
            print("[AbEvalTimeline] no data to plot")
            return

        # Height above top lane for floating (non-lane) marks
        float_area_h = _N_MARK_ROWS * _MARK_ROW_STEP + 0.3

        n = len(logs)
        subplot_h = N_LANES * 0.85 + float_area_h + 1.2
        fig, axes = plt.subplots(
            n, 1,
            figsize=(17, subplot_h * n),
            facecolor="#111111",
            squeeze=False,
        )
        fig.suptitle(
            "do_ab_eval() — Episode Timing Breakdown",
            color="#dddddd", fontsize=12, fontweight="bold", y=1.01,
        )

        for i, log in enumerate(logs):
            ax = axes[i, 0]
            ax.set_facecolor("#0d1117")
            for sp in ax.spines.values():
                sp.set_color("#2a2a2a")
            ax.tick_params(colors="#888888", labelsize=8)
            ax.set_title(
                f"Episode {log.ep_num}",
                color="#cccccc", fontsize=9, loc="left", pad=4,
            )
            ax.set_xlabel("seconds from episode start", color="#888888", fontsize=8)

            # y-axis: use display names (motion → timer)
            ax.set_yticks(range(N_LANES))
            ax.set_yticklabels(
                [_LANE_DISPLAY.get(_LANE_NAMES.get(y, ""), str(y))
                 for y in range(N_LANES)],
                fontsize=8, color="#aaaaaa",
            )
            top_lane_top = (N_LANES - 1) + LANE_H / 2
            ax.set_ylim(-0.6, top_lane_top + float_area_h)
            ax.grid(axis="x", color="#1e2530", linewidth=0.5, zorder=0)

            t_max = max(
                [s.t_end for s in log.spans if s.t_end > 0] +
                [m.t     for m in log.marks             ] +
                [1.0],
            )
            ax.set_xlim(-0.5, t_max + 4)

            # ── draw spans ────────────────────────────────────────────────────
            for s in log.spans:
                dur = s.t_end - s.t_start
                if dur < 0.02:
                    continue
                ax.barh(
                    s.lane, dur,
                    left=s.t_start, height=LANE_H,
                    color=s.color, alpha=0.85, align="center", zorder=2,
                )
                if dur > 0.35:
                    ax.text(
                        s.t_start + dur / 2, s.lane,
                        f"{s.label}\n{dur:.1f}s",
                        ha="center", va="center",
                        fontsize=6.5, color="#ffffff",
                        fontweight="bold", zorder=3, clip_on=True,
                    )

            # ── separate marks into lane-anchored and floating ────────────────
            marks_anchored = [m for m in log.marks if m.lane_y is not None]
            marks_float    = [m for m in log.marks if m.lane_y is None]

            # ── draw lane-anchored marks AT their lane y-position ─────────────
            for m in marks_anchored:
                ly = m.lane_y
                half = LANE_H * 0.65
                # Short vertical tick spanning the lane bar height
                ax.plot([m.t, m.t], [ly - half, ly + half],
                        color=m.color, linewidth=1.8, zorder=6, clip_on=True)
                # Symbol at lane centre
                ax.scatter([m.t], [ly], marker=m.marker,
                           color=m.color, s=70, zorder=7, clip_on=False,
                           edgecolors="#000000", linewidths=0.4)
                # Label just above the lane bar (short, not rotated)
                ax.text(
                    m.t + 0.18, ly + half + 0.03, m.label,
                    rotation=28, fontsize=6, color=m.color,
                    va="bottom", ha="left", clip_on=False, zorder=8,
                )
                # Thin full-height dashed rule so the eye can follow down
                ax.axvline(m.t, color=m.color, linewidth=0.7,
                           linestyle=":", alpha=0.35, zorder=3)

            # ── draw floating marks in stagger rows above the plot ────────────
            mark_base_y = top_lane_top + _MARK_BASE_OFFSET
            last_used   = [-1e9] * _N_MARK_ROWS

            for m in sorted(marks_float, key=lambda m: m.t):
                row = _N_MARK_ROWS - 1
                for r in range(_N_MARK_ROWS):
                    if m.t - last_used[r] >= _MARK_MIN_GAP_S:
                        row = r
                        break
                last_used[row] = m.t
                mark_y = mark_base_y + row * _MARK_ROW_STEP

                ax.axvline(m.t, color=m.color, linewidth=0.9,
                           linestyle="--", alpha=0.45, zorder=4)
                ax.plot([m.t, m.t], [top_lane_top, mark_y - 0.07],
                        color=m.color, linewidth=0.8, alpha=0.6,
                        zorder=5, clip_on=False)
                ax.scatter([m.t], [mark_y], marker=m.marker,
                           color=m.color, s=65, zorder=6, clip_on=False)
                ax.text(
                    m.t + 0.20, mark_y + 0.06, m.label,
                    rotation=28, fontsize=6.5, color=m.color,
                    va="bottom", ha="left", clip_on=False, zorder=7,
                )

        # ── legend ────────────────────────────────────────────────────────────
        legend_items = (
            [Patch(color=c, label=k, alpha=0.85)
             for k, c in SPAN_COLORS.items()] +
            [Line2D([0], [0], marker=mk, color=cl, linestyle="--",
                    label=lbl, markersize=7)
             for lbl, (cl, mk) in MARK_STYLES.items()]
        )
        fig.legend(
            handles=legend_items,
            loc="lower center", ncol=5,
            fontsize=7, facecolor="#1a1a1a",
            labelcolor="#cccccc", edgecolor="#444444",
            bbox_to_anchor=(0.5, -0.04),
        )

        fig.tight_layout(rect=[0, 0.05, 1, 0.99])
        fig.savefig(path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[AbEvalTimeline] saved → {path}")
        if show:
            plt.show()
        plt.close(fig)
