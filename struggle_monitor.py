"""
struggle_monitor — Gemini-based interrupt detector for robot manipulation policies
==================================================================================

Motivation
----------
Imitation-learning policies for robot manipulation can fail silently. When a
pick-and-place policy starts looping, drops the block repeatedly, or gets stuck,
a human operator needs to notice and switch to a different policy. Detecting this
automatically with classical heuristics alone is hard: jerk and Lyapunov metrics
flag erratic motion but lack the semantic understanding to tell apart a hesitant
but ultimately successful grasp from a genuine failure.

We use Gemini as a vision-language supervisor. Every few seconds it receives a
short clip (12 frames evenly sampled from a 20-second rolling window) alongside
two supplementary text blocks derived from the robot's sensor stream:

  * Stability block  — rolling jerk and Lyapunov trend computed from joint actions
                       and observation states.
From the frames each judge returns ``struggling`` (bool), ``confidence`` (0–1),
and ``reason``. The panel aggregates votes into ``interrupt_probability`` (vote
fraction). The interrupt decision is driven by the metric-based S score:

  1. **S score** — computed from 5 motion channels (tracking error, jerk, SPARC,
     HF power, stall/clip), each normalized by its P95 calibration threshold.
     ``is_struggling()`` fires when S crosses ``interrupt_threshold``.
  2. **Gemini** — called only when S >= threshold to supply a human-readable
     reason; does not affect the interrupt decision.

Quick start
-----------
    monitor = LiveStruggleMonitor(key_file=r"C:/path/to/gem.txt")
    monitor.start()

    # inside your control loop (called every step):
    monitor.push_frame(bgr_frame)               # numpy BGR uint8
    monitor.push_observation(action, state)     # 1-D float32 arrays

    if monitor.is_struggling():
        detail = monitor.get_interrupt()        # {interrupt_probability, vote_count, votes, reason}
        switch_policy()

    monitor.stop()

Batch evaluation
----------------
    python struggle_monitor.py batch --dataset nc8304/my-dataset --video-dir ./vids

    Runs the interrupt monitor over every episode in a LeRobot dataset, compares
    to ground-truth labels, and optionally renders annotated 3-panel videos.
"""

import concurrent.futures
import statistics as _stats
import glob
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from huggingface_hub import snapshot_download

from analyze_rollout import load_api_key
from stability_monitor.calibration import Thresholds, _per_channel_batch, M_CHANNELS
from stability_monitor.config import Config


# ── Module-level cache for pooled thresholds ─────────────────────────────────
_POOLED_THRESHOLDS: Thresholds | None = None


def _get_pooled_thresholds() -> Thresholds:
    """Load and cache the pooled calibration thresholds."""
    global _POOLED_THRESHOLDS
    if _POOLED_THRESHOLDS is None:
        path = Path(__file__).parent / "thresholds" / "pooled.json"
        _POOLED_THRESHOLDS = Thresholds.load(path)
    return _POOLED_THRESHOLDS


_SCORE_MODES = ("max", "mean", "median", "mode")
# Weighted mode is also supported via the string "weighted:<spec>" where
# <spec> is a comma-separated list of  ChannelName=multiplier  pairs.
# Channels not mentioned default to multiplier 1.  Examples:
#   "weighted:sigma_bar=3,E_RMS=2"
#   "weighted:J_RMS=0.5,neg_SPARC=0,rho_HF=0"  (focus on jerk only)
# Channel names (in order): E_RMS, J_RMS, neg_SPARC, rho_HF, sigma_bar


def _aggregate_scores(scores: list[float], mode: str) -> float:
    """Aggregate a list of per-channel scores into a single S value.

    Args:
        scores: Per-channel contribution values (already normalized or raw),
                in M_CHANNELS order.
        mode:   "max", "mean", "median", "mode", or "weighted:<spec>".
                Weighted spec: comma-separated ChannelName=multiplier pairs;
                missing channels default to 1.  Multipliers are normalized
                to sum to 1 before computing the weighted mean.

    Returns:
        Scalar S value.
    """
    if not scores:
        return 0.0
    if mode == "mean":
        return _stats.mean(scores)
    if mode == "median":
        return _stats.median(scores)
    if mode == "mode":
        # Round to 2 dp to create discrete buckets for continuous values
        return float(_stats.mode(round(s, 2) for s in scores))
    if mode.startswith("weighted:"):
        weights = [1.0] * len(scores)
        for part in mode[len("weighted:"):].split(","):
            part = part.strip()
            if "=" not in part:
                continue
            ch_name, w_str = part.split("=", 1)
            ch_name = ch_name.strip()
            try:
                idx = list(M_CHANNELS).index(ch_name)
                if idx < len(weights):
                    weights[idx] = max(0.0, float(w_str.strip()))
            except (ValueError, IndexError):
                pass  # unknown channel — silently ignore
        total_w = sum(weights) or 1.0
        return sum(s * w for s, w in zip(scores, weights)) / total_w
    return max(scores)   # "max" (default)


def compute_struggle_score(
    actions: np.ndarray,
    states: np.ndarray,
    thresholds: Thresholds | None = None,
    cfg: Config | None = None,
    score_mode: str = "mean",
) -> dict:
    """Compute a struggle score from buffered actions/states using calibrated metrics.

    Uses the same 5-channel m-vector as the calibration pipeline (tracking error,
    jerk, SPARC, HF power, stall/clip) and normalizes each by its P95 threshold.
    S > 1.0 means the worst channel exceeds the 95th percentile of normal operation.

    Parameters
    ----------
    actions    : (T, 6) float array of recent commanded actions.
    states     : (T, 6) float array of recent observation states.
    thresholds : Calibrated Thresholds; defaults to pooled.json.
    cfg        : Monitor Config; defaults to Config().
    score_mode : How to aggregate per-channel scores into S.
                 One of "max" (default), "mean", "median", "mode".
                 The sigma_bar (Stall) channel always contributes its raw value;
                 all other channels contribute their normalized ratio (m / theta).

    Returns
    -------
    dict with keys:
        channels : dict mapping channel name -> raw metric value (last time step)
        ratios   : dict mapping channel name -> normalized ratio (m / theta)
        thetas   : dict mapping channel name -> calibration threshold theta
        scores   : dict mapping channel name -> per-channel S contribution
                   (raw for sigma_bar, normalized ratio for all others)
        S        : float, aggregated score across channels
    """
    if thresholds is None:
        thresholds = _get_pooled_thresholds()
    if cfg is None:
        cfg = Config()

    # Run the 5-channel batch computation (same as calibration)
    ch_arrays = _per_channel_batch(actions, states, cfg, thresholds)

    channels: dict[str, float] = {}
    ratios: dict[str, float] = {}
    thetas: dict[str, float] = {}
    scores: dict[str, float] = {}   # per-channel S contribution

    for i, ch_name in enumerate(M_CHANNELS):
        arr = np.asarray(ch_arrays[ch_name], dtype=np.float64)
        # Take the last non-NaN value (some metrics leave trailing NaNs due to
        # forward-stencil warmup, e.g. J_RMS leaves the last 2 rows NaN)
        valid = arr[~np.isnan(arr)]
        raw = float(valid[-1]) if valid.size > 0 else float("nan")
        channels[ch_name] = raw
        # NaN means the metric couldn't be computed (e.g. stationary arm →
        # SPARC collapses to DC). Treat as 0 so it doesn't poison S.
        val = 0.0 if (raw != raw) else raw   # nan != nan

        theta_i = thresholds.theta[i]
        thetas[ch_name] = float(theta_i)
        if ch_name == "sigma_bar":
            # sigma_bar cannot be normalised (theta=0 by design — any stall is a
            # violation). Use the raw windowed stall rate, which is already in [0, 1].
            ratios[ch_name] = 1.0 if val > 0 else 0.0
            scores[ch_name] = val
        elif theta_i > 0:
            # Normalise by P95 calibration threshold, then clip to [0, 1] so every
            # channel contributes equally to the mean regardless of scale.
            normalised = min(val / theta_i, 1.0)
            ratios[ch_name] = normalised
            scores[ch_name] = normalised
        else:
            ratios[ch_name] = 0.0
            scores[ch_name] = 0.0

    S = _aggregate_scores(list(scores.values()), score_mode) if scores else 0.0

    return {"channels": channels, "ratios": ratios, "thetas": thetas, "scores": scores, "S": S}


def compute_struggle_score_series(
    actions: np.ndarray,
    states: np.ndarray,
    thresholds: Thresholds | None = None,
    cfg: Config | None = None,
    score_mode: str = "mean",
) -> np.ndarray:
    """Compute a time-series of S values over an episode using the same
    normalization logic as :func:`compute_struggle_score`.

    Each element S[t] is the aggregated struggle score at window t, computed
    from the five-channel m-vector normalized by the calibrated P95 thresholds.
    NaN windows (metric warmup period) are preserved as NaN so callers can
    mask them.

    Parameters
    ----------
    actions, states : (T, 6) arrays for a full episode.
    thresholds      : Calibrated Thresholds; defaults to pooled.json.
    cfg             : Monitor Config; defaults to Config().
    score_mode      : Aggregation mode passed to :func:`_aggregate_scores`
                      ("max", "mean", "median", "mode"). Default "max".

    Returns
    -------
    np.ndarray of shape (T,) with one S value per window.
    """
    if thresholds is None:
        thresholds = _get_pooled_thresholds()
    if cfg is None:
        cfg = Config()

    ch_arrays = _per_channel_batch(actions, states, cfg, thresholds)

    T = len(next(iter(ch_arrays.values())))
    score_matrix = np.full((len(M_CHANNELS), T), np.nan, dtype=np.float64)

    for i, ch_name in enumerate(M_CHANNELS):
        arr = np.asarray(ch_arrays[ch_name], dtype=np.float64)
        theta_i = thresholds.theta[i]
        if ch_name == "sigma_bar":
            # Raw stall rate — already in [0, 1], theta=0 by design.
            score_matrix[i] = arr
        elif theta_i > 0:
            # Normalise by P95 threshold then clip to [0, 1], matching
            # the scalar compute_struggle_score() logic.
            score_matrix[i] = np.clip(arr / theta_i, 0.0, 1.0)
        # else: leave as NaN (theta_i == 0 for non-sigma_bar = calibration bug)

    # NaN in any channel at time t means the metric hadn't warmed up yet;
    # preserve those NaNs so the caller can skip warmup windows.
    # Suppress the "All-NaN slice" warning — expected for warmup frames where
    # every channel is still NaN; numpy raises it even inside errstate on some
    # builds, so we also filter at the warnings module level.
    if T == 0:
        return np.array([])
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if score_mode == "mean":
            return np.nanmean(score_matrix, axis=0)
        if score_mode == "median":
            return np.nanmedian(score_matrix, axis=0)
        return np.nanmax(score_matrix, axis=0)


# ── Configuration ─────────────────────────────────────────────────────────────

# Panel of judges: each judge uses the same prompt but a different temperature,
# creating a conservative→liberal spectrum. Low-temp judges only flag obvious
# failures; high-temp judges catch borderline cases too.
JUDGE_TEMPERATURES: tuple[float, ...] = (0.10, 0.25, 0.40, 0.55, 0.70)


# ── Prompts & schemas ─────────────────────────────────────────────────────────

# Legacy prompt — used by assess_frames (binary struggling judgment).
# ── Judge prompts ─────────────────────────────────────────────────────────────
# Switch between prompts by setting STRUGGLE_PROMPT = PROMPT_<name> below.

# v1 — original: visual + injected sensor metrics (jerk, Lyapunov)
PROMPT_V1 = """You are watching a sequence of frames from a live robot arm
doing a pick-and-place task. The frames are evenly sampled from the last few seconds.
Is the robot currently struggling?
Struggling signals:
  - Gripper missed the block or has an unstable / tilted grip
  - Block is rotating or slipping inside the gripper
  - Arm making repeated small corrections without making progress
  - Arm oscillating, stuck, or moving erratically
  - Block has been dropped or is about to fall
  - Jerk is elevated far above baseline — arm is erratic

Not struggling:
  - Arm moving smoothly and purposefully toward a clear goal
  - Block held firmly and stably
  - Clean, confident approach to pick or drop point

Return ONLY JSON with fields: struggling (bool), confidence (0-1), reason (one sentence).
"""

# v2 — visual only: pure movement smoothness, no task-specific cues
PROMPT_V2 = """You are watching a sequence of frames from a robot arm performing a task.
You will vote is the gripper is holding the block.

Vote struggling=true if:
  - The gripper is either picking up the block ot holding the block. 

Vote struggling=false if:
  - The gripper is not hodling any objects and making no attempts to pick up. 
  
Return ONLY JSON with fields: struggling (bool), confidence (0-1), reason (one sentence).
"""

STRUGGLE_PROMPT = PROMPT_V2   # ← swap here to compare prompts

# JSON schema for structured Gemini output — matches STRUGGLE_PROMPT.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "struggling": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason":     {"type": "string"},
    },
    "required": ["struggling", "confidence", "reason"],
}

# Sentinel values returned before the first Gemini call completes.
_DEFAULT_SIGNAL = {"struggling": False, "confidence": 0.0, "reason": "no assessment yet"}
_DEFAULT_INTERRUPT = {
    "interrupt_probability": 0.0,
    "vote_count": 0,
    "votes": [],        # per-judge: [{"temp": float, "struggling": bool, "confidence": float, "reason": str}, ...]
    "reason": "no assessment yet",
}


# ── Sensor signal helpers ─────────────────────────────────────────────────────

def _build_stability_block(actions: np.ndarray, states: np.ndarray) -> str:
    """Compute rolling jerk / Lyapunov metrics and format as a prompt text block.

    Args:
        actions: (T, D_a) float32 array of recent actions.
        states:  (T, D_s) float32 array of recent observation states.

    Returns:
        Formatted stability context string, or "" if insufficient data.
    """
    if len(actions) < 5:
        return ""

    delta      = np.diff(actions, axis=0)
    jerk       = np.linalg.norm(delta, axis=1)
    mean_jerk  = float(jerk.mean())
    cur_jerk   = float(jerk[-1])
    jerk_ratio = cur_jerk / (mean_jerk + 1e-9)
    jerk_tag   = f"HIGH ({jerk_ratio:.1f}x baseline)" if jerk_ratio > 2.0 else "normal"

    # Lyapunov proxy: V(t) = ||s_t - s_latest||^2, goal approximated as most recent state.
    s_ref      = states[-1]
    V          = np.sum((states - s_ref) ** 2, axis=1)
    dV         = np.diff(V)
    viol_rate  = float((dV > 0).mean())
    lyap_trend = "DIVERGING" if viol_rate > 0.5 else "CONVERGING"
    lyap_detail = f"V increased {viol_rate*100:.0f}% of recent steps"

    thresh = mean_jerk + 3.0 * float(jerk.std())
    n_disc = int((jerk > thresh).sum())

    return (
        f"\n--- STABILITY METRICS (last {len(actions)} frames) ---\n"
        f"  Jerk          : mean={mean_jerk:.2f}  current={cur_jerk:.2f}  [{jerk_tag}]\n"
        f"  Lyapunov trend: {lyap_trend}  ({lyap_detail})\n"
        f"  Discontinuities in window: {n_disc}\n"
        f"--- END METRICS ---\n"
    )


def _build_gripper_block(gripper: np.ndarray, threshold: float = 20.0) -> str:
    """Format the gripper angle signal as a supplementary text block for the prompt.

    Gemini is asked to count pickup/drop attempts from the video frames; this
    block provides the raw sensor trace so it can cross-check its visual read.

    Args:
        gripper:   1-D float32 array of gripper angles (degrees).
        threshold: Angle (degrees) above which the gripper is considered closed.

    Returns:
        Formatted gripper context string, or "" if fewer than 2 samples.
    """
    n = len(gripper)
    if n < 2:
        return ""

    closes = [i for i in range(1, n) if gripper[i-1] < threshold <= gripper[i]]
    opens  = [i for i in range(1, n) if gripper[i-1] >= threshold > gripper[i]]

    step    = max(1, n // 20)
    trace   = "  ".join(f"{v:.1f}" for v in gripper[::step])

    return (
        f"\n--- GRIPPER SIGNAL (last {n} frames, threshold={threshold}°) ---\n"
        f"  Closes (pickup events) detected: {len(closes)}\n"
        f"  Opens  (drop events)   detected: {len(opens)}\n"
        f"  Sampled trace (deg): {trace}\n"
        f"--- END GRIPPER ---\n"
    )


# ── Gemini assessment functions ───────────────────────────────────────────────

def _encode_frames(frames: list[np.ndarray], jpeg_quality: int) -> list:
    """Encode a list of BGR frames as inline JPEG Parts for a Gemini request."""
    parts = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            parts.append(types.Part.from_bytes(data=bytes(buf), mime_type="image/jpeg"))
    return parts


def assess_frames(
    frames: list[np.ndarray],
    client: genai.Client,
    model: str = "gemini-2.5-flash",
    jpeg_quality: int = 75,
    actions: np.ndarray | None = None,
    states:  np.ndarray | None = None,
) -> dict:
    """Binary struggle assessment (legacy).

    Sends frames and optional stability metrics to Gemini using STRUGGLE_PROMPT.

    Returns:
        {"struggling": bool, "confidence": float, "reason": str}
    """
    if not frames:
        return _DEFAULT_SIGNAL.copy()

    stability_block = ""
    if actions is not None and states is not None and len(actions) >= 5:
        stability_block = _build_stability_block(actions, states)

    parts = _encode_frames(frames, jpeg_quality)
    if not parts:
        return _DEFAULT_SIGNAL.copy()

    parts.append(types.Part.from_text(text=STRUGGLE_PROMPT.format(
        stability_block=stability_block,
    )))

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JUDGE_SCHEMA,
            temperature=0.2,
        ),
    )
    return json.loads(response.text)


def assess_interrupt(
    frames: list[np.ndarray],
    client: genai.Client,
    model: str = "gemini-2.5-flash",
    jpeg_quality: int = 75,
    temperature: float = 0.40,
    prompt: str | None = None,
) -> dict:
    """Single-judge binary struggle assessment.

    Sends frames to Gemini using STRUGGLE_PROMPT (or a custom prompt) at the
    given temperature.

    Args:
        temperature: Controls judge conservatism. Low (0.1) = fires only on
            clear failures. High (0.7) = fires on borderline cases too.
        prompt: Override the module-level STRUGGLE_PROMPT. Pass None to use
            the default.

    Returns:
        {"struggling": bool, "confidence": float, "reason": str}
    """
    if not frames:
        return _DEFAULT_SIGNAL.copy()

    parts = _encode_frames(frames, jpeg_quality)
    if not parts:
        return _DEFAULT_SIGNAL.copy()

    parts.append(types.Part.from_text(text=prompt if prompt is not None else STRUGGLE_PROMPT))

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JUDGE_SCHEMA,
            temperature=temperature,
        ),
    )
    return json.loads(response.text)


def assess_interrupt_panel(
    frames: list[np.ndarray],
    client: genai.Client,
    model: str = "gemini-2.5-flash",
    jpeg_quality: int = 75,
    temperatures: tuple[float, ...] = JUDGE_TEMPERATURES,
    prompt: str | None = None,
) -> dict:
    """Panel of N judges voting in parallel on whether to interrupt.

    Each judge uses the same prompt but a different temperature, creating a
    conservative→liberal spectrum. Low-temp judges only flag clear failures;
    high-temp judges catch borderline cases. interrupt_probability = vote_count
    / n_judges.

    Returns:
        {"interrupt_probability": float, "vote_count": int,
         "votes": [{"temp": float, "struggling": bool, "confidence": float, "reason": str}, ...],
         "reason": str}
    """
    if not frames:
        return _DEFAULT_INTERRUPT.copy()

    # Each judge gets its own genai.Client so concurrent threads never share
    # the underlying httpx connection pool — sharing a single client causes
    # race conditions where 2 of 3 judges silently fail.
    api_key = client._api_client.api_key

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(temperatures)) as pool:
        future_to_temp = {
            pool.submit(
                assess_interrupt, frames, genai.Client(api_key=api_key), model,
                jpeg_quality, temp, prompt,
            ): temp
            for temp in temperatures
        }
        ordered: list[tuple[float, dict]] = []
        for future in concurrent.futures.as_completed(future_to_temp):
            temp = future_to_temp[future]
            try:
                ordered.append((temp, future.result()))
            except Exception as e:
                print(f"[assess_interrupt_panel] judge temp={temp:.2f} failed: {e}")

    n_expected = len(temperatures)
    if not ordered:
        print(f"[assess_interrupt_panel] WARNING: all {n_expected} judges failed — returning default")
        return _DEFAULT_INTERRUPT.copy()
    if len(ordered) < n_expected:
        print(
            f"[assess_interrupt_panel] WARNING: only {len(ordered)}/{n_expected} judges responded"
        )

    ordered.sort(key=lambda x: x[0])   # stable order by temperature
    n = len(ordered)

    vote_count = sum(1 for _, v in ordered if v.get("struggling", False))
    interrupt_probability = vote_count / n

    # Per-judge vote details (sorted by temperature)
    votes = [
        {
            "temp": temp,
            "struggling": v.get("struggling", False),
            "confidence": v.get("confidence", 0.0),
            "reason": v.get("reason", ""),
        }
        for temp, v in ordered
    ]
    reason = votes[n // 2]["reason"]   # median judge's reasoning

    return {
        "interrupt_probability": round(interrupt_probability, 4),
        "vote_count":            vote_count,
        "votes":                 votes,
        "reason":                reason,
    }


# ── Probability post-processing ───────────────────────────────────────────────

class _EpisodeScorer:
    """Tracks the S score across a batch episode.

    struggle_score is set to the latest S value each check.
    peak_score tracks the highest S seen in the episode.
    """

    def __init__(self) -> None:
        self.struggle_score: float = 0.0
        self.peak_score:    float = 0.0

    def reset(self) -> None:
        self.struggle_score = 0.0
        self.peak_score     = 0.0

    def update(self, S: float) -> None:
        """Update with the latest S score."""
        self.struggle_score = S
        self.peak_score = max(self.peak_score, S)


# ── Episode state tracking ────────────────────────────────────────────────────

@dataclass
class EpisodeState:
    """Per-episode summary accumulated across all Gemini checks.

    Designed to track policy quality across training iterations — e.g. comparing
    clean-pickup rate or success-window distribution before and after teacher training.

    Fields
    ------
    target_reached   : block was successfully placed at the target.
    clean_pickup     : single pickup attempt (no retries); None if no pickup seen yet.
    pickup_attempts  : total pickup attempts (max seen across all checks this episode).
    clean_drop       : single drop attempt (no retries); None if no drop seen yet.
    drop_recorrected : had a failed drop (≥2 attempts) but still completed the task.
    max_probability  : peak interrupt_probability seen this episode.
    avg_probability  : mean interrupt_probability across all checks.
    success_window   : when task was first accomplished —
                       "first_15s" (0–15 s), "next_15s" (15–30 s), "last" (>30 s),
                       or None (task not completed).
    """
    target_reached:         bool        = False
    clean_pickup:           bool | None = None
    pickup_attempts:        int         = 0
    clean_drop:             bool | None = None
    drop_recorrected:       bool | None = None
    max_probability:        float       = 0.0
    avg_probability:        float       = 0.0
    success_window:         str | None  = None
    time_to_first_pickup_s: float | None = None  # seconds from episode start to first grasp
    time_to_complete_s:     float | None = None  # seconds from episode start to task accomplished

    def __str__(self) -> str:
        pickup_t = f"{self.time_to_first_pickup_s:.1f}s" if self.time_to_first_pickup_s is not None else "None"
        complete_t = f"{self.time_to_complete_s:.1f}s" if self.time_to_complete_s is not None else "None"
        return (
            f"target_reached={self.target_reached}"
            f"  pickup_attempts={self.pickup_attempts}  clean_pickup={self.clean_pickup}"
            f"  time_to_pickup={pickup_t}  time_to_complete={complete_t}"
            f"  clean_drop={self.clean_drop}  drop_recorrected={self.drop_recorrected}"
            f"  max_p={self.max_probability:.3f}  avg_p={self.avg_probability:.3f}"
            f"  success_window={self.success_window}"
        )


class EpisodeStateTracker:
    """Accumulates per-check Gemini results into an EpisodeState summary.

    Call ``update()`` with each Gemini result and the elapsed episode time.
    Call ``get_state()`` at any point (typically episode end) for the summary.
    """

    _WIN1: float = 15.0   # boundary between first and second window (seconds)
    _WIN2: float = 30.0   # boundary between second and last window (seconds)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._probs:            list[float]  = []
        self._max_picks:        int          = 0
        self._max_drops:        int          = 0
        self._first_success_t:  float | None = None
        self._drops_at_success: int          = 0   # drop count when task first accomplished
        self._first_pickup_t:   float | None = None

    def update(self, result: dict, ep_elapsed: float) -> None:
        """Ingest one Gemini check result. ep_elapsed = seconds since episode start."""
        p = float(result.get("interrupt_probability", 0.0))
        self._probs.append(p)

        picks = int(result.get("pickup_attempts", 0))
        drops = int(result.get("drop_attempts",   0))
        self._max_picks = max(self._max_picks, picks)
        self._max_drops = max(self._max_drops, drops)

        if picks > 0 and self._first_pickup_t is None:
            self._first_pickup_t = ep_elapsed

        if result.get("task_accomplished", False) and self._first_success_t is None:
            self._first_success_t  = ep_elapsed
            self._drops_at_success = drops

    def get_state(self) -> EpisodeState:
        """Return accumulated EpisodeState (safe to call at any time)."""
        n     = len(self._probs)
        avg_p = float(np.mean(self._probs)) if n else 0.0
        max_p = float(max(self._probs))     if n else 0.0

        # None = we haven't seen any pickup/drop yet (can't assess quality)
        clean_pickup: bool | None = (self._max_picks == 1) if self._max_picks > 0 else None
        clean_drop:   bool | None = (self._max_drops == 1) if self._max_drops > 0 else None

        # Re-correction: had ≥2 drop attempts but the task was still accomplished
        drop_recorrected: bool | None = None
        if self._first_success_t is not None and self._max_drops > 0:
            drop_recorrected = self._drops_at_success >= 2

        t = self._first_success_t
        if t is None:
            success_window: str | None = None
        elif t <= self._WIN1:
            success_window = "first_15s"
        elif t <= self._WIN2:
            success_window = "next_15s"
        else:
            success_window = "last"

        return EpisodeState(
            target_reached          = self._first_success_t is not None,
            clean_pickup            = clean_pickup,
            pickup_attempts         = self._max_picks,
            clean_drop              = clean_drop,
            drop_recorrected        = drop_recorrected,
            max_probability         = round(max_p, 3),
            avg_probability         = round(avg_p, 3),
            success_window          = success_window,
            time_to_first_pickup_s  = round(self._first_pickup_t, 1) if self._first_pickup_t is not None else None,
            time_to_complete_s      = round(self._first_success_t, 1) if self._first_success_t is not None else None,
        )


# ── Live monitor ──────────────────────────────────────────────────────────────

class LiveStruggleMonitor:
    """Continuously watches a 20-second rolling buffer and polls the judge panel.

    Every ``check_interval`` seconds the S score is computed from buffered
    actions/states. ``is_struggling()`` fires when S crosses
    ``interrupt_threshold``. Gemini is called when S >= threshold to provide
    a human-readable reason.

    Args:
        key_file:            Path to Gemini API key file (or None for env var).
        model:               Gemini model to use.
        check_interval:      How often (seconds) to call the panel.
        buffer_seconds:      Rolling buffer length — 20 s gives full episode context.
        n_sample_frames:     Frames subsampled per panel call.
        fps:                 Camera frame rate (used for buffer sizing).
        interrupt_threshold: S score above which is_struggling() → True.
        vote_dwell_s:        Seconds S must stay above threshold before a Gemini
                             vote is fired.  0 (default) fires immediately.
                             After a vote fires the timer resets, so the next
                             vote requires another full dwell period.
        gripper_col_idx:     Index into observation.state for gripper position.
        gripper_threshold:   Degrees above which the gripper is considered closed.
    """

    def __init__(
        self,
        key_file: str | None = None,
        model: str = "gemini-2.5-flash",
        check_interval: float = 2.0,
        buffer_seconds: float = 20.0,
        n_sample_frames: int = 12,
        fps: float = 30.0,
        interrupt_threshold: float = 0.5,
        vote_dwell_s: float = 0.0,
        score_mode: str = "mean",
        alpha_ramp_s: float = 10.0,
        warmup_s: float = 0.0,
        temperatures: tuple[float, ...] | list[float] | None = None,
        gripper_col_idx: int = 5,
        gripper_threshold: float = 20.0,
        thresholds: "Thresholds | None" = None,
        struggle_prompt: str | None = None,
    ):
        self._thresholds     = thresholds   # None → pooled.json inside compute_struggle_score
        self._prompt         = struggle_prompt  # None → module-level STRUGGLE_PROMPT
        self._client         = genai.Client(api_key=load_api_key(key_file))
        self._score_mode     = score_mode
        self._model          = model
        self._check_interval = check_interval
        self._n_sample       = int(n_sample_frames)
        self._threshold      = interrupt_threshold
        # vote_dwell_s: S must stay above threshold for this many seconds before
        # a Gemini vote fires.  0 fires immediately (legacy behaviour).
        self._vote_dwell_s   = max(0.0, vote_dwell_s)
        # Wall-clock time when S first exceeded the threshold in the current
        # dwell window.  None means S is currently below threshold.
        self._above_threshold_since: float | None = None
        # alpha_ramp_s > 0: linearly ramp alpha 0→1 over this many seconds so
        # the monitor cannot trigger in the first few seconds of an episode.
        # 0 disables the ramp (alpha = 1.0 always).
        self._alpha_ramp_s   = max(0.0, alpha_ramp_s)
        # warmup_s > 0: hard skip — do not compute S or call Gemini at all
        # for the first warmup_s seconds of each episode.
        self._warmup_s       = max(0.0, warmup_s)
        # Judge temperatures — controls panel size and conservatism spectrum.
        # None falls back to the module-level JUDGE_TEMPERATURES default.
        self._temperatures   = tuple(temperatures) if temperatures is not None else JUDGE_TEMPERATURES
        self._gripper_col    = gripper_col_idx
        self._gripper_thresh = gripper_threshold

        buffer_size             = int(buffer_seconds * fps)
        self._buffer: deque     = deque(maxlen=buffer_size)
        self._action_buf: deque = deque(maxlen=buffer_size)
        self._state_buf: deque  = deque(maxlen=buffer_size)
        self._lock              = threading.Lock()
        # Flag to prevent overlapping Gemini calls — set True when a panel
        # call is in-flight, cleared when it completes or errors.
        self._gemini_in_flight: bool  = False
        # Incremented each time a panel vote completes (result stored or error).
        # External threads can watch this to detect new votes without polling
        # _gemini_in_flight (which can flip too fast to catch reliably).
        self._interrupt_seq: int      = 0

        # Optional timeline recorder (AbEvalTimeline) — set via set_timeline().
        self._tl                  = None
        self._tl_warmup_span      = None   # open span for warmup lane
        self._tl_dwell_span       = None   # open span for dwell lane
        self._tl_vote_span        = None   # open span for vote_inflight lane
        self._tl_status_span      = None   # open span for monitor_active / monitor_suppress
        self._tl_metrics_started  = False  # True once metrics_start mark fired this episode
        # When True, S is still computed every check_interval but the Gemini
        # judge panel is skipped entirely.  Set during policy-B interventions
        # where a vote can't trigger anything and would just waste API quota.
        self._suppress_judges: bool   = False

        self._signal: dict          = _DEFAULT_SIGNAL.copy()
        self._interrupt: dict       = _DEFAULT_INTERRUPT.copy()
        self._struggle_score: float = 0.0
        self._channel_ratios: dict  = {}   # latest per-channel ratios from compute_struggle_score
        self._channel_values: dict  = {}   # latest raw (un-normalised) channel values
        self._channel_thetas: dict  = {}   # calibration thresholds (theta) per channel
        self._stop_event            = threading.Event()
        self._transfer_active       = threading.Event()
        self._transfer_epoch: int   = 0
        self._thread: threading.Thread | None         = None
        self._capture_thread: threading.Thread | None = None
        self._capture_fps: float = fps

        self._state_tracker: EpisodeStateTracker = EpisodeStateTracker()
        self._episode_start: float               = time.time()
        self._last_check_t: float                = 0.0   # wall time of last Gemini assessment

    # ── Public API ────────────────────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray) -> None:
        """Add a BGR camera frame to the rolling buffer. Call every control step."""
        with self._lock:
            self._buffer.append(frame.copy())

    def push_observation(self, action: np.ndarray, state: np.ndarray) -> None:
        """Feed the latest action and observation state for stability/gripper metrics.

        Call alongside push_frame() every control step.
        """
        with self._lock:
            self._action_buf.append(np.asarray(action, dtype=np.float32))
            self._state_buf.append(np.asarray(state, dtype=np.float32))

    def is_struggling(self) -> bool:
        """True when the Gemini panel vote fraction meets the interrupt threshold.

        S is used only as a gate to trigger a Gemini call; the actual interrupt
        decision is made by the panel of judges via assess_interrupt_panel().
        Before the first Gemini call the interrupt_probability is 0.0, so this
        returns False until Gemini has had a chance to assess the episode.
        """
        return self._interrupt.get("interrupt_probability", 0.0) >= self._threshold

    def is_vote_in_flight(self) -> bool:
        """True while a Gemini panel call is currently in-flight."""
        return self._gemini_in_flight

    @property
    def interrupt_seq(self) -> int:
        """Monotonically increasing counter incremented after each completed vote.

        Watchers can compare against a cached value to detect a new vote without
        polling ``_gemini_in_flight`` (which can flip too fast to catch reliably).
        """
        return self._interrupt_seq

    def get_signal(self) -> dict:
        """Return the legacy signal dict: {struggling, confidence, reason}."""
        return self._signal.copy()

    def get_interrupt(self) -> dict:
        """Return the latest interrupt assessment dict."""
        return self._interrupt.copy()

    def get_struggle_score(self) -> float:
        """Return the current S score."""
        return self._struggle_score

    def get_channel_ratios(self) -> dict:
        """Return the latest per-channel ratios (channel name -> ratio)."""
        return self._channel_ratios.copy()

    def get_channel_values(self) -> dict:
        """Return the latest raw (un-normalised) channel values."""
        return self._channel_values.copy()

    def get_channel_thetas(self) -> dict:
        """Return the calibration threshold (theta) per channel."""
        return self._channel_thetas.copy()

    @property
    def latest_frame(self):
        """Return a copy of the most recent camera frame, or None if buffer is empty."""
        with self._lock:
            return self._buffer[-1].copy() if self._buffer else None

    def get_episode_state(self) -> EpisodeState:
        """Return the accumulated episode state summary (safe to call at any time)."""
        return self._state_tracker.get_state()

    def set_timeline(self, tl) -> None:
        """Attach an AbEvalTimeline recorder. Pass None to detach."""
        self._tl = tl

    def set_thresholds(self, thresholds: "Thresholds | None") -> None:
        """Swap the calibration thresholds used for S normalization.

        Safe to call from any thread — Python's GIL makes the attribute
        assignment atomic.  Pass None to fall back to pooled.json.
        """
        self._thresholds = thresholds

    def note_first_action(self) -> None:
        """Reset the episode clock to now so warmup/metrics count from first robot motion.

        Call this when the robot sends its first action (via _MotionTimerRobot).
        Without this, warmup starts from reset_signal() which fires during go_home
        and save_wait, so metrics would start scoring before the robot has moved.
        """
        self._episode_start      = time.time()
        self._tl_metrics_started = False   # warmup just restarted
        # Reopen the warmup span at the correct time (first motion, not episode start).
        tl = self._tl
        if tl is not None:
            from ab_eval_timeline import SPAN_COLORS
            if self._tl_warmup_span is not None:
                tl.span_end(self._tl_warmup_span)
                self._tl_warmup_span = None
            if self._warmup_s > 0:
                self._tl_warmup_span = tl.span_start(
                    "monitor", "warmup", SPAN_COLORS["warmup"]
                )

    def suppress_judges(self, suppress: bool = True) -> None:
        """Enable or disable the Gemini judge panel.

        When suppressed, S is still computed and displayed every
        check_interval but no panel calls are made.  Use this during
        policy-B interventions where a vote can't trigger anything.
        """
        self._suppress_judges = suppress
        tl = self._tl
        if tl is None:
            return
        from ab_eval_timeline import SPAN_COLORS
        if suppress:
            tl.mark_monitor_suppress()
            # Close active span and open suppress span
            if self._tl_status_span is not None:
                tl.span_end(self._tl_status_span)
            self._tl_status_span = tl.span_start(
                "monitor", "monitor_suppress", SPAN_COLORS["monitor_suppress"]
            )
        else:
            tl.mark_monitor_resume()
            # Close suppress span; reopen active only if past warmup
            if self._tl_status_span is not None:
                tl.span_end(self._tl_status_span)
                self._tl_status_span = None
            if self._tl_warmup_span is None:   # warmup already expired
                self._tl_status_span = tl.span_start(
                    "monitor", "monitor_active", SPAN_COLORS["monitor_active"]
                )

    def reset_signal(self, keep_timer: bool = False) -> None:
        """Clear all signals, EMA score, frame buffer, and episode state.

        Call at the start of each episode. Clearing the frame buffer prevents
        stale frames from the previous episode being used in the first Gemini
        call of the new episode (which would cause false-positive interrupts).

        keep_timer: when True, preserve _episode_start so the GUI timer
            continues uninterrupted across a mid-episode policy switch.
        """
        with self._lock:
            self._buffer.clear()
            self._action_buf.clear()
            self._state_buf.clear()
        self._signal           = _DEFAULT_SIGNAL.copy()
        self._interrupt        = _DEFAULT_INTERRUPT.copy()
        self._struggle_score   = 0.0
        self._channel_ratios   = {}
        self._gemini_in_flight = False
        self._state_tracker.reset()
        if not keep_timer:
            self._episode_start = time.time()

        self._tl_metrics_started = False   # reset for new episode

        # Close any spans left open from the previous episode.
        # Warmup span is now opened in note_first_action() (when robot first
        # moves), not here — so metrics count from first motion, not from
        # episode start which includes go_home / save_wait dead time.
        tl = self._tl
        if tl is not None:
            for attr in ("_tl_warmup_span", "_tl_dwell_span",
                         "_tl_vote_span", "_tl_status_span"):
                old = getattr(self, attr)
                if old is not None:
                    tl.span_end(old)
                    setattr(self, attr, None)
            tl.mark_monitor_reset()

    def pause_for_transfer(self) -> None:
        """Suspend Gemini polling during a policy transfer and clear stale state.

        Increments the transfer epoch so any in-flight Gemini call that returns
        after this point is silently discarded rather than written to state.
        """
        self._transfer_epoch += 1
        self._transfer_active.set()
        self._interrupt      = _DEFAULT_INTERRUPT.copy()
        self._struggle_score = 0.0

    def resume_from_transfer(self) -> None:
        """Resume normal monitoring after a policy transfer completes."""
        self._transfer_active.clear()

    @property
    def transfer_active(self) -> bool:
        """True while a policy transfer is in progress."""
        return self._transfer_active.is_set()

    @property
    def buf_len(self) -> int:
        """Number of frames currently in the rolling buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def secs_since_last_check(self) -> float:
        """Seconds elapsed since the last completed Gemini assessment (0 if never)."""
        if self._last_check_t == 0.0:
            return 0.0
        return time.time() - self._last_check_t

    @property
    def episode_elapsed(self) -> float:
        """Seconds elapsed since the current episode started."""
        return time.time() - self._episode_start

    def start(self) -> None:
        """Start the background Gemini monitoring thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(
            f"[StruggleMonitor] started  model={self._model}"
            f"  interval={self._check_interval}s"
            f"  panel={len(self._temperatures)} judges {list(self._temperatures)}"
            f"  s_threshold={self._threshold}"
            f"  s_mode={self._score_mode}"
        )

    def stop(self) -> None:
        """Stop the monitoring thread (and capture thread if running)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
        print("[StruggleMonitor] stopped")

    def start_capture(self, camera_index: int = 1) -> None:
        """Open a dedicated capture thread that feeds frames into the buffer.

        Use when the camera is shared or the control loop is blocking.
        """
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_index,),
            daemon=True,
        )
        self._capture_thread.start()
        print(f"[StruggleMonitor] capture thread started  camera_index={camera_index}")

    def stop_capture(self) -> None:
        """Signal the capture thread to stop."""
        self._stop_event.set()

    # ── Background loops ──────────────────────────────────────────────────────

    def _capture_loop(self, camera_index: int) -> None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[StruggleMonitor] WARNING: could not open camera {camera_index}")
            return
        delay = 1.0 / self._capture_fps
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if ok:
                self.push_frame(frame)
            self._stop_event.wait(timeout=delay)
        cap.release()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()

            with self._lock:
                buf     = list(self._buffer)
                actions = (
                    np.stack(list(self._action_buf))
                    if len(self._action_buf) >= 5 else None
                )
                states = (
                    np.stack(list(self._state_buf))
                    if len(self._state_buf) >= 5 else None
                )

            if len(buf) >= self._n_sample and not self._transfer_active.is_set():
                if actions is not None and states is not None:
                    ep_elapsed = time.time() - self._episode_start

                    # Hard warmup skip: don't compute S or call Gemini at all.
                    if self._warmup_s > 0 and ep_elapsed < self._warmup_s:
                        print(
                            f"[StruggleMonitor] warmup   t={ep_elapsed:.0f}s"
                            f"  (skipping for {self._warmup_s - ep_elapsed:.0f}s more)"
                        )
                        elapsed = time.time() - start
                        self._stop_event.wait(timeout=max(0.0, self._check_interval - elapsed))
                        continue

                    # First iteration past warmup — close warmup span, mark
                    # metrics_start, and open monitor_active span.
                    if self._tl is not None and self._tl_warmup_span is not None:
                        self._tl.span_end(self._tl_warmup_span)
                        self._tl_warmup_span = None
                    if self._tl is not None and not self._tl_metrics_started:
                        self._tl_metrics_started = True
                        self._tl.mark_metrics_start()
                        # Open active span (green = judges ON) if not suppressed
                        if not self._suppress_judges and self._tl_status_span is None:
                            from ab_eval_timeline import SPAN_COLORS
                            self._tl_status_span = self._tl.span_start(
                                "monitor", "monitor_active", SPAN_COLORS["monitor_active"]
                            )

                    score_result = compute_struggle_score(actions, states, thresholds=self._thresholds, score_mode=self._score_mode)
                    S = score_result["S"]
                    self._struggle_score  = S
                    self._channel_ratios  = score_result["ratios"]
                    self._channel_values  = score_result["channels"]
                    self._channel_thetas  = score_result["thetas"]
                    alpha_t = (
                        min(ep_elapsed / self._alpha_ramp_s, 1.0)
                        if self._alpha_ramp_s > 0 else 1.0
                    )

                    if self._suppress_judges:
                        pass  # S updated above; judges suppressed (post-switch)
                    elif alpha_t * S < self._threshold:
                        print(
                            f"[StruggleMonitor] ok      t={ep_elapsed:.0f}s"
                            f"  alpha={alpha_t:.2f}  S={S:.2f}"
                            f"  scaled={alpha_t * S:.2f} < {self._threshold}  (skipping Gemini)"
                        )
                        self._above_threshold_since = None  # reset dwell when S falls below
                        # Close dwell span if robot was dwelling above threshold
                        if self._tl is not None and self._tl_dwell_span is not None:
                            self._tl.span_end(self._tl_dwell_span)
                            self._tl_dwell_span = None
                    elif not self._gemini_in_flight:
                        # S is above threshold — enforce dwell before firing a vote
                        if self._above_threshold_since is None:
                            self._above_threshold_since = time.time()
                            # Open dwell span on first threshold crossing
                            if self._tl is not None and self._tl_dwell_span is None:
                                from ab_eval_timeline import SPAN_COLORS
                                self._tl_dwell_span = self._tl.span_start(
                                    "vote", "dwell", SPAN_COLORS["dwell"]
                                )
                        dwell = time.time() - self._above_threshold_since
                        if dwell < self._vote_dwell_s:
                            remaining = self._vote_dwell_s - dwell
                            print(
                                f"[StruggleMonitor] dwell   t={ep_elapsed:.0f}s"
                                f"  alpha={alpha_t:.2f}  S={S:.2f}"
                                f"  scaled={alpha_t * S:.2f} >= {self._threshold}"
                                f"  dwell={dwell:.2f}s / {self._vote_dwell_s:.2f}s"
                                f"  (firing in {remaining:.2f}s)"
                            )
                            # Sleep exactly until dwell expires — not a full
                            # check_interval — so the vote fires immediately
                            # when the dwell period ends, not at the next
                            # check_interval boundary.
                            self._stop_event.wait(timeout=remaining)
                            continue
                        # Dwell satisfied — reset timer so next vote needs another full dwell
                        self._above_threshold_since = None
                        # Close dwell span (vote is about to fire)
                        if self._tl is not None and self._tl_dwell_span is not None:
                            self._tl.span_end(self._tl_dwell_span)
                            self._tl_dwell_span = None
                    if not self._gemini_in_flight and not self._suppress_judges and alpha_t * S >= self._threshold:
                        # Fire the panel asynchronously so _run never blocks on
                        # Gemini. S checks continue every check_interval regardless
                        # of how long the API takes to respond.
                        frames         = self._subsample(buf, self._n_sample)
                        epoch_snapshot = self._transfer_epoch
                        self._gemini_in_flight = True
                        # Open vote_inflight span
                        if self._tl is not None and self._tl_vote_span is None:
                            from ab_eval_timeline import SPAN_COLORS
                            self._tl_vote_span = self._tl.span_start(
                                "vote", "vote_inflight", SPAN_COLORS["vote_inflight"]
                            )

                        def _gemini_worker(frames=frames, epoch=epoch_snapshot,
                                           S=S, ep_t=ep_elapsed):
                            try:
                                result = assess_interrupt_panel(
                                    frames, self._client, self._model,
                                    temperatures=self._temperatures,
                                    prompt=self._prompt,
                                )
                                if not self._transfer_active.is_set() and self._transfer_epoch == epoch:
                                    prev_p = self._interrupt.get("interrupt_probability", 0.0)
                                    new_p  = result.get("interrupt_probability", 0.0)
                                    if new_p >= prev_p or new_p >= self._threshold:
                                        self._interrupt = result
                                    self._last_check_t = time.time()
                                    self._state_tracker.update(result, ep_t)
                                    status   = "INTERRUPT" if self.is_struggling() else "ok     "
                                    vote_str = " ".join(
                                        f"{'Y' if v['struggling'] else 'N'}@{v['temp']:.2f}"
                                        for v in result.get("votes", [])
                                    )
                                    print(
                                        f"[StruggleMonitor] {status}"
                                        f"  t={ep_t:.0f}s"
                                        f"  S={S:.2f}"
                                        f"  votes={result['vote_count']}/{len(self._temperatures)}"
                                        f"  [{vote_str}]"
                                        f"  gemini_p={result['interrupt_probability']:.2f}"
                                        f"  | {result['reason']}"
                                    )
                            except Exception as e:
                                print(f"[StruggleMonitor] Gemini error: {e}")
                            finally:
                                # Close vote_inflight span and mark result
                                if self._tl is not None:
                                    if self._tl_vote_span is not None:
                                        self._tl.span_end(self._tl_vote_span)
                                        self._tl_vote_span = None
                                    self._tl.mark_vote_result(self.is_struggling())
                                self._interrupt_seq += 1
                                self._gemini_in_flight = False

                        threading.Thread(target=_gemini_worker, daemon=True).start()

            elapsed = time.time() - start
            self._stop_event.wait(timeout=max(0.0, self._check_interval - elapsed))

    @staticmethod
    def _subsample(frames: list, n: int) -> list:
        """Evenly subsample n frames from a list."""
        if len(frames) <= n:
            return frames
        indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
        return [frames[i] for i in indices]


# ── Video rendering ───────────────────────────────────────────────────────────

def _draw_interrupt_hud(
    frame: np.ndarray,
    interrupt_prob: float,
    s_score: float,
    votes: list[dict],
    reason: str,
    threshold: float = 0.5,
    ep_elapsed: float = 0.0,
) -> np.ndarray:
    """Draw a probability gauge overlay at the top-left of a BGR frame."""
    frame = frame.copy()
    font  = cv2.FONT_HERSHEY_SIMPLEX

    pw, ph = 260, 130
    x0, y0 = 8, 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Colour: green → yellow → red
    r = int(min(255, interrupt_prob * 2 * 255))
    g = int(min(255, (1 - interrupt_prob) * 2 * 255))
    prob_color = (0, g, r)
    s_color    = (0, 200, 80) if s_score < threshold else (0, 60, 220)

    # S score bar
    bar_x0  = x0 + 4
    bar_y0  = y0 + ph - 14
    bar_len = int((pw - 8) * min(s_score, 2.0) / 2.0)
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + pw - 8, bar_y0 + 8), (50, 50, 50), -1)
    if bar_len > 0:
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_len, bar_y0 + 8), s_color, -1)
    thr_x = bar_x0 + int((pw - 8) * threshold / 2.0)
    cv2.line(frame, (thr_x, bar_y0 - 2), (thr_x, bar_y0 + 10), (255, 255, 255), 1)

    # Row 1: Gemini vote probability + S score
    cv2.putText(frame, f"p={interrupt_prob:.2f}",
                (x0 + 6, y0 + 20), font, 0.52, prob_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"S={s_score:.2f}",
                (x0 + 90, y0 + 20), font, 0.45, s_color, 1, cv2.LINE_AA)

    # Row 2: per-judge vote boxes (green = ok, red = struggling) + timer
    box_x = x0 + 6
    box_y = y0 + 30
    box_w, box_h = 30, 16
    gap = 4
    for v in votes:
        color = (0, 0, 220) if v["struggling"] else (0, 180, 0)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, -1)
        cv2.putText(frame, f"{v['temp']:.2f}",
                    (box_x + 2, box_y + box_h - 3), font, 0.30,
                    (255, 255, 255), 1, cv2.LINE_AA)
        box_x += box_w + gap

    mins, secs = divmod(int(ep_elapsed), 60)
    t_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"
    cv2.putText(frame, f"t={t_str}",
                (box_x + 4, box_y + box_h - 3), font, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)

    # Rows 3-5: reason text with word-wrap
    reason_short = (reason[:110] + "..") if len(reason) > 112 else reason
    words, line, lines = reason_short.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > 38:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    for i, ln in enumerate(lines[:3]):
        cv2.putText(frame, ln, (x0 + 6, y0 + 56 + i * 16),
                    font, 0.34, (210, 210, 210), 1, cv2.LINE_AA)

    return frame


def _render_episode_interrupt_video(
    ep_idx: int,
    checks: list[dict],
    actions: np.ndarray,
    states: np.ndarray,
    gripper: np.ndarray,
    vid_path: Path,
    from_ts: float,
    to_ts: float,
    out_path: Path,
    interrupt_threshold: float = 0.5,
    gripper_threshold: float = 20.0,
    strip_height: int = 130,
) -> None:
    """Render a 3-panel interrupt video (stability strip / camera / gripper strip).

    Caller passes pre-loaded episode arrays — no dataset re-loading required.
    """
    from stability_eval import _render_stability_strip
    from batch_detect_phases import _render_gripper_strip, detect_phases

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    delta       = np.diff(actions, axis=0)
    jerk        = np.linalg.norm(delta, axis=1)
    s_final     = states[-1]
    V           = np.sum((states - s_final) ** 2, axis=1)
    dV          = np.diff(V)
    mean_jerk   = jerk.mean()
    disc_frames = list(np.where(jerk > mean_jerk + 3 * jerk.std())[0].astype(int))

    pickup_frame, drop_frame = detect_phases(gripper, gripper_threshold)

    print(f"  rendering ep {ep_idx:4d} ...")
    cap = cv2.VideoCapture(str(vid_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, from_ts * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    top_strip = _render_stability_strip(jerk, dV, disc_frames, W, strip_height)
    bot_strip = _render_gripper_strip(gripper, pickup_frame, drop_frame,
                                      gripper_threshold, W, strip_height)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, strip_height + H + strip_height),
    )

    sorted_checks = sorted(checks, key=lambda c: c["frame_idx"])
    check_by_frame = {ck["frame_idx"]: ck for ck in sorted_checks}
    sorted_check_frames = sorted(check_by_frame.keys())

    # Mark peak-probability frame on the stability strip
    T_jerk    = len(jerk)
    T_gripper = len(gripper)
    if sorted_checks:
        peak_ck = max(sorted_checks, key=lambda c: c["p"])
        peak_cx = int(np.clip(peak_ck["frame_idx"] / max(T_jerk, 1) * W, 0, W - 1))
        cv2.line(top_strip, (peak_cx, 0), (peak_cx, strip_height), (0, 100, 255), 2)
        cv2.putText(top_strip, f"peak p={peak_ck['p']:.2f}",
                    (min(peak_cx + 3, W - 95), strip_height - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1, cv2.LINE_AA)

    frame_idx         = 0
    cur_check         = {"p": 0.0, "S": 0.0, "votes": [], "reason": ""}
    cur_check_key     = -1
    check_start_frame = 0
    type_out_frames   = int(fps * 1.5)
    font              = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 > to_ts + 0.1:
            break

        # Advance to most recent check at or before this frame
        new_key = -1
        for cf in sorted_check_frames:
            if cf <= frame_idx:
                new_key = cf
        if new_key != cur_check_key:
            cur_check         = check_by_frame[new_key] if new_key >= 0 else cur_check
            cur_check_key     = new_key
            check_start_frame = frame_idx

        # Typewriter effect: reveal reason progressively after each check
        elapsed     = frame_idx - check_start_frame
        chars       = max(1, int(elapsed / max(type_out_frames, 1) * len(cur_check["reason"]) + 0.5))
        live_reason = cur_check["reason"][:min(chars, len(cur_check["reason"]))]

        frame = _draw_interrupt_hud(
            frame,
            interrupt_prob=cur_check["p"],
            s_score=cur_check["S"],
            votes=cur_check.get("votes", []),
            reason=live_reason,
            threshold=interrupt_threshold,
            ep_elapsed=frame_idx / fps,
        )
        cv2.putText(frame, f"Episode {ep_idx}", (W - 130, 22),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        top = top_strip.copy()
        cx  = int(np.clip(frame_idx / max(T_jerk, 1) * W, 0, W - 1))
        cv2.line(top, (cx, 0), (cx, strip_height), (0, 255, 255), 2)
        if frame_idx < T_jerk:
            cv2.putText(top, f"jerk {jerk[frame_idx]:.1f}",
                        (min(cx + 4, W - 80), 16), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        bot = bot_strip.copy()
        cx2 = int(np.clip(frame_idx / max(T_gripper - 1, 1) * W, 0, W - 1))
        cv2.line(bot, (cx2, 0), (cx2, strip_height), (0, 255, 255), 2)
        if frame_idx < T_gripper:
            cv2.putText(bot, f"{gripper[frame_idx]:.1f}",
                        (min(cx2 + 4, W - 70), 16), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        writer.write(np.vstack([top, frame, bot]))
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"           -> {out_path}")


# ── Batch evaluation ──────────────────────────────────────────────────────────

def test_on_dataset(
    dataset_id: str = "nc8304/eval_smolvla-phase-split_combined",
    key_file: str | None = None,
    model: str = "gemini-2.5-flash",
    check_interval_s: float = 2.0,
    window_seconds: float = 20.0,
    n_sample_frames: int = 12,
    fps: float = 30.0,
    interrupt_threshold: float = 0.5,
    gripper_col_idx: int = 5,
    gripper_threshold: float = 20.0,
    results_csv: str | None = None,
    out_csv: str = "struggle_test_results.csv",
    video_dir: str | None = None,
    max_episodes: int | None = None,
    score_mode: str = "max",
) -> pd.DataFrame:
    """Run the interrupt monitor over every episode in a LeRobot dataset.

    Simulates the live pipeline in batch: steps through each episode's frames,
    calls assess_interrupt at ``check_interval_s`` intervals, computes the S
    score from buffered actions/states, and compares interrupt decisions to
    ground-truth labels.

    Optionally renders 3-panel annotated videos per episode to ``video_dir``.

    Returns:
        DataFrame with per-episode results (interrupt decision, EMA scores,
        attempt counts, labels).
    """
    print(f"Downloading dataset {dataset_id} ...")
    # Use local cache if available to avoid slow HF hub freshness checks.
    try:
        repo_dir = snapshot_download(repo_id=dataset_id, repo_type="dataset", local_files_only=True)
        print(f"  Using local cache: {repo_dir}")
    except Exception:
        repo_dir = snapshot_download(repo_id=dataset_id, repo_type="dataset")
    data_files = sorted(glob.glob(f"{repo_dir}/data/**/*.parquet", recursive=True))
    meta_files = sorted(glob.glob(f"{repo_dir}/meta/episodes/**/*.parquet", recursive=True))
    df_data    = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    df_meta    = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

    # Load ground-truth labels
    if results_csv is None:
        script_dir  = Path(__file__).parent
        candidates  = [script_dir / "results.csv", Path("results.csv")]
        results_csv = next((str(p) for p in candidates if p.exists()), None)
    labels: dict[int, dict] = {}
    if results_csv and Path(results_csv).exists():
        for _, row in pd.read_csv(results_csv).iterrows():
            labels[int(row["episode_index"])] = {
                "success": bool(row.get("overall_success", False)),
                "pick":    str(row.get("pick_quality", "")),
                "drop":    str(row.get("drop_quality", "")),
            }
        print(f"Loaded {len(labels)} labels from {results_csv}")

    client           = genai.Client(api_key=load_api_key(key_file))
    buffer_size      = int(window_seconds * fps)
    frames_per_check = int(check_interval_s * fps)

    # Auto-detect camera key from meta columns
    _cam_candidates = [
        "videos/observation.images.camera1",
        "videos/observation.images.front",
        "videos/observation.images.top",
        "videos/observation.images.camera_0",
    ]
    cam_key = next(
        (k for k in _cam_candidates if f"{k}/from_timestamp" in df_meta.columns),
        "videos/observation.images.camera1",
    )
    print(f"  Using camera key: {cam_key}")
    chunk_col = f"{cam_key}/chunk_index"
    file_col  = f"{cam_key}/file_index"
    from_col  = f"{cam_key}/from_timestamp"
    to_col    = f"{cam_key}/to_timestamp"

    episodes = sorted(df_data["episode_index"].unique())
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    # Resume: skip episodes already written to the CSV.
    already_done: set[int] = set()
    if Path(out_csv).exists():
        import csv as _csv
        with open(out_csv, newline="") as _f:
            for _row in _csv.DictReader(_f):
                try:
                    already_done.add(int(_row["episode_index"]))
                except (KeyError, ValueError):
                    pass
        if already_done:
            print(f"  Resuming: {len(already_done)} episodes already done, skipping them.")

    rows: list[dict]                    = []
    episode_checks: dict[int, list]     = {}
    video_paths: list[Path]             = []

    for ep_idx in episodes:
        if ep_idx in already_done:
            continue
        ep_df   = df_data[df_data["episode_index"] == ep_idx].reset_index(drop=True)
        ep_meta = df_meta[df_meta["episode_index"] == ep_idx]

        if ep_meta.empty or from_col not in ep_meta.columns:
            print(f"  [ep {ep_idx}] no meta — skipping")
            continue
        row_m   = ep_meta.iloc[0]
        from_ts = float(row_m[from_col])
        to_ts   = float(row_m[to_col]) if to_col in ep_meta.columns else None

        cam_subdir = cam_key.removeprefix("videos/")
        vid_path = (
            Path(repo_dir) / "videos" / cam_subdir
            / f"chunk-{int(row_m[chunk_col]):03d}"
            / f"file-{int(row_m[file_col]):03d}.mp4"
        )
        if not vid_path.exists():
            print(f"  [ep {ep_idx}] video not found — skipping")
            continue

        cap        = cv2.VideoCapture(str(vid_path))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.set(cv2.CAP_PROP_POS_MSEC, from_ts * 1000)

        frame_buf:  deque = deque(maxlen=buffer_size)
        action_buf: deque = deque(maxlen=buffer_size)
        state_buf:  deque = deque(maxlen=buffer_size)

        scorer              = _EpisodeScorer()
        ep_tracker          = EpisodeStateTracker()
        n_checks            = 0
        first_interrupt_ts: float | None = None
        last_result         = _DEFAULT_INTERRUPT.copy()
        frame_counter       = 0
        next_check_at       = int(10.0 * actual_fps)   # first check after 10 s
        all_s: list[float] = []
        episode_checks[ep_idx] = []

        print(f"\n[ep {ep_idx:02d}] {len(ep_df)} frames  from={from_ts:.1f}s")

        for data_idx in range(len(ep_df)):
            ok, frame = cap.read()
            if not ok:
                break
            if to_ts is not None and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 > to_ts + 0.1:
                break

            frame_buf.append(frame)
            action = np.asarray(ep_df.at[data_idx, "action"],            dtype=np.float32)
            state  = np.asarray(ep_df.at[data_idx, "observation.state"], dtype=np.float32)
            action_buf.append(action)
            state_buf.append(state)
            frame_counter += 1

            if frame_counter >= next_check_at and len(frame_buf) >= n_sample_frames:
                next_check_at += frames_per_check
                frames_snap  = LiveStruggleMonitor._subsample(list(frame_buf), n_sample_frames)
                actions_snap = np.stack(list(action_buf)) if len(action_buf) >= 5 else None
                states_snap  = np.stack(list(state_buf))  if len(state_buf)  >= 5 else None
                cur_ts     = from_ts + frame_counter / actual_fps
                ep_elapsed = frame_counter / actual_fps

                # Compute S score from buffered sensor data
                S = 0.0
                if actions_snap is not None and states_snap is not None:
                    S = compute_struggle_score(actions_snap, states_snap, score_mode=score_mode)["S"]
                scorer.update(S)
                all_s.append(S)

                flagged = scorer.struggle_score >= interrupt_threshold
                if flagged and first_interrupt_ts is None:
                    first_interrupt_ts = cur_ts

                try:
                    result = assess_interrupt_panel(
                        frames_snap, client, model,
                        actions=actions_snap, states=states_snap,
                    )
                    last_result = result
                    n_checks   += 1
                    ep_tracker.update(result, ep_elapsed)

                    p_raw = float(result["interrupt_probability"])

                    episode_checks[ep_idx].append({
                        "frame_idx":  frame_counter,
                        "ts":         cur_ts,
                        "ep_elapsed": ep_elapsed,
                        "p":          p_raw,
                        "S":          S,
                        "votes":      result.get("votes", []),
                        "reason":     result["reason"],
                    })

                    vote_str = " ".join(
                        f"{'Y' if v['struggling'] else 'N'}@{v['temp']:.2f}"
                        for v in result.get("votes", [])
                    )
                    status = "INTERRUPT" if flagged else "ok     "
                    print(
                        f"  t={cur_ts:.1f}s  {status}"
                        f"  S={S:.2f}"
                        f"  votes={result['vote_count']}/{len(JUDGE_TEMPERATURES)}"
                        f"  [{vote_str}]"
                        f"  gemini_p={p_raw:.2f}"
                        f"  | {result['reason']}"
                    )
                except Exception as e:
                    print(f"  t={cur_ts:.1f}s  Gemini error: {e}")

        cap.release()

        lab             = labels.get(ep_idx, {})
        final_interrupt = scorer.struggle_score >= interrupt_threshold
        mean_s          = float(np.mean(all_s)) if all_s else 0.0
        ep_state        = ep_tracker.get_state()

        row = {
            "episode_index":         ep_idx,
            "n_checks":              n_checks,
            "mean_s_score":          round(mean_s, 3),
            "peak_s_score":          round(scorer.peak_score, 3),
            "final_s_score":         round(scorer.struggle_score, 3),
            "final_interrupt":       final_interrupt,
            "first_interrupt_ts":    first_interrupt_ts,
            "final_vote_count":      last_result.get("vote_count", 0),
            "final_reason":          last_result["reason"],
            # Episode state (for cross-iteration policy comparison)
            "target_reached":        ep_state.target_reached,
            "clean_pickup":          ep_state.clean_pickup,
            "pickup_attempts":       ep_state.pickup_attempts,
            "clean_drop":            ep_state.clean_drop,
            "drop_recorrected":      ep_state.drop_recorrected,
            "max_probability":       ep_state.max_probability,
            "avg_probability":       ep_state.avg_probability,
            "success_window":        ep_state.success_window,
            "time_to_first_pickup_s": ep_state.time_to_first_pickup_s,
            "time_to_complete_s":    ep_state.time_to_complete_s,
            # Ground-truth labels
            "label_success":         lab.get("success", None),
            "label_pick":            lab.get("pick",    None),
            "label_drop":            lab.get("drop",    None),
        }
        rows.append(row)
        # Write this episode's row immediately so progress survives a crash
        import csv as _csv
        _write_header = not Path(out_csv).exists()
        with open(out_csv, "a", newline="") as _f:
            _w = _csv.DictWriter(_f, fieldnames=list(row.keys()))
            if _write_header:
                _w.writeheader()
            _w.writerow(row)
        print(
            f"  -> {'INTERRUPT' if final_interrupt else 'ok'}"
            f"  S={scorer.struggle_score:.2f}  peak={scorer.peak_score:.2f}"
            f"  mean_S={mean_s:.2f}"
            f"  pickup_t={ep_state.time_to_first_pickup_s}s  complete_t={ep_state.time_to_complete_s}s"
            + (f"  label_success={lab.get('success')}" if lab else "")
        )

        if video_dir is not None and episode_checks.get(ep_idx):
            actions_full = np.stack(ep_df["action"].values).astype(np.float32)
            states_full  = np.stack(ep_df["observation.state"].values).astype(np.float32)
            gripper_full = states_full[:, gripper_col_idx]
            out_vid      = Path(video_dir) / f"episode_{ep_idx:04d}_interrupt.mp4"
            _render_episode_interrupt_video(
                ep_idx=ep_idx,
                checks=episode_checks[ep_idx],
                actions=actions_full,
                states=states_full,
                gripper=gripper_full,
                vid_path=vid_path,
                from_ts=from_ts,
                to_ts=to_ts if to_ts is not None else from_ts + len(ep_df) / fps,
                out_path=out_vid,
                interrupt_threshold=interrupt_threshold,
                gripper_threshold=gripper_threshold,
            )
            video_paths.append(out_vid)

    results_df = pd.DataFrame(rows)
    print(f"\nSaved results to {out_csv}  ({len(rows)} episodes)")

    if labels and "label_success" in results_df.columns:
        labeled = results_df.dropna(subset=["label_success"])
        failed  = labeled[labeled["label_success"] == False]
        success = labeled[labeled["label_success"] == True]

        sensitivity  = (failed["final_interrupt"]  == True).mean()  if len(failed)  else float("nan")
        specificity  = (success["final_interrupt"] == False).mean() if len(success) else float("nan")
        mean_p_fail  = failed["mean_s_score"].mean()                 if len(failed)  else float("nan")
        mean_p_succ  = success["mean_s_score"].mean()                if len(success) else float("nan")

        print(f"\n=== Summary ===")
        print(f"  Episodes evaluated  : {len(results_df)}")
        print(f"  Labeled episodes    : {len(labeled)}  (failed={len(failed)}, success={len(success)})")
        print(f"  Sensitivity         : {sensitivity:.2%}")
        print(f"  Specificity         : {specificity:.2%}")
        print(f"  Mean S (failed)     : {mean_p_fail:.3f}")
        print(f"  Mean S (success)    : {mean_p_succ:.3f}")
        print(f"  Separation          : {mean_p_fail - mean_p_succ:.3f}  (higher = better)")

    if video_paths:
        from batch_detect_phases import concatenate_videos
        concat_out = Path(video_dir) / "all_interrupt.mp4"
        print(f"\nConcatenating {len(video_paths)} videos -> {concat_out}")
        concatenate_videos(video_paths, concat_out)
        print(f"Done: {concat_out}")

    return results_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Struggle monitor tools.")
    sub    = parser.add_subparsers(dest="cmd")

    p_live = sub.add_parser("live", help="Simulate live monitoring on a single .mp4.")
    p_live.add_argument("video",       help="Path to an episode .mp4.")
    p_live.add_argument("--key-file",  default=r"C:/Users/calle/Desktop/gem.txt")
    p_live.add_argument("--model",     default="gemini-2.5-flash")
    p_live.add_argument("--interval",  type=float, default=2.0)
    p_live.add_argument("--threshold", type=float, default=0.5,
                        help="S score threshold above which is_struggling() fires")
    p_live.add_argument("--score-mode", default="mean", choices=list(_SCORE_MODES),
                        help="Aggregation for S: mean (default), max, median, mode")
    p_live.add_argument("--alpha-ramp", type=float, default=10.0,
                        help="Seconds to ramp alpha 0→1; 0 disables ramp (alpha=1 always)")

    p_batch = sub.add_parser("batch", help="Run batch test on a full LeRobot dataset.")
    p_batch.add_argument("--dataset",      default="nc8304/eval_smolvla-phase-split_combined")
    p_batch.add_argument("--key-file",     default=r"C:/Users/calle/Desktop/gem.txt")
    p_batch.add_argument("--model",        default="gemini-2.5-flash")
    p_batch.add_argument("--interval",     type=float, default=2.0)
    p_batch.add_argument("--window",       type=float, default=20.0,  help="Rolling buffer seconds")
    p_batch.add_argument("--threshold",    type=float, default=0.5,   help="S score interrupt threshold")
    p_batch.add_argument("--results-csv",  default=None)
    p_batch.add_argument("--out-csv",      default="struggle_test_results.csv")
    p_batch.add_argument("--video-dir",    default=None,
                         help="If set, render 3-panel interrupt videos here.")
    p_batch.add_argument("--max-episodes", type=int, default=None,
                         help="Stop after this many episodes.")
    p_batch.add_argument("--score-mode", default="mean", choices=list(_SCORE_MODES),
                         help="Aggregation for S: mean (default), max, median, mode")
    p_batch.add_argument("--alpha-ramp", type=float, default=10.0,
                         help="Seconds to ramp alpha 0→1; 0 disables ramp (alpha=1 always)")

    args = parser.parse_args()

    if args.cmd == "live" or args.cmd is None:
        video_path = getattr(args, "video", None)
        if video_path is None:
            parser.print_help()
            raise SystemExit(1)

        monitor = LiveStruggleMonitor(
            key_file=args.key_file,
            model=args.model,
            check_interval=args.interval,
            interrupt_threshold=args.threshold,
            score_mode=args.score_mode,
            alpha_ramp_s=args.alpha_ramp,
        )
        monitor.start()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"Simulating live feed from {video_path}  ({fps:.0f} fps)\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            monitor.push_frame(frame)
            time.sleep(1.0 / fps)

        cap.release()
        monitor.stop()

    elif args.cmd == "batch":
        test_on_dataset(
            dataset_id=args.dataset,
            key_file=args.key_file,
            model=args.model,
            check_interval_s=args.interval,
            window_seconds=args.window,
            interrupt_threshold=args.threshold,
            results_csv=args.results_csv,
            out_csv=args.out_csv,
            video_dir=args.video_dir,
            max_episodes=args.max_episodes,
        )
