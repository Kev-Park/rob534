"""
Trajectory stability evaluation pipeline for LeRobot robotics datasets.

For each episode the following metrics are computed
────────────────────────────────────────────────────
Action jerk
  delta_t  = a_{t+1} - a_t                  (consecutive action difference)
  jerk_t   = ||delta_t||₂                   (L2 magnitude)
  → mean_jerk, max_jerk, std_jerk

Discontinuities
  A frame is flagged when jerk_t > mean + 3σ.
  → discontinuity_count, discontinuity_frames

Action variance
  Mean per-dimension variance of actions across the episode.
  → action_variance

Lyapunov-inspired convergence  (applied to observation state)
  Candidate:  V(t) = ||s_t - s_final||²     (distance to terminal state)
  Positive definite, V(s_final) = 0.
  Stable iff dV/dt ≤ 0 along the trajectory.

  violation_rate : fraction of steps where V(t+1) > V(t)
  decay_rate λ   : estimated from log(V+ε) ~ −λt regression;
                   λ > 0 ↔ exponential convergence

Compounding drift
  Slope of jerk_t over time via linear regression.
  Positive slope → later frames are jerkier (instability compounds).

Composite stability score  [0, 1],  higher = more stable
  score = 0.30·smoothness + 0.25·continuity + 0.25·lyapunov + 0.20·drift

Usage
─────
    python stability_eval.py
    python stability_eval.py --dataset nc8304/eval_smolvla-phase-split_combined
    python stability_eval.py --threshold 0.5 --output my_report.csv
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from analyze_rollout import analyze_rollout, extract_episode_clip

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_ID          = "nc8304/eval_smolvla-phase-split_combined"
DISCONTINUITY_SIGMA = 3.0          # flag jerk > mean + N·σ

WEIGHTS = {
    "smoothness": 0.30,
    "continuity": 0.25,
    "lyapunov":   0.25,
    "drift":      0.20,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_array(series: pd.Series) -> np.ndarray:
    """Series of lists/arrays → 2-D float32 array (T, D)."""
    return np.stack([np.asarray(v, dtype=np.float32) for v in series])


def _episode_metrics(ep_df: pd.DataFrame) -> dict:
    ep_df  = ep_df.sort_values("frame_index").reset_index(drop=True)
    T      = len(ep_df)
    ep_idx = int(ep_df["episode_index"].iloc[0])

    base = {"episode_index": ep_idx, "n_frames": T}

    if T < 3:
        return {
            **base,
            "mean_jerk": np.nan, "max_jerk": np.nan, "std_jerk": np.nan,
            "action_variance": np.nan, "discontinuity_count": 0,
            "discontinuity_frames": [],
            "lyapunov_violation_rate": np.nan, "lyapunov_decay_rate": np.nan,
            "compounding_drift_slope": np.nan,
            "smoothness_score": np.nan, "continuity_score": np.nan,
            "lyapunov_score": np.nan, "drift_score": np.nan,
            "stability_score": np.nan,
        }

    actions = _to_array(ep_df["action"])            # (T, 6)
    states  = _to_array(ep_df["observation.state"]) # (T, 6)

    # ── Action jerk ───────────────────────────────────────────────────────────
    delta = np.diff(actions, axis=0)          # (T-1, 6)
    jerk  = np.linalg.norm(delta, axis=1)    # (T-1,)

    mean_jerk = float(jerk.mean())
    max_jerk  = float(jerk.max())
    std_jerk  = float(jerk.std())

    # ── Action variance ───────────────────────────────────────────────────────
    action_var = float(np.var(actions, axis=0).mean())

    # ── Discontinuities ───────────────────────────────────────────────────────
    threshold   = mean_jerk + DISCONTINUITY_SIGMA * std_jerk
    disc_mask   = jerk > threshold
    disc_count  = int(disc_mask.sum())
    disc_frames = list(map(int, np.where(disc_mask)[0]))

    # ── Lyapunov convergence ─────────────────────────────────────────────────
    #   V(t) = ||s_t − s_final||²   (quadratic candidate; goal ≈ terminal state)
    s_final = states[-1]
    V       = np.sum((states - s_final) ** 2, axis=1)   # (T,)
    dV      = np.diff(V)                                  # (T-1,)

    # Violation rate: fraction of steps where V increases
    lyapunov_violation_rate = float((dV > 0).mean())

    # Exponential decay rate λ from  log(V+ε) = −λt + c
    t_arr    = np.arange(T, dtype=np.float32)
    log_V    = np.log(V + 1e-8)
    coeffs   = np.polyfit(t_arr, log_V, 1)   # coeffs[0] = slope = −λ
    lyapunov_decay_rate = float(-coeffs[0])  # positive ↔ converging

    # ── Compounding drift ─────────────────────────────────────────────────────
    #   Slope of jerk over time; positive = instability compounds
    t_jerk      = np.arange(len(jerk), dtype=np.float32)
    drift_slope = float(np.polyfit(t_jerk, jerk, 1)[0])

    # ── Component scores  [0, 1] ──────────────────────────────────────────────

    # Smoothness: low coefficient-of-variation of jerk → smooth
    cv = std_jerk / (mean_jerk + 1e-9)
    smoothness_score = float(np.clip(np.exp(-cv), 0.0, 1.0))

    # Continuity: fraction of non-discontinuous steps
    continuity_score = float(1.0 - disc_count / max(len(jerk), 1))

    # Lyapunov: rewards both low violation rate and positive decay rate
    decay_norm    = float(np.clip(lyapunov_decay_rate / 0.1, 0.0, 1.0))
    lyapunov_score = float(
        0.6 * (1.0 - lyapunov_violation_rate) +
        0.4 * decay_norm
    )

    # Drift: negative slope → improving (score → 1); positive → compounding (→ 0)
    drift_score = float(
        np.clip(0.5 - drift_slope / (2.0 * mean_jerk + 1e-9), 0.0, 1.0)
    )

    # ── Composite stability score ─────────────────────────────────────────────
    stability_score = (
        WEIGHTS["smoothness"] * smoothness_score +
        WEIGHTS["continuity"] * continuity_score +
        WEIGHTS["lyapunov"]   * lyapunov_score   +
        WEIGHTS["drift"]      * drift_score
    )

    return {
        **base,
        "mean_jerk":               round(mean_jerk,               4),
        "max_jerk":                round(max_jerk,                4),
        "std_jerk":                round(std_jerk,                4),
        "action_variance":         round(action_var,              4),
        "discontinuity_count":     disc_count,
        "discontinuity_frames":    disc_frames,
        "lyapunov_violation_rate": round(lyapunov_violation_rate, 4),
        "lyapunov_decay_rate":     round(lyapunov_decay_rate,     6),
        "compounding_drift_slope": round(drift_slope,             6),
        "smoothness_score":        round(smoothness_score,        4),
        "continuity_score":        round(continuity_score,        4),
        "lyapunov_score":          round(lyapunov_score,          4),
        "drift_score":             round(drift_score,             4),
        "stability_score":         round(stability_score,         4),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_trajectory_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-episode stability metrics from a rollout DataFrame.

    Args:
        df: output of analyze_rollout(); must contain columns
            [action, observation.state, frame_index, episode_index].

    Returns:
        Per-episode stability DataFrame sorted cleanest → worst,
        with a 'rank' column prepended.
    """
    rows   = [_episode_metrics(ep_df) for _, ep_df in df.groupby("episode_index")]
    result = (
        pd.DataFrame(rows)
        .sort_values("stability_score", ascending=False)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def print_stability_report(
    stability_df: pd.DataFrame,
    unstable_threshold: float = 0.5,
) -> None:
    """Print a human-readable summary report to stdout."""
    n_total    = len(stability_df)
    n_unstable = int((stability_df["stability_score"] < unstable_threshold).sum())
    mean_score = stability_df["stability_score"].dropna().mean()

    W = 74
    print("=" * W)
    print("  TRAJECTORY STABILITY REPORT")
    print("=" * W)
    print(f"  Episodes evaluated  : {n_total}")
    print(f"  Unstable (< {unstable_threshold:.2f})   : {n_unstable}")
    print(f"  Mean stability      : {mean_score:.3f}")
    print(f"  Score weights       : {WEIGHTS}")
    print()

    display_cols = [
        "rank", "episode_index", "stability_score",
        "mean_jerk", "max_jerk", "discontinuity_count",
        "lyapunov_violation_rate", "lyapunov_decay_rate",
        "compounding_drift_slope",
    ]
    print(stability_df[display_cols].to_string(index=False))

    unstable = stability_df[stability_df["stability_score"] < unstable_threshold]
    if not unstable.empty:
        print()
        print("  ── UNSTABLE SEQUENCES " + "─" * (W - 22))
        for _, row in unstable.iterrows():
            frames = row["discontinuity_frames"]
            snippet = frames[:10]
            suffix  = "..." if len(frames) > 10 else ""
            print(
                f"  ep {int(row['episode_index']):4d}  "
                f"score={row['stability_score']:.3f}  "
                f"disc={int(row['discontinuity_count'])}  "
                f"lyap_viol={row['lyapunov_violation_rate']:.2f}  "
                f"λ={row['lyapunov_decay_rate']:.4f}  "
                f"drift={row['compounding_drift_slope']:.4f}"
            )
            if frames:
                print(f"         discontinuity frames: {snippet}{suffix}")

    print("=" * W)


# ── Video overlay ─────────────────────────────────────────────────────────────

def _render_stability_strip(
    jerk: np.ndarray,       # (T-1,)  action jerk magnitudes
    dV:   np.ndarray,       # (T-1,)  Lyapunov differences
    disc_frames: list,
    width: int,
    height: int,
) -> np.ndarray:
    """Pre-render the full-episode stability plot as a BGR image (height, width, 3).

    Colour encodes Lyapunov dV per segment:
      green  → V decreasing (converging)
      red    → V increasing (diverging)
    Orange dashed verticals mark discontinuity frames.
    """
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#111111")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#111111")

    T = len(jerk)
    x = np.arange(T)

    # Colour each line segment by dV sign
    for i in range(T - 1):
        color = "#e74c3c" if dV[i] > 0 else "#2ecc71"
        ax.plot([x[i], x[i + 1]], [jerk[i], jerk[i + 1]],
                color=color, linewidth=0.8, alpha=0.9, solid_capstyle="round")

    # Discontinuity markers
    for df_idx in disc_frames:
        if df_idx < T:
            ax.axvline(x=df_idx, color="#f39c12", alpha=0.7,
                       linewidth=1.0, linestyle="--")

    ax.set_xlim(0, T)
    ax.set_ylim(0, float(jerk.max()) * 1.2 + 1e-9)
    ax.set_ylabel("jerk", color="white", fontsize=7, labelpad=2)
    ax.tick_params(colors="white", labelsize=6, length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    legend_elements = [
        Line2D([0], [0], color="#2ecc71", linewidth=1.2, label="V↓ converging"),
        Line2D([0], [0], color="#e74c3c", linewidth=1.2, label="V↑ diverging"),
        Line2D([0], [0], color="#f39c12", linewidth=1.0,
               linestyle="--", label="discontinuity"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=6,
              facecolor="#111111", edgecolor="#444444", labelcolor="white",
              framealpha=0.8)

    fig.tight_layout(pad=0.4)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba   = np.asarray(canvas.buffer_rgba())          # (H, W, 4) uint8
    strip  = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    strip  = cv2.resize(strip, (width, height), interpolation=cv2.INTER_LINEAR)
    return strip


def overlay_stability(
    video_path: Path,
    ep_df: pd.DataFrame,
    stability_row: pd.Series,
    out_path: Path,
    strip_height: int = 130,
) -> Path:
    """Burn a live stability strip onto the bottom of every frame.

    The strip shows the full-episode jerk plot (coloured by Lyapunov dV),
    a yellow cursor tracking the current frame, and a text score label.
    Same semi-transparent compositing style as overlay_result.

    Args:
        video_path:      Source clip (.mp4).
        ep_df:           Per-frame DataFrame for this episode.
        stability_row:   Row from evaluate_trajectory_stability() output.
        out_path:        Destination path for the annotated video.
        strip_height:    Pixel height of the stability strip.

    Returns:
        out_path
    """
    ep_df   = ep_df.sort_values("frame_index").reset_index(drop=True)
    actions = _to_array(ep_df["action"])
    states  = _to_array(ep_df["observation.state"])

    delta   = np.diff(actions, axis=0)
    jerk    = np.linalg.norm(delta, axis=1)
    s_final = states[-1]
    V       = np.sum((states - s_final) ** 2, axis=1)
    dV      = np.diff(V)
    disc_frames = list(stability_row["discontinuity_frames"])

    cap  = cv2.VideoCapture(str(video_path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-render full strip (rendered once, cursor added per-frame)
    base_strip = _render_stability_strip(jerk, dV, disc_frames, W, strip_height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    T_jerk    = len(jerk)
    score_txt = f"stability={stability_row['stability_score']:.3f}"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── Compositing: semi-transparent strip over bottom of frame ──────────
        strip_region = frame[H - strip_height:H, 0:W]
        blended      = cv2.addWeighted(base_strip, 0.80, strip_region, 0.20, 0)

        # Moving cursor
        cursor_x = int(np.clip(frame_idx / max(T_jerk, 1) * W, 0, W - 1))
        cv2.line(blended, (cursor_x, 0), (cursor_x, strip_height), (0, 255, 255), 2)

        # Current jerk label next to cursor
        if frame_idx < T_jerk:
            label_x = min(cursor_x + 4, W - 80)
            cv2.putText(blended, f"{jerk[frame_idx]:.1f}",
                        (label_x, 16), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # Score label bottom-left
        cv2.putText(blended, score_txt,
                    (6, strip_height - 6), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        frame[H - strip_height:H, 0:W] = blended
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Stability video saved to {out_path}")
    return out_path


def create_stability_videos(
    dataset_id: str,
    df: pd.DataFrame,
    stability_df: pd.DataFrame,
    output_dir: str = "stability_videos",
    max_episodes: int | None = None,
) -> None:
    """Extract each episode clip and burn in the stability overlay.

    Args:
        dataset_id:   HuggingFace dataset ID (used to locate the local cache).
        df:           Full rollout DataFrame from analyze_rollout().
        stability_df: Output of evaluate_trajectory_stability().
        output_dir:   Folder where annotated videos are written.
        max_episodes: If set, only process this many episodes.
    """
    from huggingface_hub import snapshot_download

    repo_dir  = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))
    cam_key   = "observation.images.camera1"
    video_dir = repo_dir / "videos" / cam_key
    out_dir   = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load episode timestamps from meta
    meta_files  = sorted((repo_dir / "meta" / "episodes").glob("**/*.parquet"))
    episodes_df = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

    # Join stability info by episode_index
    stab_lookup = stability_df.set_index("episode_index")

    rows = episodes_df.iterrows()
    n_done = 0
    for _, ep in rows:
        ep_idx  = int(ep["episode_index"])
        if ep_idx not in stab_lookup.index:
            continue

        chunk   = int(ep[f"videos/{cam_key}/chunk_index"])
        file_i  = int(ep[f"videos/{cam_key}/file_index"])
        from_ts = float(ep[f"videos/{cam_key}/from_timestamp"])
        to_ts   = float(ep[f"videos/{cam_key}/to_timestamp"])

        video_path = video_dir / f"chunk-{chunk:03d}" / f"file-{file_i:03d}.mp4"
        if not video_path.exists():
            print(f"  skipping ep {ep_idx}: video not found")
            continue

        clip_path = out_dir / f"episode_{ep_idx:04d}_clip.mp4"
        out_path  = out_dir / f"episode_{ep_idx:04d}_stability.mp4"

        print(f"  [{n_done + 1}] ep {ep_idx:4d}  extracting clip...")
        extract_episode_clip(video_path, from_ts, to_ts, clip_path)

        ep_df        = df[df["episode_index"] == ep_idx]
        stab_row     = stab_lookup.loc[ep_idx]
        # restore discontinuity_frames list (may have been dropped from CSV save)
        if not isinstance(stab_row.get("discontinuity_frames", None), list):
            stab_row = stab_row.copy()
            stab_row["discontinuity_frames"] = []

        overlay_stability(clip_path, ep_df, stab_row, out_path)
        clip_path.unlink(missing_ok=True)

        n_done += 1
        if max_episodes is not None and n_done >= max_episodes:
            break

    print(f"\nDone — {n_done} stability videos written to {output_dir}/")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory stability for a LeRobot dataset."
    )
    parser.add_argument("--dataset",      default=DATASET_ID)
    parser.add_argument("--threshold",    type=float, default=0.5,
                        help="Score below which an episode is flagged unstable.")
    parser.add_argument("--output",       default="stability_report.csv")
    parser.add_argument("--videos",       action="store_true",
                        help="Also produce per-episode stability videos.")
    parser.add_argument("--video-dir",    default="stability_videos")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Limit video generation to N episodes (0 = all).")
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    df = analyze_rollout(args.dataset)

    n_ep = df["episode_index"].nunique()
    print(f"Evaluating {n_ep} trajectories ({len(df)} total frames)...")
    stability_df = evaluate_trajectory_stability(df)

    print_stability_report(stability_df, unstable_threshold=args.threshold)

    out_df = stability_df.drop(columns=["discontinuity_frames"])
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    if args.videos:
        max_ep = args.max_episodes if args.max_episodes > 0 else None
        print(f"\nGenerating stability videos -> {args.video_dir}/")
        create_stability_videos(
            args.dataset, df, stability_df,
            output_dir=args.video_dir,
            max_episodes=max_ep,
        )
