"""
Detect pickup and drop frames for every episode using the gripper position signal.

Pickup = first frame where gripper closes (crosses GRIP_THRESHOLD going up).
Drop   = first frame where gripper opens  (crosses GRIP_THRESHOLD going down)
         after the pickup.

Output: outputs/episode_phases.parquet  (columns: episode_index, pickup_frame,
        drop_frame, pickup_time_s, drop_time_s, n_frames, hold_frames)

Usage:
    python batch_detect_phases.py
    python batch_detect_phases.py --threshold 20 --out outputs/my_phases.parquet
    python batch_detect_phases.py --plot          # show per-episode gripper plots
"""

import argparse
import glob
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

FPS = 30.0
GRIPPER_COL_IDX = 5        # index inside observation.state array
DEFAULT_THRESHOLD = 20.0   # degrees; below = open, above = closed/gripping
DEFAULT_OUT = Path(__file__).parent.parent / "outputs" / "episode_phases.parquet"


# ── detection ─────────────────────────────────────────────────────────────────

def detect_phases(gripper: np.ndarray, threshold: float) -> tuple[int | None, int | None]:
    """
    Returns (pickup_frame, drop_frame) as indices into the episode's frame array.

    The gripper signal has open→close→open cycles.  The robot may make
    multiple gripping attempts before a successful pick-and-place.
    We enumerate all cycles and return the one with the LONGEST hold duration
    (close→open interval), which corresponds to the successful grasp.

    pickup_frame: DOWN crossing of the longest cycle.
    drop_frame  : next UP crossing after that DOWN.
    """
    n = len(gripper)

    ups   = [i for i in range(1, n) if gripper[i - 1] < threshold <= gripper[i]]
    downs = [i for i in range(1, n) if gripper[i - 1] >= threshold > gripper[i]]

    if not ups or not downs:
        return None, None

    best_pickup = best_drop = None
    best_hold   = -1

    for down in downs:
        # must have at least one UP before this close (approach)
        if not any(u < down for u in ups):
            continue
        # drop = next UP after this close
        next_ups = [u for u in ups if u > down]
        if not next_ups:
            continue
        drop  = next_ups[0]
        hold  = drop - down
        if hold > best_hold:
            best_hold   = hold
            best_pickup = down
            best_drop   = drop

    return best_pickup, best_drop


# ── per-phase metrics ─────────────────────────────────────────────────────────

def _lyapunov_stats(states: np.ndarray) -> dict:
    """Compute violation_rate and decay_rate for a state sequence."""
    if len(states) < 2:
        return {"violation_rate": float("nan"), "decay_rate": float("nan")}
    s_ref = states[-1]
    V     = np.sum((states - s_ref) ** 2, axis=1)
    dV    = np.diff(V)
    viol  = float((dV > 0).mean())
    # decay rate from log-linear fit
    eps   = 1e-9
    t     = np.arange(len(V), dtype=float)
    logV  = np.log(V + eps)
    try:
        slope = float(np.polyfit(t, logV, 1)[0])
        decay = -slope  # positive = converging
    except Exception:
        decay = float("nan")
    return {"violation_rate": viol, "decay_rate": decay}


def compute_phase_metrics(
    actions: np.ndarray,
    states: np.ndarray,
    gripper: np.ndarray,
    pickup_frame: int | None,
    drop_frame: int | None,
    fps: float = FPS,
) -> dict:
    """Return a flat dict of per-episode and per-phase stability metrics.

    Phases:
      approach  — frame 0 → pickup_frame
      hold      — pickup_frame → drop_frame  (block in gripper)
      release   — drop_frame → end

    For each phase: lyapunov_violation_rate (fraction of steps V increases)
    and lyapunov_decay_rate (λ from log(V)~−λt; positive = converging).
    """
    n = len(actions)
    pf = pickup_frame if pickup_frame is not None else n
    df = drop_frame   if drop_frame   is not None else n

    # Overall jerk
    delta = np.diff(actions, axis=0)
    jerk  = np.linalg.norm(delta, axis=1)
    mean_jerk = float(jerk.mean())
    thresh_d  = mean_jerk + 3.0 * float(jerk.std())
    disc_count = int((jerk > thresh_d).sum())

    overall = _lyapunov_stats(states)

    # Phase slices (at least 2 frames each)
    approach_states = states[: max(pf, 2)]
    hold_states     = states[pf: max(df, pf + 2)]
    release_states  = states[df: max(n, df + 2)]

    approach = _lyapunov_stats(approach_states)
    hold     = _lyapunov_stats(hold_states)
    release  = _lyapunov_stats(release_states)

    return {
        "pickup_frame":               pickup_frame,
        "drop_frame":                 drop_frame,
        "pickup_time_s":              round(pf / fps, 3) if pickup_frame is not None else None,
        "drop_time_s":                round(df / fps, 3) if drop_frame   is not None else None,
        "hold_frames":                (df - pf) if (pickup_frame is not None and drop_frame is not None) else None,
        "hold_time_s":                round((df - pf) / fps, 3) if (pickup_frame is not None and drop_frame is not None) else None,
        "n_frames":                   n,
        "mean_jerk":                  round(mean_jerk, 4),
        "max_jerk":                   round(float(jerk.max()), 4),
        "discontinuity_count":        disc_count,
        "overall_violation_rate":     round(overall["violation_rate"], 4),
        "overall_decay_rate":         round(overall["decay_rate"], 6),
        "approach_violation_rate":    round(approach["violation_rate"], 4),
        "approach_decay_rate":        round(approach["decay_rate"], 6),
        "hold_violation_rate":        round(hold["violation_rate"], 4),
        "hold_decay_rate":            round(hold["decay_rate"], 6),
        "release_violation_rate":     round(release["violation_rate"], 4),
        "release_decay_rate":         round(release["decay_rate"], 6),
    }


# ── video concatenation ───────────────────────────────────────────────────────

def concatenate_videos(video_paths: list[Path], out_path: Path) -> None:
    """Concatenate a list of mp4 files into one using ffmpeg."""
    filelist = out_path.parent / "_concat_list.txt"
    filelist.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in video_paths),
        encoding="utf-8",
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(filelist),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an", str(out_path),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    filelist.unlink(missing_ok=True)
    print(f"Concatenated video -> {out_path}")


# ── video rendering ───────────────────────────────────────────────────────────

def _render_gripper_strip(
    gripper: np.ndarray,
    pickup_frame: int | None,
    drop_frame: int | None,
    threshold: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Pre-render the full-episode gripper plot as a BGR image (height, width, 3).

    Green region = holding (above threshold).
    Vertical lines mark pickup (lime) and drop (red).
    """
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#111111")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#111111")

    T = len(gripper)
    t = np.arange(T)

    ax.plot(t, gripper, color="#5dade2", linewidth=0.9, alpha=0.9)
    ax.axhline(threshold, color="#f39c12", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.fill_between(t, threshold, gripper, where=(gripper >= threshold),
                    color="#2ecc71", alpha=0.25)

    if pickup_frame is not None:
        ax.axvline(pickup_frame, color="#2ecc71", linewidth=1.5, label=f"pickup f{pickup_frame}")
    if drop_frame is not None:
        ax.axvline(drop_frame, color="#e74c3c", linewidth=1.5, label=f"drop f{drop_frame}")

    ax.set_xlim(0, T)
    ax.set_ylim(gripper.min() - 2, gripper.max() + 2)
    ax.set_ylabel("gripper (deg)", color="white", fontsize=7, labelpad=2)
    ax.tick_params(colors="white", labelsize=6, length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    if pickup_frame is not None or drop_frame is not None:
        ax.legend(loc="upper right", fontsize=6, facecolor="#111111",
                  edgecolor="#444444", labelcolor="white", framealpha=0.8)

    fig.tight_layout(pad=0.4)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba  = np.asarray(canvas.buffer_rgba())
    strip = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    strip = cv2.resize(strip, (width, height), interpolation=cv2.INTER_LINEAR)
    return strip


def _extract_clip(video_path: Path, from_ts: float, to_ts: float, out_path: Path) -> None:
    duration = to_ts - from_ts
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(from_ts),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an", str(out_path),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def overlay_phases(
    clip_path: Path,
    gripper: np.ndarray,
    pickup_frame: int | None,
    drop_frame: int | None,
    threshold: float,
    out_path: Path,
    strip_height: int = 130,
) -> None:
    """Burn the gripper strip onto the bottom of every frame of the clip."""
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_strip = _render_gripper_strip(gripper, pickup_frame, drop_frame,
                                       threshold, W, strip_height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_idx = 0
    T = len(gripper)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        strip_region = frame[H - strip_height:H, 0:W]
        blended = cv2.addWeighted(base_strip, 0.85, strip_region, 0.15, 0)

        # Moving cursor
        cursor_x = int(np.clip(frame_idx / max(T - 1, 1) * W, 0, W - 1))
        cv2.line(blended, (cursor_x, 0), (cursor_x, strip_height), (0, 255, 255), 2)

        # Current gripper value label
        if frame_idx < T:
            label_x = min(cursor_x + 4, W - 70)
            cv2.putText(blended, f"{gripper[frame_idx]:.1f}",
                        (label_x, 16), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        frame[H - strip_height:H, 0:W] = blended
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def create_combined_videos(
    dataset_id: str,
    phases_parquet: str | Path,
    output_dir: str = "combined_videos",
    threshold: float = DEFAULT_THRESHOLD,
    max_episodes: int | None = None,
    strip_height: int = 130,
    concat_out: str | None = None,
    metrics_csv: str | None = None,
) -> None:
    """Render a 3-panel video per episode: stability strip / camera / gripper strip.

    Top strip    — action jerk coloured by Lyapunov dV (green=converging, red=diverging)
    Middle       — raw camera frame with episode label
    Bottom strip — gripper position with pickup (green) and drop (red) markers

    Optionally concatenates all episode videos into one file and writes a
    per-phase metrics CSV.
    """
    from stability_eval import _render_stability_strip

    dataset_path = Path(dataset_id)
    if dataset_path.exists():
        repo_dir = dataset_path
    else:
        print(f"Locating dataset {dataset_id} ...")
        repo_dir = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))

    data_files  = sorted(glob.glob(str(repo_dir / "data" / "**" / "*.parquet"), recursive=True))
    df_data     = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    meta_files  = sorted((repo_dir / "meta" / "episodes").glob("**/*.parquet"))
    episodes_df = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    phases_df   = pd.read_parquet(phases_parquet).set_index("episode_index")

    cam_key   = "observation.images.camera1"
    video_dir = repo_dir / "videos" / cam_key
    out_dir   = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    produced_videos: list[Path] = []
    n_done = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    for _, ep in episodes_df.iterrows():
        ep_idx  = int(ep["episode_index"])
        if ep_idx not in phases_df.index:
            continue

        chunk   = int(ep[f"videos/{cam_key}/chunk_index"])
        file_i  = int(ep[f"videos/{cam_key}/file_index"])
        from_ts = float(ep[f"videos/{cam_key}/from_timestamp"])
        to_ts   = float(ep[f"videos/{cam_key}/to_timestamp"])

        video_path = video_dir / f"chunk-{chunk:03d}" / f"file-{file_i:03d}.mp4"
        if not video_path.exists():
            print(f"  skipping ep {ep_idx}: video not found")
            continue

        ep_df   = df_data[df_data["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)
        actions = np.stack(ep_df["action"].values).astype(np.float32)
        states  = np.stack(ep_df["observation.state"].values).astype(np.float32)
        gripper = states[:, GRIPPER_COL_IDX]

        # Stability
        delta       = np.diff(actions, axis=0)
        jerk        = np.linalg.norm(delta, axis=1)
        s_final     = states[-1]
        V           = np.sum((states - s_final) ** 2, axis=1)
        dV          = np.diff(V)
        mean_jerk   = jerk.mean()
        thresh_disc = mean_jerk + 3.0 * jerk.std()
        disc_frames = list(np.where(jerk > thresh_disc)[0].astype(int))

        phase_row    = phases_df.loc[ep_idx]
        pickup_frame = int(phase_row["pickup_frame"]) if pd.notna(phase_row["pickup_frame"]) else None
        drop_frame   = int(phase_row["drop_frame"])   if pd.notna(phase_row["drop_frame"])   else None

        # Metrics CSV row
        m = compute_phase_metrics(actions, states, gripper, pickup_frame, drop_frame)
        metrics_rows.append({"episode_index": ep_idx, **m})

        # Extract clip
        clip_path = out_dir / f"episode_{ep_idx:04d}_clip.mp4"
        out_path  = out_dir / f"episode_{ep_idx:04d}_combined.mp4"
        print(f"  [{n_done + 1}] ep {ep_idx:4d}  pickup={pickup_frame}  drop={drop_frame}  extracting...")
        _extract_clip(video_path, from_ts, to_ts, clip_path)

        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        top_strip = _render_stability_strip(jerk, dV, disc_frames, W, strip_height)
        bot_strip = _render_gripper_strip(gripper, pickup_frame, drop_frame, threshold, W, strip_height)

        total_H = strip_height + H + strip_height
        writer  = cv2.VideoWriter(str(out_path),
                                  cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, total_H))

        T_jerk    = len(jerk)
        T_gripper = len(gripper)

        # Label shown on camera frame
        ep_label      = f"Episode {ep_idx}"
        hold_viol     = m["hold_violation_rate"]
        trend_str     = "DIVERGING" if hold_viol > 0.5 else "converging"
        trend_color   = (0, 0, 220) if hold_viol > 0.5 else (0, 200, 80)

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Top strip + cursor
            top = top_strip.copy()
            cx  = int(np.clip(frame_idx / max(T_jerk, 1) * W, 0, W - 1))
            cv2.line(top, (cx, 0), (cx, strip_height), (0, 255, 255), 2)
            if frame_idx < T_jerk:
                cv2.putText(top, f"jerk {jerk[frame_idx]:.1f}",
                            (min(cx + 4, W - 80), 16), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            # Camera frame labels
            cv2.putText(frame, ep_label, (8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"hold lyap: {trend_str}",
                        (8, 50), font, 0.55, trend_color, 1, cv2.LINE_AA)
            # Phase marker
            if pickup_frame is not None and frame_idx == pickup_frame:
                cv2.putText(frame, "PICKUP", (W // 2 - 40, H // 2),
                            font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            if drop_frame is not None and frame_idx == drop_frame:
                cv2.putText(frame, "DROP", (W // 2 - 30, H // 2),
                            font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            # Bottom strip + cursor
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
        clip_path.unlink(missing_ok=True)
        produced_videos.append(out_path)
        print(f"           -> {out_path}")

        n_done += 1
        if max_episodes is not None and n_done >= max_episodes:
            break

    print(f"\nDone — {n_done} combined videos written to {output_dir}/")

    # ── metrics CSV ───────────────────────────────────────────────────────────
    if metrics_rows:
        csv_path = Path(metrics_csv) if metrics_csv else out_dir / "phase_metrics.csv"
        pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)
        print(f"Phase metrics CSV -> {csv_path}")

    # ── concatenate ───────────────────────────────────────────────────────────
    if concat_out and produced_videos:
        concat_path = Path(concat_out)
        concat_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nConcatenating {len(produced_videos)} videos -> {concat_path}")
        concatenate_videos(produced_videos, concat_path)


def create_phase_videos(
    dataset_id: str,
    phases_parquet: str | Path,
    output_dir: str = "phase_videos",
    threshold: float = DEFAULT_THRESHOLD,
    max_episodes: int | None = None,
    strip_height: int = 130,
) -> None:
    """Extract episode clips and burn in the gripper phase strip.

    Args:
        dataset_id:      HuggingFace dataset ID (or local path).
        phases_parquet:  Path to episode_phases.parquet from batch_detect_phases.
        output_dir:      Folder where annotated videos are written.
        threshold:       Gripper threshold used for detection (drawn on plot).
        max_episodes:    If set, only process this many episodes.
        strip_height:    Pixel height of the gripper strip.
    """
    dataset_path = Path(dataset_id)
    if dataset_path.exists():
        repo_dir = dataset_path
    else:
        print(f"Locating dataset {dataset_id} ...")
        repo_dir = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))

    # Load data parquets
    data_files = sorted(glob.glob(str(repo_dir / "data" / "**" / "*.parquet"), recursive=True))
    df_data = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)

    # Load meta for timestamps
    meta_files  = sorted((repo_dir / "meta" / "episodes").glob("**/*.parquet"))
    episodes_df = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

    # Load phases
    phases_df = pd.read_parquet(phases_parquet).set_index("episode_index")

    cam_key   = "observation.images.camera1"
    video_dir = repo_dir / "videos" / cam_key
    out_dir   = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for _, ep in episodes_df.iterrows():
        ep_idx = int(ep["episode_index"])
        if ep_idx not in phases_df.index:
            continue

        chunk   = int(ep[f"videos/{cam_key}/chunk_index"])
        file_i  = int(ep[f"videos/{cam_key}/file_index"])
        from_ts = float(ep[f"videos/{cam_key}/from_timestamp"])
        to_ts   = float(ep[f"videos/{cam_key}/to_timestamp"])

        video_path = video_dir / f"chunk-{chunk:03d}" / f"file-{file_i:03d}.mp4"
        if not video_path.exists():
            print(f"  skipping ep {ep_idx}: video not found")
            continue

        ep_df   = df_data[df_data["episode_index"] == ep_idx].sort_values("frame_index")
        states  = np.stack(ep_df["observation.state"].values)
        gripper = states[:, GRIPPER_COL_IDX]

        phase_row    = phases_df.loc[ep_idx]
        pickup_frame = int(phase_row["pickup_frame"]) if pd.notna(phase_row["pickup_frame"]) else None
        drop_frame   = int(phase_row["drop_frame"])   if pd.notna(phase_row["drop_frame"])   else None

        clip_path = out_dir / f"episode_{ep_idx:04d}_clip.mp4"
        out_path  = out_dir / f"episode_{ep_idx:04d}_phases.mp4"

        print(f"  [{n_done + 1}] ep {ep_idx:4d}  "
              f"pickup={pickup_frame}  drop={drop_frame}  extracting clip...")
        _extract_clip(video_path, from_ts, to_ts, clip_path)
        overlay_phases(clip_path, gripper, pickup_frame, drop_frame,
                       threshold, out_path, strip_height)
        clip_path.unlink(missing_ok=True)
        print(f"           -> {out_path}")

        n_done += 1
        if max_episodes is not None and n_done >= max_episodes:
            break

    print(f"\nDone — {n_done} phase videos written to {output_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch detect pickup/drop frames from gripper signal")
    parser.add_argument("--dataset", type=str,
                        default="nc8304/eval_smolvla-phase-split_combined",
                        help="HuggingFace dataset ID or local path to dataset root")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Gripper position threshold in degrees (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="Output parquet path")
    parser.add_argument("--plot", action="store_true",
                        help="Show gripper plot for each episode")
    parser.add_argument("--videos", action="store_true",
                        help="Render phase overlay videos after detection")
    parser.add_argument("--combined", action="store_true",
                        help="Render 3-panel combined videos (stability / camera / gripper)")
    parser.add_argument("--video-dir", default="phase_videos",
                        help="Output folder for phase videos")
    parser.add_argument("--combined-dir", default="combined_videos",
                        help="Output folder for combined videos")
    parser.add_argument("--concat-out", default="combined_videos/all_episodes.mp4",
                        help="Output path for concatenated video (used with --combined)")
    parser.add_argument("--metrics-csv", default=None,
                        help="Output path for phase metrics CSV (default: combined_videos/phase_metrics.csv)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of videos to render")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        root = dataset_path
    else:
        print(f"Downloading dataset {args.dataset} ...")
        root = Path(snapshot_download(repo_id=args.dataset, repo_type="dataset"))

    parquet_files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root}/data/")
    print(f"Loading {len(parquet_files)} parquet file(s) from {root} ...")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    episodes = sorted(df["episode_index"].unique())
    print(f"Episodes: {len(episodes)}  |  threshold: {args.threshold} deg\n")

    rows = []
    failed = []

    for ep_idx in episodes:
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        states = np.stack(ep_df["observation.state"].values)
        gripper = states[:, GRIPPER_COL_IDX]
        n_frames = len(gripper)

        pf, df_ = detect_phases(gripper, args.threshold)

        if pf is None:
            print(f"  ep {ep_idx:3d}: [FAIL] no pickup detected  "
                  f"(gripper range: {gripper.min():.1f}–{gripper.max():.1f})")
            failed.append(ep_idx)
            rows.append(dict(episode_index=ep_idx,
                             pickup_frame=None, drop_frame=None,
                             pickup_time_s=None, drop_time_s=None,
                             n_frames=n_frames, hold_frames=None))
            continue

        hold_frames = (df_ - pf) if df_ is not None else None
        status = "OK" if df_ is not None else "no-drop"
        print(f"  ep {ep_idx:3d}: pickup={pf:4d} ({pf/FPS:.2f}s)  "
              f"drop={str(df_):>5} ({(df_/FPS if df_ else 0):.2f}s)  "
              f"hold={hold_frames}f  [{status}]")

        rows.append(dict(
            episode_index=ep_idx,
            pickup_frame=pf,
            drop_frame=df_,
            pickup_time_s=round(pf / FPS, 3),
            drop_time_s=round(df_ / FPS, 3) if df_ is not None else None,
            n_frames=n_frames,
            hold_frames=hold_frames,
        ))

        if args.plot:
            _plot_episode(gripper, pf, df_, ep_idx, args.threshold)

    result = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    ok = result["pickup_frame"].notna().sum()
    has_drop = result["drop_frame"].notna().sum()
    print(f"Summary: {ok}/{len(episodes)} episodes have pickup  |  "
          f"{has_drop}/{len(episodes)} have both pickup+drop")
    if failed:
        print(f"Failed episodes: {failed}")

    if args.videos:
        print(f"\nRendering phase videos -> {args.video_dir}/")
        create_phase_videos(
            dataset_id=args.dataset,
            phases_parquet=out_path,
            output_dir=args.video_dir,
            threshold=args.threshold,
            max_episodes=args.max_episodes,
        )

    if args.combined:
        print(f"\nRendering combined videos -> {args.combined_dir}/")
        create_combined_videos(
            dataset_id=args.dataset,
            phases_parquet=out_path,
            output_dir=args.combined_dir,
            threshold=args.threshold,
            max_episodes=args.max_episodes,
            concat_out=args.concat_out,
            metrics_csv=args.metrics_csv,
        )


def _plot_episode(gripper: np.ndarray, pickup: int | None,
                  drop: int | None, ep_idx: int, threshold: float):
    import matplotlib.pyplot as plt
    t = np.arange(len(gripper)) / FPS
    plt.figure(figsize=(12, 3))
    plt.plot(t, gripper, color="steelblue", linewidth=0.8, label="gripper.pos")
    plt.axhline(threshold, color="orange", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    if pickup is not None:
        plt.axvline(pickup / FPS, color="lime", linewidth=1.5, label=f"pickup f{pickup}")
    if drop is not None:
        plt.axvline(drop / FPS, color="tomato", linewidth=1.5, label=f"drop f{drop}")
    plt.title(f"Episode {ep_idx} — gripper position")
    plt.xlabel("time (s)")
    plt.ylabel("gripper.pos (deg)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
