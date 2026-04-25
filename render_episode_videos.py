"""
Render annotated videos for all 80 original episodes, concatenated into one file.

Each episode stacks:
    ┌──────────────────────┐
    │   camera frame        │  640 × 480
    │   (pickup/drop marks) │
    ├──────────────────────┤
    │   gripper signal plot │  640 × 240
    │   with current-frame  │
    │   vertical line       │
    └──────────────────────┘

Output:  outputs/all_episodes_annotated.mp4   (all 80 episodes in one file)

Usage:
    python render_episode_videos.py
    python render_episode_videos.py --episodes 0 1 2   # specific episodes only
    python render_episode_videos.py --out outputs/my_review.mp4
"""

import argparse
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

# re-use constants and helpers from episode_labeler
from episode_labeler import (
    ORIG_ROOT, VIDEO_KEY, FPS,
    FRAME_W, FRAME_H, PLOT_H,
    C_PICKUP, C_DROP, C_CURR,
    load_meta, episode_info, gripper_signal, build_base_plot,
)

DEFAULT_OUT = Path(__file__).parent.parent / "outputs" / "all_episodes_annotated.mp4"
PHASES_CSV  = Path(__file__).parent.parent / "outputs" / "episode_phases.csv"


# ── frame decoding (synchronous, no background thread needed) ─────────────────

def decode_all_frames(vid_path: Path, from_ts: float, n_frames: int) -> list[np.ndarray]:
    """Decode every frame for one episode using PyAV (handles AV1)."""
    frames = []
    with av.open(str(vid_path)) as c:
        stream = c.streams.video[0]
        c.seek(int(from_ts * 1_000_000))
        for pkt in c.demux(stream):
            if len(frames) >= n_frames:
                break
            try:
                for frame in pkt.decode():
                    if len(frames) >= n_frames:
                        break
                    img = frame.to_ndarray(format="bgr24")
                    if img.shape[:2] != (FRAME_H, FRAME_W):
                        img = cv2.resize(img, (FRAME_W, FRAME_H))
                    frames.append(img)
            except av.error.InvalidDataError:
                pass
    return frames


# ── per-frame overlay (simplified from episode_labeler — no UI controls) ──────

def annotate_frame(frame: np.ndarray, ep_idx: int,
                   local_frame: int, n_frames: int,
                   pickup: int | None, drop: int | None) -> np.ndarray:
    """Draw header text and coloured border on a copy of the camera frame."""
    img = frame.copy()
    h, w = img.shape[:2]

    # coloured border at pickup and drop frames
    if local_frame == pickup:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), C_PICKUP, 6)
    elif local_frame == drop:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), C_DROP, 6)

    # header bar
    cv2.rectangle(img, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(img,
                f"Episode {ep_idx}   frame {local_frame}/{n_frames - 1}"
                f"  ({local_frame / FPS:.2f}s)",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)

    pk_str = f"PICKUP = {pickup}" if pickup is not None else "PICKUP = --"
    dr_str = f"DROP   = {drop}"   if drop   is not None else "DROP = --"
    cv2.putText(img, pk_str, (8,   38), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                C_PICKUP if pickup is not None else (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, dr_str, (210, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                C_DROP   if drop   is not None else (90, 90, 90), 1, cv2.LINE_AA)
    return img


# ── render all episodes into one video ────────────────────────────────────────

def render_all(episodes: list[int], ep_df: pd.DataFrame, data_df: pd.DataFrame,
               phases: dict, out_path: Path) -> None:
    """
    Decode all episodes and write them sequentially into a single MP4.
    A 2-second title card is inserted between episodes so you can tell them apart.
    """
    out_h      = FRAME_H + PLOT_H
    speed      = 2.0                       # playback speed multiplier
    out_fps    = FPS * speed               # write at 60 fps -> plays back 2x faster
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(str(out_path), fourcc, out_fps, (FRAME_W, out_h))
    speed_str  = f"{speed:.0f}x"          # label shown in top-left corner

    title_frames = int(out_fps * 2)   # 2-second title card between episodes

    for i, ep_idx in enumerate(episodes):
        # ── phase labels ──────────────────────────────────────────────────────
        p      = phases.get(ep_idx, {})
        pickup = p.get("pickup")
        drop   = p.get("drop")

        # ── episode source ────────────────────────────────────────────────────
        vid_path, from_ts, n_frames = episode_info(ep_df, ep_idx)

        # ── gripper plot (static base rendered once per episode) ──────────────
        grip = gripper_signal(data_df, ep_idx)
        base_plot, frame_to_x = build_base_plot(grip, pickup, drop,
                                                 auto_pk=pickup, auto_dr=drop)

        # ── title card ────────────────────────────────────────────────────────
        card = np.full((out_h, FRAME_W, 3), 15, dtype=np.uint8)
        cv2.putText(card, f"Episode {ep_idx}  ({i+1}/{len(episodes)})",
                    (40, out_h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (220, 220, 220), 2, cv2.LINE_AA)
        pk_str = f"pickup f{pickup}" if pickup is not None else "pickup: undetected"
        dr_str = f"drop   f{drop}"   if drop   is not None else "drop:   undetected"
        cv2.putText(card, pk_str, (40, out_h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_PICKUP, 1, cv2.LINE_AA)
        cv2.putText(card, dr_str, (40, out_h // 2 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_DROP,   1, cv2.LINE_AA)
        for _ in range(title_frames):
            writer.write(card)

        # ── decode frames ─────────────────────────────────────────────────────
        print(f"  [{i+1:2d}/{len(episodes)}] ep {ep_idx:3d}: "
              f"decoding {n_frames} frames ...", end=" ", flush=True)
        raw_frames = decode_all_frames(vid_path, from_ts, n_frames)
        print(f"got {len(raw_frames)}")

        # ── write frames ──────────────────────────────────────────────────────
        for fi, raw in enumerate(raw_frames):
            vid_img  = annotate_frame(raw, ep_idx, fi, n_frames, pickup, drop)

            # speed label — top-left corner, inside a small dark pill
            label_x, label_y = 8, 60
            (tw, th), _ = cv2.getTextSize(speed_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vid_img,
                          (label_x - 4, label_y - th - 4),
                          (label_x + tw + 4, label_y + 4),
                          (0, 0, 0), -1)
            cv2.putText(vid_img, speed_str,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 230), 2, cv2.LINE_AA)

            plot_img = base_plot.copy()
            cx = frame_to_x(fi)
            cv2.line(plot_img, (cx, 0), (cx, PLOT_H), C_CURR, 2)
            writer.write(np.vstack([vid_img, plot_img]))

    writer.release()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render gripper-annotated videos for all episodes into one MP4")
    parser.add_argument("--episodes", nargs="*", type=int, default=None,
                        help="Episode indices to render (default: all 80)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="Output MP4 path (default: outputs/all_episodes_annotated.mp4)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset metadata ...")
    ep_df, data_df = load_meta()
    all_episodes = sorted(ep_df["episode_index"].unique().tolist())
    episodes = args.episodes if args.episodes is not None else all_episodes

    # load phase labels
    phases: dict[int, dict] = {}
    if PHASES_CSV.exists():
        df = pd.read_csv(PHASES_CSV)
        for _, row in df.iterrows():
            ei = int(row["episode_index"])
            phases[ei] = {
                "pickup": None if pd.isna(row["pickup_frame"]) else int(row["pickup_frame"]),
                "drop":   None if pd.isna(row["drop_frame"])   else int(row["drop_frame"]),
            }
    else:
        print(f"WARNING: {PHASES_CSV} not found — videos will have no phase markers")

    print(f"Rendering {len(episodes)} episodes -> {out_path}\n")
    render_all(episodes, ep_df, data_df, phases, out_path)
    print(f"\nDone -> {out_path}")


if __name__ == "__main__":
    main()
