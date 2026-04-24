"""
Phase viewer — scrub through all episodes with a single slider.
Shows PICKUP | HOLD | DROP frames side-by-side for each episode.

Usage:
    python phase_viewer.py                        # original dataset
    python phase_viewer.py --variant blue_a05     # color-augmented variant
    python phase_viewer.py --cached               # reload from saved cache
    python phase_viewer.py --variant red_a07 --cached
    python phase_viewer.py --video-dir C:/Users/calle/Downloads   # custom video files

Controls:
    Drag "Episode" trackbar   jump to any episode
    LEFT / RIGHT              ±1 episode
    q / Esc                   quit
"""

import argparse
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
ORIG_ROOT  = Path(
    r"C:\Users\calle\.cache\huggingface\hub"
    r"\datasets--nc8304--so101_combined_cubeONLY"
    r"\snapshots\acc242c231f60171a5b2833442d176cd793ea8c9"
)
AUG_ROOT   = Path(__file__).parent.parent / "outputs" / "augmented_datasets"
VIDEO_KEY  = "observation.images.front"
FPS        = 30.0
PHASES_IN  = Path(__file__).parent.parent / "outputs" / "episode_phases.parquet"
OUTPUTS    = Path(__file__).parent.parent / "outputs"

THUMB_W, THUMB_H = 426, 320   # each thumbnail  (3×426 = 1278 wide)
GAP              = 6
LABEL_H          = 32
HEADER_H         = 26
STRIP_W          = THUMB_W * 3 + GAP * 2
STRIP_H          = HEADER_H + THUMB_H + LABEL_H

C_PICKUP = (0,   210,  60)
C_HOLD   = (0,   200, 200)
C_DROP   = (60,   60, 220)
LABELS   = ["PICKUP", "HOLD", "DROP"]
COLORS   = [C_PICKUP, C_HOLD, C_DROP]


# ── frame decoding ────────────────────────────────────────────────────────────

def episode_video_info(ep_df, ep_idx, dataset_root, video_dir=None):
    row      = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
    file_idx = int(row[f"videos/{VIDEO_KEY}/file_index"])
    from_ts  = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
    if video_dir is not None:
        vid_path = Path(video_dir) / f"file-{file_idx:03d}.mp4"
    else:
        vid_path = dataset_root / "videos" / VIDEO_KEY / "chunk-000" / f"file-{file_idx:03d}.mp4"
    return vid_path, from_ts


def decode_frames_at(vid_path, from_ts, local_targets):
    """Seek to episode start, step forward, capture frames at local_targets."""
    targets  = sorted(t for t in local_targets if t is not None)
    captured = {}
    if not targets:
        return captured

    with av.open(str(vid_path)) as c:
        stream = c.streams.video[0]
        c.seek(int(from_ts * 1_000_000))
        local = 0
        for pkt in c.demux(stream):
            if len(captured) == len(targets) or local > targets[-1] + 5:
                break
            try:
                for frame in pkt.decode():
                    if local in targets:
                        img = frame.to_ndarray(format="bgr24")
                        captured[local] = cv2.resize(img, (THUMB_W, THUMB_H))
                    local += 1
                    if local > targets[-1] + 5:
                        break
            except av.error.InvalidDataError:
                pass
    return captured


# ── composite builder ─────────────────────────────────────────────────────────

def make_strip(ep_idx, pickup, hold, drop, decoded):
    """Build one STRIP_H × STRIP_W BGR image for this episode."""

    # header
    header = np.full((HEADER_H, STRIP_W, 3), 18, dtype=np.uint8)
    hold_str = f"hold {drop - pickup}f  ({(drop - pickup)/FPS:.1f}s)" \
               if (pickup is not None and drop is not None) else ""
    cv2.putText(header, f"Episode {ep_idx}   {hold_str}",
                (8, HEADER_H - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (230, 230, 230), 1, cv2.LINE_AA)

    panels = []
    for frame_no, label, color in zip([pickup, hold, drop], LABELS, COLORS):
        # thumbnail
        img = decoded.get(frame_no)
        if img is None:
            thumb = np.full((THUMB_H, THUMB_W, 3), 35, dtype=np.uint8)
            cv2.putText(thumb, "N/A", (THUMB_W // 2 - 28, THUMB_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (110, 110, 110), 2)
        else:
            thumb = img.copy()
        cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1), color, 4)

        # label bar
        bar = np.full((LABEL_H, THUMB_W, 3), 22, dtype=np.uint8)
        frame_str = str(frame_no) if frame_no is not None else "?"
        txt = f"{label}   f {frame_str}"
        cv2.putText(bar, txt, (8, LABEL_H - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

        panels.append(np.vstack([thumb, bar]))

    gap = np.full((THUMB_H + LABEL_H, GAP, 3), 8, dtype=np.uint8)
    row = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    return np.vstack([header, row])


# ── build / load cache ────────────────────────────────────────────────────────

def build_frames(ep_df, phases, episodes, dataset_root, video_dir=None):
    """Decode and return list of strip images (one per episode)."""
    strips = []
    n = len(episodes)
    for i, ep_idx in enumerate(episodes):
        row = phases[phases["episode_index"] == ep_idx]
        if row.empty:
            pickup = drop = None
        else:
            r = row.iloc[0]
            pickup = None if pd.isna(r["pickup_frame"]) else int(r["pickup_frame"])
            drop   = None if pd.isna(r["drop_frame"])   else int(r["drop_frame"])
        hold = ((pickup + drop) // 2) if (pickup is not None and drop is not None) else None

        decoded = {}
        targets = [t for t in [pickup, hold, drop] if t is not None]
        if targets:
            vid_path, from_ts = episode_video_info(ep_df, ep_idx, dataset_root, video_dir)
            decoded = decode_frames_at(vid_path, from_ts, targets)

        strip = make_strip(ep_idx, pickup, hold, drop, decoded)
        strips.append(strip)

        # show progress in a preview window
        pct = int((i + 1) / n * (STRIP_W - 20))
        prog = np.full((40, STRIP_W, 3), 20, dtype=np.uint8)
        cv2.rectangle(prog, (10, 12), (10 + pct, 28), (60, 160, 60), -1)
        cv2.putText(prog, f"Building cache: {i + 1}/{n}",
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Phase Viewer", np.vstack([strip, prog]))
        cv2.waitKey(1)

    return strips


def save_cache(strips, path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w   = strips[0].shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, 1.0, (w, h))
    for s in strips:
        writer.write(s)
    writer.release()
    print(f"Cache saved -> {path}")


def load_cache(path):
    cap = cv2.VideoCapture(str(path))
    strips = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        strips.append(frame)
    cap.release()
    return strips


# ── viewer ────────────────────────────────────────────────────────────────────

def run_viewer(strips):
    n = len(strips)
    WIN = "Phase Viewer"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, STRIP_W, STRIP_H + 40)

    ep_pos = [0]

    def on_tb(val):
        ep_pos[0] = val

    cv2.createTrackbar("Episode", WIN, 0, n - 1, on_tb)

    while True:
        idx   = ep_pos[0]
        img   = strips[idx]
        cv2.imshow(WIN, img)

        key = cv2.waitKeyEx(20)
        if key == -1:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        if key in (2424832, 65361, ord("a")):    # LEFT
            ep_pos[0] = max(idx - 1, 0)
            cv2.setTrackbarPos("Episode", WIN, ep_pos[0])
        elif key in (2555904, 65363, ord("d")):  # RIGHT
            ep_pos[0] = min(idx + 1, n - 1)
            cv2.setTrackbarPos("Episode", WIN, ep_pos[0])
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default=None,
                        help="Color-augmented variant name, e.g. 'blue_a05'. "
                             "Omit to use the original dataset.")
    parser.add_argument("--cached", action="store_true",
                        help="Load from saved cache instead of re-decoding")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory containing file-000.mp4, file-001.mp4 … "
                             "(overrides dataset video path; parquet metadata still "
                             "comes from the original dataset root)")
    args = parser.parse_args()

    # resolve dataset root and cache path
    if args.variant:
        dataset_root = AUG_ROOT / args.variant
        if not dataset_root.exists():
            print(f"Variant not found: {dataset_root}")
            print("Available:", [d.name for d in AUG_ROOT.iterdir() if d.is_dir()])
            return
        cache_vid = OUTPUTS / f"phase_viewer_cache_{args.variant}.mp4"
        label = args.variant
    else:
        dataset_root = ORIG_ROOT
        cache_vid    = OUTPUTS / "phase_viewer_cache.mp4"
        label        = "original"

    cv2.namedWindow("Phase Viewer", cv2.WINDOW_NORMAL)

    if args.cached and cache_vid.exists():
        print(f"Loading cache ({label}): {cache_vid}")
        strips = load_cache(cache_vid)
    else:
        print(f"Loading metadata ({label})...")
        ep_df    = pd.read_parquet(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        phases   = pd.read_parquet(PHASES_IN)
        episodes = sorted(ep_df["episode_index"].unique().tolist())
        strips   = build_frames(ep_df, phases, episodes, dataset_root,
                                video_dir=args.video_dir)
        save_cache(strips, cache_vid)

    print(f"Loaded {len(strips)} episodes ({label}). Opening viewer...")
    run_viewer(strips)


if __name__ == "__main__":
    main()
