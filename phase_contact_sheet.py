"""
Generate a contact sheet with 3 frames per episode:
  [PICKUP]  [HOLD CENTER]  [DROP]

Output: outputs/phase_contact_sheet.png
"""

from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
ORIG_ROOT = Path(
    r"C:\Users\calle\.cache\huggingface\hub"
    r"\datasets--nc8304--so101_combined_cubeONLY"
    r"\snapshots\acc242c231f60171a5b2833442d176cd793ea8c9"
)
VIDEO_KEY  = "observation.images.front"
FPS        = 30.0
PHASES_IN  = Path(__file__).parent.parent / "outputs" / "episode_phases.parquet"
OUT_IMG    = Path(__file__).parent.parent / "outputs" / "phase_contact_sheet.png"

THUMB_W, THUMB_H = 320, 240   # each thumbnail
LABEL_H          = 28          # text row below each thumbnail strip
GAP              = 4           # gap between thumbnails
STRIP_W          = THUMB_W * 3 + GAP * 2
STRIP_H          = THUMB_H + LABEL_H

C_PICKUP = (0,   200,  50)
C_HOLD   = (200, 200,   0)
C_DROP   = (50,   50, 220)
LABELS   = ["PICKUP", "HOLD", "DROP"]
COLORS   = [C_PICKUP, C_HOLD, C_DROP]


# ── helpers ───────────────────────────────────────────────────────────────────

def episode_video_info(ep_df, ep_idx):
    row      = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
    file_idx = int(row[f"videos/{VIDEO_KEY}/file_index"])
    from_ts  = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
    vid_path = ORIG_ROOT / "videos" / VIDEO_KEY / "chunk-000" / f"file-{file_idx:03d}.mp4"
    return vid_path, from_ts


def decode_frames_at(vid_path: Path, from_ts: float, local_targets: list[int]) -> dict[int, np.ndarray]:
    """
    Seek to episode start (from_ts) then step forward, capturing frames at
    local_targets (sorted ascending). Returns {local_frame: bgr_array}.
    """
    targets = sorted(set(local_targets))
    captured = {}

    with av.open(str(vid_path)) as c:
        stream = c.streams.video[0]
        c.seek(int(from_ts * 1_000_000))
        local = 0
        for pkt in c.demux(stream):
            if len(captured) == len(targets):
                break
            if local > targets[-1] + 5:   # overshot, stop
                break
            try:
                for frame in pkt.decode():
                    if local in targets:
                        img = frame.to_ndarray(format="bgr24")
                        img = cv2.resize(img, (THUMB_W, THUMB_H))
                        captured[local] = img
                    local += 1
                    if local > targets[-1] + 5:
                        break
            except av.error.InvalidDataError:
                pass

    return captured


def make_thumbnail(img: np.ndarray | None, label: str, color, ep_idx: int, frame_no: int | None) -> np.ndarray:
    """Return (THUMB_H + LABEL_H) × THUMB_W BGR strip."""
    if img is None:
        thumb = np.full((THUMB_H, THUMB_W, 3), 40, dtype=np.uint8)
        cv2.putText(thumb, "N/A", (THUMB_W // 2 - 20, THUMB_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
    else:
        thumb = img.copy()

    # coloured border
    cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1), color, 3)

    # label bar below
    bar = np.full((LABEL_H, THUMB_W, 3), 20, dtype=np.uint8)
    frame_str = f"f{frame_no}" if frame_no is not None else "?"
    cv2.putText(bar, f"{label}  {frame_str}",
                (6, LABEL_H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    return np.vstack([thumb, bar])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading metadata...")
    ep_df   = pd.read_parquet(ORIG_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    phases  = pd.read_parquet(PHASES_IN)

    episodes = sorted(ep_df["episode_index"].unique().tolist())
    n_ep     = len(episodes)

    rows = []

    for i, ep_idx in enumerate(episodes):
        row = phases[phases["episode_index"] == ep_idx]
        if row.empty:
            pickup = drop = None
        else:
            r = row.iloc[0]
            pickup = None if pd.isna(r["pickup_frame"]) else int(r["pickup_frame"])
            drop   = None if pd.isna(r["drop_frame"])   else int(r["drop_frame"])

        hold = ((pickup + drop) // 2) if (pickup is not None and drop is not None) else None

        print(f"  ep {ep_idx:3d} ({i+1}/{n_ep})  pickup={pickup}  hold={hold}  drop={drop}")

        # decode 3 frames in one pass
        targets = [t for t in [pickup, hold, drop] if t is not None]
        decoded = {}
        if targets:
            vid_path, from_ts = episode_video_info(ep_df, ep_idx)
            decoded = decode_frames_at(vid_path, from_ts, targets)

        # build 3 thumbnails
        thumbs = []
        for frame_no, label, color in zip([pickup, hold, drop], LABELS, COLORS):
            img = decoded.get(frame_no) if frame_no is not None else None
            thumbs.append(make_thumbnail(img, label, color, ep_idx, frame_no))

        # episode header
        header = np.full((22, STRIP_W, 3), 15, dtype=np.uint8)
        cv2.putText(header, f"Episode {ep_idx}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        strip = np.hstack([
            thumbs[0],
            np.full((STRIP_H, GAP, 3), 10, dtype=np.uint8),
            thumbs[1],
            np.full((STRIP_H, GAP, 3), 10, dtype=np.uint8),
            thumbs[2],
        ])
        rows.append(np.vstack([header, strip]))

    sheet = np.vstack(rows)
    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_IMG), sheet)
    print(f"\nSaved -> {OUT_IMG}  ({sheet.shape[1]}×{sheet.shape[0]} px)")


if __name__ == "__main__":
    main()
