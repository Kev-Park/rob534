"""
Build augmented LeRobot datasets from pre-recolored video files.

For each color+alpha the script:
  1. Counts actual frames in each recolored AVI via PyAV (cv2 cannot read AV1)
  2. Finds episodes fully covered by the recolored frames
  3. Transcodes the AVI → AV1 MP4 (matching the original codec/format)
  4. Copies only the motor parquet rows for complete episodes
  5. Writes a valid LeRobot v3.0 dataset at outputs/augmented_datasets/<color>_a<alpha>/

NOTE: the original dataset uses AV1 video which cv2 cannot fully decode.
      batch_recolor.py therefore produced partial AVIs (cv2 stopped early).
      This script handles partial coverage gracefully by only keeping episodes
      whose video is fully present in the recolored file.

Usage:
    python make_augmented_dataset.py --color blue --alpha 0.5
    python make_augmented_dataset.py --all          # all 13 colors × 3 alphas

    # To merge augmented variants with the original for training:
    #   from lerobot.datasets.dataset_tools import merge_datasets
    #   from lerobot.datasets.lerobot_dataset import LeRobotDataset
    #   orig  = LeRobotDataset("nc8304/so101_combined")
    #   aug   = LeRobotDataset("nc8304/so101_blue_a05", root=<aug_root>)
    #   merge_datasets([orig, aug], "nc8304/so101_augmented", output_dir=<out>)
"""

import argparse
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import av
import pandas as pd
from tqdm import tqdm

# ── config ─────────────────────────────────────────────────────────────────────
ORIG_ROOT   = Path(r"C:\Users\calle\.cache\huggingface\lerobot\nc8304\so101_combined")
BATCH_ROOT  = Path(__file__).parent.parent / "outputs" / "batch"
OUTPUT_ROOT = Path(__file__).parent.parent / "outputs" / "augmented_datasets"

VIDEO_KEY   = "observation.images.front"
FPS         = 30
VCODEC      = "libsvtav1"
PIX_FMT     = "yuv420p"

# (chunk_index, file_index) pairs that exist in the original dataset
CHUNK_FILES = [(0, 0), (0, 1)]

COLORS = [
    "red", "orange", "yellow", "lime", "green", "teal",
    "cyan", "blue", "indigo", "purple", "pink", "white", "gray",
]
ALPHAS = [0.3, 0.5, 0.7]


# ── frame counting ─────────────────────────────────────────────────────────────

def count_frames(path: Path) -> int:
    """
    Count frames via full PyAV decode.  Works for both MPEG4-in-AVI and AV1-in-MP4.
    cv2 cannot decode AV1 on Windows and MPEG4 AVI header durations are unreliable.
    """
    count = 0
    with av.open(str(path)) as c:
        for pkt in c.demux(c.streams.video[0]):
            try:
                for _ in pkt.decode():
                    count += 1
            except av.error.InvalidDataError:
                break
    return count


# ── episode helpers ────────────────────────────────────────────────────────────

def get_complete_episodes(ep_df: pd.DataFrame, file_index: int,
                          recolored_frames: int) -> pd.DataFrame:
    """
    Return rows from ep_df whose video lies entirely within recolored_frames.
    Uses a small tolerance (one frame) for floating-point timestamps.
    """
    col_file = f"videos/{VIDEO_KEY}/file_index"
    col_to   = f"videos/{VIDEO_KEY}/to_timestamp"
    tolerance = 1.0 / FPS + 1e-4
    mask = (
        (ep_df[col_file] == file_index) &
        (ep_df[col_to] <= recolored_frames / FPS + tolerance)
    )
    return ep_df[mask].copy()


# ── video transcoding ──────────────────────────────────────────────────────────

def transcode_to_mp4(src: Path, dst: Path, n_frames: int):
    """
    Read the first n_frames from src (MPEG4 AVI) and write dst (AV1 MP4).
    Timestamps are reset to 0-based @ FPS.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    fps_frac = Fraction(FPS).limit_denominator(1000)

    with av.open(str(src)) as in_c, av.open(str(dst), mode="w") as out_c:
        in_v = in_c.streams.video[0]

        out_v            = out_c.add_stream(VCODEC, rate=fps_frac)
        out_v.width      = in_v.codec_context.width
        out_v.height     = in_v.codec_context.height
        out_v.pix_fmt    = PIX_FMT
        out_v.time_base  = Fraction(1, FPS)
        out_c.start_encoding()

        written = 0
        bar = tqdm(total=n_frames, desc=f"  transcode {dst.name}", leave=False)
        for pkt in in_c.demux(in_v):
            if written >= n_frames:
                break
            for frame in pkt.decode():
                if written >= n_frames:
                    break
                new_frame = frame.reformat(
                    width=out_v.width, height=out_v.height, format=PIX_FMT)
                new_frame.pts       = written
                new_frame.time_base = Fraction(1, FPS)
                for out_pkt in out_v.encode(new_frame):
                    out_c.mux(out_pkt)
                written += 1
                bar.update(1)
        bar.close()

        for out_pkt in out_v.encode():   # flush encoder
            out_c.mux(out_pkt)

    return written


# ── dataset builder ────────────────────────────────────────────────────────────

def make_dataset(color: str, alpha: float):
    alpha_str  = f"{alpha:.1f}"
    avi_stem   = f"{color}_a{alpha_str}"
    repo_label = f"{color}_a{alpha_str.replace('.', '')}"
    out_root   = OUTPUT_ROOT / repo_label

    if out_root.exists():
        print(f"[skip] {out_root} already exists.")
        return

    print(f"\n=== {repo_label} ===")

    # load original episode metadata
    ep_parquet = ORIG_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_df = pd.read_parquet(ep_parquet)

    # ── 1. count recolored frames & collect complete episodes ──────────────────
    valid_ep_rows   = []          # rows from ep_df that are fully covered
    file_n_frames   = {}          # (chunk, file) → frame count to encode

    for chunk_idx, file_idx in CHUNK_FILES:
        avi_path = BATCH_ROOT / f"file-{file_idx:03d}" / "recolored" / f"{avi_stem}.avi"
        orig_mp4 = (ORIG_ROOT / "videos" / VIDEO_KEY
                    / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4")

        if not avi_path.exists():
            print(f"  [skip] {avi_path.name} not found")
            continue

        print(f"  counting frames: {avi_path.name} ...", end=" ", flush=True)
        n_recolored = count_frames(avi_path)

        with av.open(str(orig_mp4)) as c:
            n_orig = c.streams.video[0].frames
        coverage = f"({n_recolored}/{n_orig} frames, {100*n_recolored/n_orig:.1f}%)"
        status = "OK full" if n_recolored >= n_orig else "PARTIAL"
        print(f"{coverage}  {status}")

        complete = get_complete_episodes(ep_df, file_idx, n_recolored)
        print(f"    complete episodes: {len(complete)}")

        valid_ep_rows.append(complete)
        file_n_frames[(chunk_idx, file_idx)] = n_recolored

    if not valid_ep_rows:
        print("  [error] no complete episodes found — aborting.")
        return

    valid_ep_df = pd.concat(valid_ep_rows).sort_values("episode_index").reset_index(drop=True)
    old_ep_indices = valid_ep_df["episode_index"].tolist()   # original indices
    old_to_new     = {old: new for new, old in enumerate(old_ep_indices)}
    n_valid        = len(valid_ep_df)
    print(f"  total valid episodes: {n_valid} / {len(ep_df)}")

    out_root.mkdir(parents=True, exist_ok=True)

    # ── 2. transcode each recolored AVI → AV1 MP4 ─────────────────────────────
    for (chunk_idx, file_idx), n_frames in file_n_frames.items():
        avi_path = BATCH_ROOT / f"file-{file_idx:03d}" / "recolored" / f"{avi_stem}.avi"
        mp4_dst  = (out_root / "videos" / VIDEO_KEY
                    / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4")
        written = transcode_to_mp4(avi_path, mp4_dst, n_frames)
        print(f"  video written: {mp4_dst.relative_to(out_root)}  ({written} frames)")

    # ── 3. copy & filter motor data (parquet) ─────────────────────────────────
    data_src = ORIG_ROOT / "data" / "chunk-000" / "file-000.parquet"
    data_dst = out_root  / "data" / "chunk-000" / "file-000.parquet"
    data_dst.parent.mkdir(parents=True, exist_ok=True)

    data_df = pd.read_parquet(data_src)
    data_df = data_df[data_df["episode_index"].isin(set(old_ep_indices))].copy()
    data_df = data_df.reset_index(drop=True)
    # remap episode_index to 0-based
    data_df["episode_index"] = data_df["episode_index"].map(old_to_new)
    # rewrite global frame index
    data_df["index"] = range(len(data_df))
    data_df.to_parquet(data_dst, index=False)
    print(f"  motor data: {len(data_df)} frames")

    # ── 4. build episode metadata parquet ─────────────────────────────────────
    valid_ep_df = valid_ep_df.copy()
    valid_ep_df["episode_index"] = valid_ep_df["episode_index"].map(old_to_new)

    # patch dataset_from_index / dataset_to_index to match the filtered data
    ep_to_from = data_df.groupby("episode_index")["index"].min().to_dict()
    ep_to_to   = (data_df.groupby("episode_index")["index"].max() + 1).to_dict()
    valid_ep_df["dataset_from_index"] = valid_ep_df["episode_index"].map(ep_to_from)
    valid_ep_df["dataset_to_index"]   = valid_ep_df["episode_index"].map(ep_to_to)

    meta_ep_dst = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta_ep_dst.parent.mkdir(parents=True, exist_ok=True)
    valid_ep_df.to_parquet(meta_ep_dst, index=False)

    # ── 5. write info.json ─────────────────────────────────────────────────────
    with open(ORIG_ROOT / "meta" / "info.json") as f:
        info = json.load(f)
    info["total_episodes"] = n_valid
    info["total_frames"]   = len(data_df)
    info["splits"]         = {"train": f"0:{n_valid}"}
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    with open(out_root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # copy tasks + stats unchanged (motor stats are the same across color variants)
    shutil.copy(ORIG_ROOT / "meta" / "tasks.parquet",
                out_root  / "meta" / "tasks.parquet")
    shutil.copy(ORIG_ROOT / "meta" / "stats.json",
                out_root  / "meta" / "stats.json")

    print(f"  DONE  {out_root}")
    print(f"     episodes={n_valid}  frames={len(data_df)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build augmented LeRobot dataset from recolored videos")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--color", choices=COLORS)
    group.add_argument("--all", action="store_true",
                       help="Process all color+alpha combinations")
    parser.add_argument("--alpha", type=float, choices=ALPHAS,
                        help="Required with --color")
    args = parser.parse_args()

    if args.all:
        for color in COLORS:
            for alpha in ALPHAS:
                try:
                    make_dataset(color, alpha)
                except Exception as e:
                    print(f"  [error] {color}_a{alpha:.1f}: {e}", file=sys.stderr)
    else:
        if args.alpha is None:
            parser.error("--alpha is required when using --color")
        make_dataset(args.color, args.alpha)


if __name__ == "__main__":
    main()
