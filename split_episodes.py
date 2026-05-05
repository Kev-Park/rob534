"""
Download eval datasets from the Hub and split their concatenated camera video
into one mp4 per episode for manual success/failure review.

Usage:
    python split_episodes.py
    python split_episodes.py --datasets nc8304/eval_smolvla-aug
    python split_episodes.py --output-dir my_episodes --cam observation.images.camera1
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pandas as pd
from huggingface_hub import snapshot_download


DEFAULT_DATASETS = [
    "nc8304/eval_smolvla-aug",
    "nc8304/eval_smolvla-phase-split-new-prompts",
    "nc8304/eval_smolvla-phase-split_combined",
]
DEFAULT_CAM = "observation.images.camera1"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def split_dataset(dataset_id: str, output_root: Path, cam_key: str) -> int:
    out_dir = output_root / dataset_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {dataset_id} ===")
    print(f"  downloading (cached on rerun)...")
    repo_dir = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))

    meta_files = sorted((repo_dir / "meta" / "episodes").glob("**/*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No episodes parquet found under {repo_dir}/meta/episodes/")
    episodes_df = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

    chunk_col = f"videos/{cam_key}/chunk_index"
    file_col  = f"videos/{cam_key}/file_index"
    from_col  = f"videos/{cam_key}/from_timestamp"
    to_col    = f"videos/{cam_key}/to_timestamp"
    for c in (chunk_col, file_col, from_col, to_col):
        if c not in episodes_df.columns:
            raise KeyError(f"Column {c!r} missing in episodes parquet for {dataset_id}")

    n_done = 0
    n_skipped = 0
    for _, ep in episodes_df.iterrows():
        ep_idx  = int(ep["episode_index"])
        chunk   = int(ep[chunk_col])
        file_i  = int(ep[file_col])
        from_ts = float(ep[from_col])
        to_ts   = float(ep[to_col])

        src = repo_dir / "videos" / cam_key / f"chunk-{chunk:03d}" / f"file-{file_i:03d}.mp4"
        if not src.exists():
            print(f"  ep {ep_idx:3d}: source video missing ({src.name}); skipping")
            n_skipped += 1
            continue

        out = out_dir / f"episode_{ep_idx:04d}.mp4"
        if out.exists():
            n_done += 1
            continue

        duration = to_ts - from_ts
        # -ss before -i for fast seek, then re-encode so the cut is frame-accurate
        cmd = [
            FFMPEG, "-y",
            "-ss", f"{from_ts}",
            "-i", str(src),
            "-t", f"{duration}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an", str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ep {ep_idx:3d}: ffmpeg failed -> {result.stderr.strip().splitlines()[-1]}")
            n_skipped += 1
            continue

        n_done += 1
        if n_done % 10 == 0 or n_done == 1:
            print(f"  ep {ep_idx:3d}: {out.name}  ({duration:.2f}s)")

    print(f"  -> {n_done} episodes written to {out_dir}/  (skipped {n_skipped})")
    return n_done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="HuggingFace dataset IDs to split")
    parser.add_argument("--output-dir", default="episode_videos",
                        help="Folder where per-episode mp4s are written")
    parser.add_argument("--cam", default=DEFAULT_CAM,
                        help="Camera key inside the dataset (default: observation.images.camera1)")
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for ds in args.datasets:
        total += split_dataset(ds, output_root, args.cam)
    print(f"\nAll done — {total} episode clips total under {output_root}/")


if __name__ == "__main__":
    main()
