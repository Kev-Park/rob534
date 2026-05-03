"""Split a LeRobot v3 dataset's concatenated MP4 into one clip per episode.

Reads per-episode timestamps from meta/episodes/chunk-XXX/file-XXX.parquet and
runs ffmpeg to cut the camera video into one file per episode.

Usage:
    python split_episodes.py                              # uses the default snapshot below
    python split_episodes.py <snapshot_dir>
    python split_episodes.py <snapshot_dir> --out clips/
    python split_episodes.py <snapshot_dir> --copy        # stream copy (fast; may snap to keyframes)
    python split_episodes.py <snapshot_dir> --camera observation.images.camera1
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

DEFAULT_SNAPSHOT = Path(
    "/Users/skywalkerli/.cache/huggingface/hub/"
    "datasets--nc8304--eval_smolvla-phase-split_combined/"
    "snapshots/8906d6d2336c60806e0c67f9ca7230a3706a7cb9"
)


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # Fall back to the binary bundled with imageio_ffmpeg (already in lerobot's deps)
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_episodes(snapshot_dir: Path) -> pd.DataFrame:
    parts = sorted((snapshot_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No episodes parquet under {snapshot_dir}/meta/episodes")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def split(snapshot_dir: Path, out_dir: Path, camera: str, copy: bool):
    ffmpeg = find_ffmpeg()
    eps = load_episodes(snapshot_dir)

    from_col  = f"videos/{camera}/from_timestamp"
    to_col    = f"videos/{camera}/to_timestamp"
    chunk_col = f"videos/{camera}/chunk_index"
    file_col  = f"videos/{camera}/file_index"
    for col in (from_col, to_col, chunk_col, file_col):
        if col not in eps.columns:
            raise KeyError(f"Missing column {col!r} — check --camera (got {camera!r})")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Splitting {len(eps)} episodes → {out_dir}")

    for _, r in eps.iterrows():
        ep_idx   = int(r["episode_index"])
        from_ts  = float(r[from_col])
        to_ts    = float(r[to_col])
        duration = to_ts - from_ts
        chunk    = int(r[chunk_col])
        file_idx = int(r[file_col])

        src = snapshot_dir / f"videos/{camera}/chunk-{chunk:03d}/file-{file_idx:03d}.mp4"
        dst = out_dir / f"episode_{ep_idx:04d}.mp4"

        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-ss", f"{from_ts:.6f}", "-i", str(src),
               "-t", f"{duration:.6f}"]
        if copy:
            cmd += ["-c", "copy"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an"]
        cmd += [str(dst)]

        print(f"  ep {ep_idx:>3}  {from_ts:7.2f}s → {to_ts:7.2f}s  ({duration:5.2f}s)  →  {dst.name}")
        subprocess.run(cmd, check=True)

    print(f"\nDone. {len(eps)} clips in {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snapshot_dir", nargs="?", default=str(DEFAULT_SNAPSHOT),
                   help="Path to the dataset snapshot (default: built-in)")
    p.add_argument("--out", default="episodes_split", help="Output directory (default: episodes_split/)")
    p.add_argument("--camera", default="observation.images.camera1",
                   help="Video feature key (default: observation.images.camera1)")
    p.add_argument("--copy", action="store_true",
                   help="Use stream copy instead of re-encoding (faster, may snap cuts to keyframes)")
    a = p.parse_args()
    split(Path(a.snapshot_dir), Path(a.out), a.camera, a.copy)


if __name__ == "__main__":
    main()
