"""
HuggingFace data pipeline utilities.

Operations:
  1. combine   -- merge source datasets idata from huggingface nto nc8304/so101_combined_cubeONLY
  2. augment   -- recolor + build augmented variants + push to nc8304/so101_color_augmented
"""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# ── config ──────────────────────────────────────────────────────────────────
ORIG_REPO_ID   = "nc8304/so101_combined_cubeONLY"
AUG_REPO_ID    = "nc8304/so101_color_augmented"
VIDEO_KEY      = "observation.images.front"
SCRIPT_DIR     = Path(__file__).parent
OUTPUT_ROOT    = SCRIPT_DIR.parent / "outputs"


# ── 1. combine ───────────────────────────────────────────────────────────────
def combine():
    """Merge source datasets into so101_combined_cubeONLY on HuggingFace."""
    subprocess.run([
        "lerobot-edit-dataset",
        f"--new_repo_id={ORIG_REPO_ID}",
        "--operation.type=merge",
        "--operation.repo_ids=[\"nc8304/so101_032326_white_back_041026_cube_10\", \"nc8304/so101_032326_white_back_040126_cube_13\"]",
        "--push_to_hub=true",
    ], check=True)


# ── completion checks ────────────────────────────────────────────────────────
EXPECTED_VARIANTS = 13 * 3  # 13 colors × 3 alphas = 39


def recolor_done(videos: list) -> bool:
    """True if every video stem has all 39 recolored AVIs (>10 MB each)."""
    for v in videos:
        recolor_dir = OUTPUT_ROOT / "batch" / Path(v).stem / "recolored"
        avis = [f for f in recolor_dir.glob("*.avi") if f.stat().st_size > 10_000_000] \
               if recolor_dir.exists() else []
        if len(avis) < EXPECTED_VARIANTS:
            return False
    return True


def augmented_done() -> bool:
    """True if all 39 augmented variant datasets exist with meta/info.json."""
    aug_root = OUTPUT_ROOT / "augmented_datasets"
    if not aug_root.exists():
        return False
    valid = [d for d in aug_root.iterdir()
             if d.is_dir() and (d / "meta" / "info.json").exists()]
    return len(valid) >= EXPECTED_VARIANTS


# ── 2. augment + push ────────────────────────────────────────────────────────
def augment():
    """
    Full pipeline:
      a) batch_recolor  -- segment orange cube and produce recolored AVIs
      b) make_augmented_dataset --all  -- transcode AVIs into LeRobot datasets
      c) merge_and_push  -- merge all variants + original, push to HF
    """
    # Resolve latest snapshot of the source dataset from local cache
    print(f"Resolving local snapshot for {ORIG_REPO_ID} ...")
    snap_root = Path(snapshot_download(repo_id=ORIG_REPO_ID, repo_type="dataset", local_files_only=True))
    vid_dir   = snap_root / "videos" / VIDEO_KEY / "chunk-000"

    videos = sorted(vid_dir.glob("*.mp4"))
    if not videos:
        print(f"[error] No videos found at {vid_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {len(videos)} video(s): {[v.name for v in videos]}")

    # a) batch recolor
    if recolor_done(videos):
        print("\n[1/3] Recolored AVIs already complete — skipping batch_recolor.")
    else:
        print(f"\n[1/3] Running batch_recolor on {len(videos)} video(s) ...")
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "batch_recolor.py"),
             "--videos", *[str(v) for v in videos],
             "--output_root", str(OUTPUT_ROOT / "batch")],
            check=True,
            cwd=str(SCRIPT_DIR),
        )

    # b) build augmented datasets
    if augmented_done():
        print("\n[2/3] Augmented datasets already complete — skipping make_augmented_dataset.")
    else:
        print("\n[2/3] Building augmented datasets ...")
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "make_augmented_dataset.py"), "--all"],
            check=True,
            cwd=str(SCRIPT_DIR),
        )

    # c) merge + push
    print("\n[3/3] Merging and pushing to HuggingFace ...")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "merge_and_push.py"),
         "--repo_id", AUG_REPO_ID],
        check=True,
        cwd=str(SCRIPT_DIR),
    )

    print(f"\nDone: https://huggingface.co/datasets/{AUG_REPO_ID}")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HuggingFace data pipeline")
    parser.add_argument("operation", choices=["combine", "augment"], nargs="?", default="augment",
                        help="combine: merge source datasets | augment: recolor + push (default)")
    args = parser.parse_args()

    if args.operation == "combine":
        combine()
    elif args.operation == "augment":
        augment()
