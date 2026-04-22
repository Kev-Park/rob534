"""
Merge all augmented color-variant datasets + the original into one HuggingFace dataset.

Run AFTER make_augmented_dataset.py --all has finished.

Steps:
  1. Loads original dataset (nc8304/so101_combined)
  2. Loads every built variant from outputs/augmented_datasets/
  3. Merges them with LeRobot's aggregate_datasets
  4. Pushes to HuggingFace as a new repo

Usage:
    python merge_and_push.py
    python merge_and_push.py --repo_id nc8304/so101_color_augmented   # custom name
    python merge_and_push.py --no_original                             # skip original
    python merge_and_push.py --dry_run                                 # local only, no push
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, whoami
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ORIG_REPO_ID    = "nc8304/so101_combined"
DEFAULT_REPO_ID = "nc8304/so101_color_augmented"
AUG_ROOT        = Path(__file__).parent.parent / "outputs" / "augmented_datasets"
MERGE_ROOT      = Path(__file__).parent.parent / "outputs" / "merged_dataset"
ORIG_CACHE      = Path.home() / ".cache" / "huggingface" / "lerobot" / "nc8304" / "so101_combined"


def list_built_variants():
    if not AUG_ROOT.exists():
        return []
    variants = sorted(p for p in AUG_ROOT.iterdir() if p.is_dir()
                      and (p / "meta" / "info.json").exists()
                      and (p / "data").exists()
                      and (p / "videos").exists())
    return variants


def merge(repo_id: str, include_original: bool):
    variants = list_built_variants()
    if not variants:
        print("[error] No augmented variants found in", AUG_ROOT)
        print("        Run: python make_augmented_dataset.py --all")
        return None

    print(f"Found {len(variants)} augmented variant(s):")
    for v in variants:
        info = json.load(open(v / "meta" / "info.json"))
        print(f"  {v.name:25s}  {info['total_episodes']} eps / {info['total_frames']} frames")

    repo_ids = []
    roots    = []

    if include_original:
        repo_ids.append(ORIG_REPO_ID)
        roots.append(ORIG_CACHE)
        orig_info = json.load(open(ORIG_CACHE / "meta" / "info.json"))
        print(f"\n  {'[original]':25s}  {orig_info['total_episodes']} eps / {orig_info['total_frames']} frames")

    for v in variants:
        repo_ids.append(f"local/{v.name}")
        roots.append(v)

    total_eps = sum(
        json.load(open(r / "meta" / "info.json"))["total_episodes"]
        for r in roots
    )
    total_frames = sum(
        json.load(open(r / "meta" / "info.json"))["total_frames"]
        for r in roots
    )
    print(f"\nMerging {len(repo_ids)} dataset(s) -> {total_eps} episodes, {total_frames} frames")
    print(f"Output: {MERGE_ROOT}\n")

    if MERGE_ROOT.exists():
        print(f"[info] Removing existing merge at {MERGE_ROOT}")
        shutil.rmtree(MERGE_ROOT)

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=repo_id,
        roots=roots,
        aggr_root=MERGE_ROOT,
    )

    # verify
    merged_info = json.load(open(MERGE_ROOT / "meta" / "info.json"))
    print(f"\nMerge complete:")
    print(f"  episodes : {merged_info['total_episodes']}")
    print(f"  frames   : {merged_info['total_frames']}")
    return MERGE_ROOT


def push(repo_id: str, merged_root: Path):
    api = HfApi()
    try:
        user = whoami()
        print(f"Pushing as: {user['name']}")
    except Exception as e:
        print(f"[error] Not logged in to HuggingFace: {e}")
        print("        Run: huggingface-cli login")
        return

    # create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repo: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"[warn] create_repo: {e}")

    print(f"Uploading {merged_root} ...")
    api.upload_folder(
        folder_path=str(merged_root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add color-augmented robot episodes",
    )
    print(f"\nDone: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Merge augmented datasets and push to HuggingFace")
    parser.add_argument("--repo_id",     default=DEFAULT_REPO_ID,
                        help=f"HuggingFace dataset repo id (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--no_original", action="store_true",
                        help="Exclude the original orange-color dataset")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Merge locally but do not push to HuggingFace")
    args = parser.parse_args()

    merged_root = merge(args.repo_id, include_original=not args.no_original)

    if merged_root is None:
        return

    if args.dry_run:
        print("\n[dry_run] Skipping HuggingFace push.")
        print(f"Merged dataset at: {merged_root}")
        return

    push(args.repo_id, merged_root)


if __name__ == "__main__":
    main()
