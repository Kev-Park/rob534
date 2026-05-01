"""
Merge multiple local LeRobot datasets into one, then optionally push to HuggingFace.

Usage:
    # Merge two local eval runs into one folder
    python merge_local.py PATH1 PATH2 --output outputs/merged_eval --repo_id SkywalkerLi/eval_smolvla-combined

    # Merge without pushing (inspect first)
    python merge_local.py PATH1 PATH2 --output outputs/merged_eval --dry_run

    # Use lerobot cache paths directly
    python merge_local.py \\
        C:/Users/calle/.cache/huggingface/lerobot/SkywalkerLi/eval_smolvla-phase-split_01 \\
        C:/Users/calle/.cache/huggingface/lerobot/SkywalkerLi/eval_smolvla-phase-split_02 \\
        --output outputs/merged_eval \\
        --repo_id SkywalkerLi/eval_smolvla-phase-split_combined
"""

import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def merge(input_paths: list[Path], output_path: Path, repo_id: str) -> Path:
    # Validate inputs
    for p in input_paths:
        info = p / "meta" / "info.json"
        if not info.exists():
            raise FileNotFoundError(f"Not a valid LeRobot dataset (no meta/info.json): {p}")

    print(f"Merging {len(input_paths)} dataset(s):")
    total_eps, total_frames = 0, 0
    for p in input_paths:
        info = json.loads((p / "meta" / "info.json").read_text())
        eps, frames = info["total_episodes"], info["total_frames"]
        total_eps += eps
        total_frames += frames
        print(f"  {p.name:40s}  {eps} eps / {frames} frames")
    print(f"\n  -> {total_eps} episodes, {total_frames} frames total")
    print(f"  -> Output: {output_path}\n")

    if output_path.exists():
        print(f"Removing existing output at {output_path}")
        shutil.rmtree(output_path)

    # aggregate_datasets needs a fake repo_id per input — we use the folder name
    aggregate_datasets(
        repo_ids=[f"local/{p.name}" for p in input_paths],
        roots=input_paths,
        aggr_repo_id=repo_id,
        aggr_root=output_path,
    )

    merged = json.loads((output_path / "meta" / "info.json").read_text())
    print(f"Merge complete: {merged['total_episodes']} eps / {merged['total_frames']} frames")
    return output_path


def push(output_path: Path, repo_id: str, private: bool):
    from huggingface_hub import HfApi, whoami
    try:
        print(f"Pushing as: {whoami()['name']}")
    except Exception as e:
        raise RuntimeError("Not logged in to HuggingFace. Run: huggingface-cli login") from e

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    print(f"Uploading {output_path} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(output_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Merge local eval datasets",
    )
    print(f"Done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge local LeRobot datasets and optionally push to HF")
    parser.add_argument("inputs", nargs="+",
                        help="Paths to local LeRobot dataset folders to merge")
    parser.add_argument("--output", required=True,
                        help="Local output folder for the merged dataset")
    parser.add_argument("--repo_id", default=None,
                        help="HuggingFace repo_id to push to (required unless --dry_run)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Merge locally only, do not push to HuggingFace")
    parser.add_argument("--private", action="store_true",
                        help="Make the HuggingFace dataset private")
    args = parser.parse_args()

    if not args.dry_run and not args.repo_id:
        parser.error("--repo_id is required unless --dry_run is set")

    input_paths = [Path(p) for p in args.inputs]
    output_path = Path(args.output)
    repo_id = args.repo_id or f"local/{output_path.name}"

    merged = merge(input_paths, output_path, repo_id)

    if args.dry_run:
        print(f"\n[dry_run] Skipping push. Merged dataset at: {merged}")
    else:
        push(merged, repo_id, private=args.private)
