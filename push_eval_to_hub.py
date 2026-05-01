"""
Push a locally-saved LeRobot eval dataset to HuggingFace Hub.

Usage:
    python push_eval_to_hub.py <local_path> <hub_repo_id>
    python push_eval_to_hub.py C:/Users/calle/.cache/huggingface/lerobot/SkywalkerLi/eval_smolvla-phase-split SkywalkerLi/eval_smolvla-phase-split
    python push_eval_to_hub.py <local_path> <hub_repo_id> --private
    python push_eval_to_hub.py <local_path> <hub_repo_id> --large
"""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LEROBOT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"


def push(local_path: str, hub_repo_id: str, private: bool, large: bool) -> None:
    root = Path(local_path)

    if not root.exists():
        raise FileNotFoundError(f"No local dataset found at {root}")

    print(f"Local path : {root}")
    print(f"Pushing to : {hub_repo_id}  (private={private})")

    dataset = LeRobotDataset(repo_id=hub_repo_id, root=root)
    print(f"  Episodes : {dataset.num_episodes}")
    print(f"  Frames   : {dataset.num_frames}")

    dataset.push_to_hub(private=private, upload_large_folder=large)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push local eval dataset to HuggingFace Hub")
    parser.add_argument("local_path",
                        help="Full local path to the dataset folder "
                             "(e.g. C:/Users/calle/.cache/huggingface/lerobot/SkywalkerLi/eval_smolvla-phase-split_03)")
    parser.add_argument("hub_repo_id",
                        help="HuggingFace destination repo_id "
                             "(e.g. SkywalkerLi/eval_smolvla-phase-split_YELLOW)")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private on HuggingFace Hub")
    parser.add_argument("--large", action="store_true",
                        help="Use upload_large_folder for very big datasets (slower but resumable)")
    args = parser.parse_args()

    push(local_path=args.local_path, hub_repo_id=args.hub_repo_id,
         private=args.private, large=args.large)
