"""
Push a locally-saved LeRobot eval dataset to HuggingFace Hub.

The hub repo_id is derived automatically from the policy path:
    SkywalkerLi/smolvla-phase-split  ->  SkywalkerLi/smolvla-phase-split_eval
    <local snapshot path>             ->  same, parsed from the models-- folder name

Usage:
    python push_eval_to_hub.py --policy SkywalkerLi/smolvla-phase-split
    python push_eval_to_hub.py --policy SkywalkerLi/smolvla-phase-split --local "C:/Users/calle/.cache/..."
    python push_eval_to_hub.py --policy SkywalkerLi/smolvla-phase-split --hub SkywalkerLi/my_custom_name
    python push_eval_to_hub.py --policy SkywalkerLi/smolvla-phase-split --private
"""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LEROBOT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"
DEFAULT_LOCAL_REPO = "SkywalkerLi/eval_smolvla_cube"


def hub_id_from_policy(policy_path: str) -> str:
    """Derive a HuggingFace repo_id from a policy path.

    Handles both a raw HF model ID and a local snapshot path:
        "SkywalkerLi/smolvla-phase-split"
            -> "SkywalkerLi/smolvla-phase-split_eval"
        "C:/.../models--SkywalkerLi--smolvla-phase-split/snapshots/abc123"
            -> "SkywalkerLi/smolvla-phase-split_eval"
    """
    for part in Path(policy_path).parts:
        if part.startswith("models--"):
            _, namespace, model_name = part.split("--", 2)
            return f"{namespace}/{model_name}_eval"
    # Already an HF model ID
    if "/" in policy_path:
        namespace, model_name = policy_path.split("/", 1)
        return f"{namespace}/{model_name}_eval"
    return f"{policy_path}_eval"


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
    parser.add_argument("--policy", required=False, default=None,
                        help="Policy path or HF model ID used during eval "
                             "(e.g. SkywalkerLi/smolvla-phase-split or a local snapshot path). "
                             "Used to derive the hub repo name automatically.")
    parser.add_argument("--local", default=str(LEROBOT_CACHE / DEFAULT_LOCAL_REPO),
                        help="Full local path to the dataset folder "
                             f"(default: lerobot cache / {DEFAULT_LOCAL_REPO})")
    parser.add_argument("--hub", default=None,
                        help="Override the HF destination repo_id. "
                             "If omitted, derived from --policy as <name>_eval.")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private on HuggingFace Hub")
    parser.add_argument("--large", action="store_true",
                        help="Use upload_large_folder for very big datasets (slower but resumable)")
    args = parser.parse_args()

    if args.hub:
        hub_repo_id = args.hub
    elif args.policy:
        hub_repo_id = hub_id_from_policy(args.policy)
    else:
        parser.error("provide --hub or --policy")
    push(local_path=args.local, hub_repo_id=hub_repo_id, private=args.private, large=args.large)
