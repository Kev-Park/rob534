from datasets import load_dataset
from huggingface_hub import list_datasets, delete_repo

What is the
input
username = "nc8304"

import subprocess
subprocess.run([
    "lerobot-edit-dataset",
    "--new_repo_id=nc8304/so101_combined_cubeONLY",
    "--operation.type=merge",
    "--operation.repo_ids=[\"nc8304/so101_032326_white_back_041026_cube_10\", \"nc8304/so101_032326_white_back_040126_cube_13\"]",
    "--push_to_hub=true",
])