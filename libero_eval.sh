#!/bin/bash

# Set pretrained policy path and other configuration variables
POLICY_PATH="/scratch/gpfs/TSILVER/kp0374/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30"
OUTPUT_DIR="./outputs/"
DATA_DIR="./analyzer_data/"
TASK_IDS="[0]"
N_ACTION_STEPS=10
RENAME_MAP='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.right_wrist_0_rgb"}'

# Export environment variable for LEROBOT_DATA_DIR
export LEROBOT_DATA_DIR="$DATA_DIR"
export TORCH_CUDAGRAPHS_DISABLE="1"

if [ -z "$POLICY_PATH" ]; then
    echo "Error: POLICY_PATH is empty. Set it to your pretrained policy directory (contains config.json and model.safetensors)."
    exit 1
fi


# Directly run the python command
echo "Running evaluation..."
python lerobot/src/lerobot/scripts/lerobot_eval.py \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids="$TASK_IDS" \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --policy.path="$POLICY_PATH" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --rename_map="$RENAME_MAP" \
    --output_dir="$OUTPUT_DIR" \
    --env.max_parallel_tasks=1

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
    exit 0
else
    echo "Error running evaluation with exit code: $?"
    exit 1
fi