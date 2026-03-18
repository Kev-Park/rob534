# rob534
ROB534 final project @ Princeton. Designed to run on an SO101 (follower + leader).

## Getting Started on a Cluster
**Note**: It is highly recommended to reconfigure conda to store environments on your `/scratch/` directory rather than in `/home/` as the latter will fill up quickly. You should also set conda to look for environments in the directory that you configure so you can continue to activate environments by name rather than by full path.

1. Create a blank conda environment named `rob534`.
2. With the environment activated run `pip install uv`.
3. If storing the environment in `/scratch/`, run `export UV_CACHE_DIR="/scratch/network/YOUR_PUID/.uv_cache` and `export TMPDIR="/scratch/network/YOUR_PUID/.tmp`. Make sure to create the specified directories.
4. Run `uv sync` to install the dependencies.
5. If storing the VLA weights in `/scratch/`, run `export HF_HOME="/scratch/network/YOUR_PUID/.cache/huggingface`. Make sure to create the specified directory.
6. Install the VLA weights using `uv run huggingface-cli download lerobot/pi05_base`.
7. Calibrate the follower (SO101) using the tutorial [here](https://huggingface.co/docs/lerobot/so101) and validate camera functionality using the tutorial [here](https://huggingface.co/docs/lerobot/cameras).

## Running Async Inference
1. With the environment activated run 
```
uv run python -m lerobot.async_inference.robot_client 
--server_address=localhost:8000 
--robot.type=so101_follower 
--robot.port=COM11 
--robot.id=kevin_follower 
--robot.cameras="{\"camera1\":{\"type\":\"opencv\",\"index_or_path\":0,\"width\":640,\"height\":480,\"fps\":30}}" 
--task="stack the blocks" 
--policy_type=smolvla 
--pretrained_name_or_path=/scratch/network/kp0374/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205 
--actions_per_chunk=5 
--chunk_size_threshold=1 
--policy_device=cuda
```

uv run python -m lerobot.async_inference.policy_server 
      --host=0.0.0.0 
      --port=8080 
      --fps=30 
      --inference_latency=0.033 
      --obs_queue_timeout=1