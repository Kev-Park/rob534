# rob534
ROB534 final project @ Princeton.

## Getting Started on a Cluster
**Note**: It is highly recommended to reconfigure conda to store environments on your `/scratch/` directory rather than in `/home/` as the latter will fill up quickly. You should also set conda to look for environments in the directory that you configure so you can continue to activate environments by name rather than by full path.

1. Create a blank conda environment named `rob534`.
2. With the environment activated run `pip install uv`.
3. If storing the environment in `/scratch/`, run `export UV_CACHE_DIR="/scratch/network/YOUR_PUID/.uv_cache` and `export TMPDIR="/scratch/network/YOUR_PUID/.tmp`. Make sure to create the specified directories.
4. Run `uv sync` to install the dependencies.
5. If storing the VLA weights in `/scratch/`, run `export HF_HOME="/scratch/network/YOUR_PUID/.cache/huggingface`. Make sure to create the specified directory.
6. Install the VLA weights using `uv run huggingface-cli download lerobot/pi05_base`.

## Running Inference
1. 