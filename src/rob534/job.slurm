#!/bin/bash
#SBATCH --job-name=sam     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of GPUs per node
#SBATCH --constraint=gpu40
#SBATCH --time=3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=kp0374@princeton.edu

module purge
module load anaconda3/2025.6

#conda activate rob534

export HF_HOME="/scratch/gpfs/TSILVER/kp0374/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SAM3_LOCAL_ONLY=1
export SAM3_MODEL_REF="facebook/sam3"

 
#uv run python -m lerobot.async_inference.policy_server   --host=0.0.0.0 --port=8000 --fps=30 --inference_latency=0.033 --obs_queue_timeout=1
uv run sam.py