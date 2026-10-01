#!/bin/bash
#SBATCH --job-name=wa-attack
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# uso:  sbatch slurm/attack.sbatch checkpoints/run_1/best_model.pt 2     (2 = até o degrau greedy)
source slurm/common.sh
CKPT=${1:?}; RUNG=${2:-1}
python run_registry.py attack --surface published/checkpoint --repo "$REPO" --ckpt "$REPO/$CKPT" --rung "$RUNG"
