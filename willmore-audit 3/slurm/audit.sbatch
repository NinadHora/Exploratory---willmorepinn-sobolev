#!/bin/bash
#SBATCH --job-name=wa-audit
#SBATCH --partition=l40s          # CPU basta; troque para a partição de CPU do RECOD se houver
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# uso:  sbatch slurm/audit.sbatch checkpoints/run_1/best_model.pt
source slurm/common.sh
CKPT=${1:?caminho do checkpoint relativo a $REPO}
python run_registry.py audit --surface published/checkpoint --repo "$REPO" --ckpt "$REPO/$CKPT" --resolution-study
