#!/bin/bash
#SBATCH --job-name=wa-arm
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-4               # 5 sementes por braço
#SBATCH --output=slurm/logs/%x-%A_%a.out
# uso:  sbatch slurm/train_arm.sbatch linear_spectral genus=1 K=6 epochs=2000
#       sbatch slurm/train_arm.sbatch optimiser/adam repo=$HOME/WillmorePINN-main genus=1
source slurm/common.sh
METHOD=${1:?nome no REGISTRY}; shift
python run_registry.py train --method "$METHOD" --kw "$@" seed=$SLURM_ARRAY_TASK_ID --resolution-study
