#!/bin/bash
# Uma vez, no headnode do RECOD (sem GPU):
#   bash slurm/env_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
conda env create -f environment.yml || conda env update -f environment.yml
conda activate willmore-audit
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# o fluxo publicado e o pré-condicionador ficam ao lado deste repo:
#   ~/WillmorePINN-main            (código do paper, com checkpoints/run_N)
#   ~/willmorepinn-sobolev         (sobolev_optim.py + patches/)
python -m pytest tests -q          # tem de ficar verde antes de qualquer job
