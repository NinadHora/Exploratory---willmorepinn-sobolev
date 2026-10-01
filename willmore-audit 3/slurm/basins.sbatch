#!/bin/bash
#SBATCH --job-name=wa-basins
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-35              # um dado inicial por tarefa (30 Fourier + 5 τ + 1 nó)
#SBATCH --output=slurm/logs/%x-%A_%a.out
# uso:  sbatch slurm/basins.sbatch          depois:  python run_registry.py basins
source slurm/common.sh
python - <<PY
import sys, torch; torch.set_default_dtype(torch.float64)
sys.path.insert(0, "."); import methods, run_registry
from methods import published
from basins import perturb_init
items = perturb_init.initial_family()
it = items[int("$SLURM_ARRAY_TASK_ID")]
print("initial data:", it["name"])
if it["kind"] == "tau_override":
    s = published.train("$REPO", 1, {"topology": {"torus": {"tau": it["tau"]}}}, tag=it["name"])
else:
    run_registry.save("audit", "init_" + it["name"], it["surface"].audit(verbose=False))   # energia do dado inicial
    orig = perturb_init.install_reference_hook("$REPO", it["surface"].phi)
    s = published.train("$REPO", 1, {}, tag=it["name"])
cert = s.audit(resolution_study=True); cert["initial_data"] = it["name"]
run_registry.save("audit", s.name, cert)
PY
