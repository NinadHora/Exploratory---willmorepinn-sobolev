# incluído por todos os jobs
source ~/.bashrc
conda activate willmore-audit
export REPO=${REPO:-$HOME/WillmorePINN-main}
export AUDIT=${AUDIT:-$HOME/willmore-audit}
cd "$AUDIT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short=8 HEAD 2>/dev/null || echo nogit)"
