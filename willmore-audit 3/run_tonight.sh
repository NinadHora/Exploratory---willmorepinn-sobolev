#!/bin/bash
# Roda em sequência, no Mac, tudo que cabe numa noite. Cada etapa grava em runs/ e em logs/;
# uma etapa que falha não derruba as seguintes. De manhã: python make_table.py
#
#   source ~/Downloads/willmore-audit/.venv/bin/activate
#   cd ~/Downloads/willmore-audit2
#   bash run_tonight.sh ~/Downloads/WillmorePINN-main
REPO=${1:?caminho do WillmorePINN-main}
mkdir -p logs runs
run() { echo; echo "===== $1"; date; shift; "$@" 2>&1 | tee -a "logs/$(date +%H%M)_$1.log"; echo "(exit $?)"; }

# 1. os dois checkpoints do fluxo publicado que já existem, pelo registro (mesma linha da tabela)
run audit_run1 python run_registry.py audit --surface published/checkpoint --repo "$REPO" --ckpt "$REPO/checkpoints/run_1/best_model.pt" --resolution-study
run audit_run2 python run_registry.py audit --surface published/checkpoint --repo "$REPO" --ckpt "$REPO/checkpoints/run_2/best_model.pt" --resolution-study

# 2. ansatz linear espectral (Lecture 3): minutos
run linear_spectral python run_registry.py train --method linear_spectral --kw genus=1 K=6 epochs=1500 seed=0 --resolution-study

# 3. PINN de resíduo (Lectures 12, 13): minutos a dezenas de minutos
run pinn_residual python run_registry.py train --method pinn_residual --kw genus=1 K=4 epochs=300 seed=0 lr=1e-4 --resolution-study

# 4. o fluxo publicado com mais duas sementes (dispersão entre runs, Lecture 2)
run adam_seed1 python run_registry.py train --method optimiser/adam --kw repo="$REPO" genus=1 seed=1 --resolution-study
run adam_seed2 python run_registry.py train --method optimiser/adam --kw repo="$REPO" genus=1 seed=2 --resolution-study

# 5. bacias reduzidas (6 Fourier + 2 τ + nó): cerca de uma hora
run basins python basins/run_local.py --repo "$REPO"
run basins_map python run_registry.py basins

# 6. tabela
run table python make_table.py
echo; echo "FIM"; date
