"""
basins/run_local.py: versão reduzida do mapa de bacias para uma noite de Mac.

Roda em sequência (sem SLURM) uma família pequena de dados iniciais pelo fluxo
publicado, audita cada saída e grava em runs/. Default: 6 tori de Fourier
(amplitudes 0.05, 0.1, 0.2, duas sementes cada), 2 τ torcidos e o nó (2,3).
Cada treino de gênero 1 leva minutos; o total fica em torno de uma hora.

    python basins/run_local.py --repo ~/Downloads/WillmorePINN-main
    python run_registry.py basins          # depois, monta a Figura 2
"""
import argparse, os, sys, time

import torch

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
import run_registry                      # noqa: E402
from methods import published            # noqa: E402
from basins import perturb_init          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--n-fourier", type=int, default=6)
    ap.add_argument("--taus", default="0.2j,0.05+0.1j")
    ap.add_argument("--no-knot", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override de épocas (default: o do config)")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    items = perturb_init.initial_family(n_fourier=a.n_fourier, taus=tuple(a.taus.split(",")), knot=not a.no_knot)
    ov_epochs = {"training": {"num_epochs": a.epochs}} if a.epochs else {}
    for k, it in enumerate(items):
        t0 = time.time()
        print(f"\n===== [{k + 1}/{len(items)}] dado inicial: {it['name']}")
        try:
            if it["kind"] == "tau_override":
                ov = {"topology": {"torus": {"tau": it["tau"]}}}; ov.update(ov_epochs)
                s = published.train(a.repo, 1, ov, tag=it["name"])
            else:
                init_cert = it["surface"].audit(grid=96, residual=False, seeds=2, verbose=False)
                run_registry.save("audit", "init_" + it["name"], init_cert)
                orig = perturb_init.install_reference_hook(a.repo, it["surface"].phi)
                try:
                    s = published.train(a.repo, 1, dict(ov_epochs), tag=it["name"])
                finally:
                    import sampling; sampling.get_reference_embedding = orig
            cert = s.audit(resolution_study=True, verbose=True)
            cert["initial_data"] = it["name"]
            run_registry.save("audit", s.name, cert)
        except Exception as e:                       # um dado inicial que falha não derruba a noite
            print(f"FALHOU {it['name']}: {e!r}")
        print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
