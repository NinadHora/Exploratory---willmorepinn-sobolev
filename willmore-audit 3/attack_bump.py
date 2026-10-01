"""
attack_bump.py: ataque da bolha enxertada + Figura 1 (energia certificada × reportada).

Pega uma superfície base (toro de Clifford exato, ou um checkpoint do WillmorePINN),
enxerta bolhas φ_ε = φ + ε·exp(−d²/ρ²)·n numa grade de (ε, ρ), e para cada uma mede:

    W_cert   quadratura espectral, grade refinada até Gauss-Bonnet fechar (certificado)
    W_rep    estimador de treino (Monte Carlo uniforme, n=5000), média ± dp sobre sementes
    W_Huber  o mesmo com o Huber h2_clip=50 do treino (o que o otimizador vê)
    reward   W_cert − W_Huber (seção 12 do protocolo)
    success  grade convergida, topologia intacta (C1) e reward > 2π

Saída: attack_results.csv e fig1_certified_vs_reported.png

    python attack_bump.py                                   # base: Clifford exato
    python attack_bump.py --repo ~/WillmorePINN-main --ckpt .../best_model.pt
"""
import argparse, csv, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certificates as C

PI = math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--eps", default="0.05,0.1,0.2,0.3")
    ap.add_argument("--rho", default="0.5,0.3,0.2,0.15")
    ap.add_argument("--uv0", default=None, help="centro da bolha 'u,v' (default: equador externo do toro, (π,0))")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--grids", default="96,192,384")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    if a.ckpt:
        from audit_checkpoint import load_model
        base, genus, _ = load_model(a.repo, a.ckpt)
        label = os.path.basename(os.path.dirname(a.ckpt))
    else:
        base, genus, label = C.exact_torus(), 1, "clifford"
    domain = C.domain_for_genus(genus)
    uv0 = tuple(map(float, a.uv0.split(","))) if a.uv0 else ((PI, 0.0) if genus == 1 else (PI, PI / 2))
    grids = tuple(int(g) for g in a.grids.split(","))

    rows = []
    W0 = C.willmore(base, domain, grids[0], grids[0])
    print(f"base {label}: W = {W0:.6f}")
    for eps in map(float, a.eps.split(",")):
        for rho in map(float, a.rho.split(",")):
            phi = C.graft_bump(base, domain, uv0, eps, rho)
            rs = C.resolution_study(phi, domain, grids=grids, verbose=False)
            est = C.estimator_report(phi, domain, rs["W"], seeds=a.seeds)
            gb = rs["rows"][-1]["gb_residual"]
            reward = rs["W"] - est["W_huber_mean"]          # seção 12: recompensa = certificada − Huber
            success = rs["converged"] and gb < C.TOL_GB and reward > 2 * PI   # meio quantum, topologia intacta
            row = dict(eps=eps, rho=rho, W_cert=rs["W"], grid=rs["n"], converged=rs["converged"], gb_residual=gb,
                       W_mc=est["W_rep_mean"], W_mc_sd=est["W_rep_sd"], W_huber=est["W_huber_mean"], W_huber_sd=est["W_huber_sd"],
                       reward=reward, success=success)
            rows.append(row)
            print(f"ε={eps:.2f} ρ={rho:.2f}  W_cert={row['W_cert']:.4f} (grid {rs['n']}, GB={gb:.1e}, conv={rs['converged']})  "
                  f"W_rep={row['W_mc']:.3f}±{row['W_mc_sd']:.3f}  W_Huber={row['W_huber']:.3f}±{row['W_huber_sd']:.3f}  "
                  f"reward={reward:+.3f}  success={success}")

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "attack_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 5))
        xs = [r["W_cert"] for r in rows]
        ax.errorbar(xs, [r["W_mc"] for r in rows], yerr=[r["W_mc_sd"] for r in rows], fmt="o", ms=4, capsize=2, label="MC uniforme (n=5000)")
        ax.errorbar(xs, [r["W_huber"] for r in rows], yerr=[r["W_huber_sd"] for r in rows], fmt="s", ms=4, capsize=2, label="MC + Huber (loss de treino)")
        lo, hi = min(xs + [W0]) - 0.5, max(xs) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="reportada = certificada")
        ax.axvline(2 * PI ** 2 if genus == 1 else 4 * PI, color="gray", lw=0.8, ls=":", label="mínimo conhecido")
        for r in rows:
            if not r["converged"]:
                ax.annotate("grade não convergiu", (r["W_cert"], r["W_mc"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("W certificada (quadratura, Gauss-Bonnet fechado)")
        ax.set_ylabel("W reportada (estimador de treino)")
        ax.set_title(f"bolhas enxertadas em {label}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "fig1_certified_vs_reported.png"), dpi=160)
        print("figura salva: fig1_certified_vs_reported.png")
    except Exception as e:  # matplotlib opcional
        print("figura não gerada:", e)


if __name__ == "__main__":
    main()
