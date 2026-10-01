"""
audit_checkpoint.py: roda todos os certificados num checkpoint do WillmorePINN.

    python audit_checkpoint.py --repo ~/WillmorePINN-main \
        --ckpt ~/WillmorePINN-main/checkpoints/run_1/best_model.pt \
        --resolution-study --out audit_run1.json

O checkpoint carrega o próprio config, então o modelo é reconstruído sem yaml.
Gênero 0 e 1 (carta única). Sem --ckpt, audita o toro de Clifford exato.
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certificates as C


def load_model(repo: str, ckpt_path: str):
    sys.path.insert(0, repo)
    from model import create_embedding_model            # do WillmorePINN
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    genus = cfg.get("topology", {}).get("genus", 1)
    if genus == 2:
        raise NotImplementedError("gênero 2 (duas cartas) ainda não suportado")
    model = create_embedding_model(cfg, torch.device("cpu"), skip_init=True)
    model.load_state_dict(ck.get("model", ck.get("model_state_dict")))
    model = model.double().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, genus, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--residual-grid", type=int, default=32)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--variational", action="store_true", help="também roda o fallback do C4 (só 2as derivadas)")
    ap.add_argument("--resolution-study", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    t0 = time.time()

    if a.ckpt:
        phi, genus, ck = load_model(a.repo, a.ckpt)
        label = os.path.basename(os.path.dirname(a.ckpt))
        print(f"checkpoint {a.ckpt}  (epoch {ck.get('epoch')}, saved loss {ck.get('loss')})  genus {genus}")
    else:
        phi, genus, ck, label = C.exact_torus(), 1, {}, "clifford_exact"
        print("no --ckpt: auditing the exact Clifford torus")
    domain = C.domain_for_genus(genus)

    rs = None
    grid = a.grid
    if a.resolution_study:
        print("== resolution study")
        rs = C.resolution_study(phi, domain)
        grid = rs["n"]
        if not rs["converged"]:
            print("  NOT CONVERGED at the largest grid: values below are bounds, not passes")

    print(f"== certificates (grid {grid}²)")
    cert = C.certify(phi, genus, nu=grid, nv=grid, residual=not a.no_residual,
                     residual_grid=(a.residual_grid, a.residual_grid))
    if a.variational:
        var = C.willmore_residual_variational(phi, domain, grid, grid)
        print(f"C4 fallback (first variation): ||r||_L2 lower bound = {var['residual_L2_lower_bound']:.3e}")
        cert["C4_variational"] = var

    print(f"== training estimator vs certified ({a.seeds} seeds, n = 5000, Huber h2_clip = 50)")
    est = C.estimator_report(phi, domain, cert["W"], seeds=a.seeds)
    print(f"  W_cert            = {cert['W']:.6f}")
    print(f"  W_rep (uncapped)  = {est['W_rep_mean']:.6f} ± {est['W_rep_sd']:.6f}   bias {est['bias']:+.6f}   significant: {est['bias_significant']}")
    print(f"  W_rep (Huber)     = {est['W_huber_mean']:.6f} ± {est['W_huber_sd']:.6f}   Huber active: {est['huber_active']}")
    if a.ckpt:
        print(f"  loss saved in checkpoint = {ck.get('loss')}")
    cert["estimator"] = est
    cert["resolution_study"] = rs
    cert["label"] = label
    cert["ckpt"] = a.ckpt
    cert["seconds"] = time.time() - t0

    if a.out:
        with open(a.out, "w") as f:
            json.dump(cert, f, indent=2, default=float)
        print(f"saved {a.out}")
    print(f"time: {cert['seconds']:.1f}s")


if __name__ == "__main__":
    main()
