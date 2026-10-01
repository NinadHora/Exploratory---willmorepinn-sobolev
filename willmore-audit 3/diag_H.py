"""
diag_H.py: de onde vem um resíduo de Willmore grande?

Avalia H numa grade uniforme, separa a energia espectral de H em modos baixos
(|k| <= kcut) e altos, e salva uma imagem de H(u, v). Ondulação de alta
frequência em H contribui a·k⁴ ao resíduo e quase nada a W: se a fração de
energia em modos altos for grande, o resíduo é da superfície; se H for liso
e o resíduo direto continuar grande, é ruído da quarta derivada por autodiff,
e vale o fallback variacional.

    python diag_H.py --repo ~/Downloads/WillmorePINN-main --ckpt .../best_model.pt
    python diag_H.py                     # toro de Clifford exato (H constante: só o modo zero)
"""
import argparse, math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certificates as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--kcut", type=int, default=6, help="frequência de corte entre baixo e alto (as features de Fourier da rede vão até 6)")
    ap.add_argument("--out", default="diag_H.png")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    if a.ckpt:
        from audit_checkpoint import load_model
        phi, genus, _ = load_model(a.repo, a.ckpt)
    else:
        phi, genus = C.exact_torus(), 1
    domain = C.domain_for_genus(genus)

    uv, _ = C.quadrature_grid(domain, a.n, a.n)
    geo = C.geometry(phi, uv)
    H = geo["H"].reshape(a.n, a.n).numpy()
    K = geo["K"].reshape(a.n, a.n).numpy()

    print(f"H: min {H.min():+.4f}  max {H.max():+.4f}  mean {H.mean():+.4f}  std {H.std():.4f}")
    print(f"K: min {K.min():+.4f}  max {K.max():+.4f}")

    if domain == "torus":       # ambos periódicos: FFT 2D direta
        F = np.fft.fft2(H - H.mean())
        P = np.abs(F) ** 2
        ku = np.fft.fftfreq(a.n, 1.0 / a.n); kv = np.fft.fftfreq(a.n, 1.0 / a.n)
        KU, KV = np.meshgrid(ku, kv, indexing="ij")
        kmax = np.maximum(np.abs(KU), np.abs(KV))
        low, high = P[kmax <= a.kcut].sum(), P[kmax > a.kcut].sum()
        print(f"energia espectral de H (sem a média): modos |k|<={a.kcut}: {low / (low + high):.4%}   modos |k|>{a.kcut}: {high / (low + high):.4%}")
        # estimativa grosseira do que os modos altos custam no resíduo: amplitude × k⁴, somado
        amp = np.sqrt(P) / (a.n * a.n)
        print(f"soma de amplitude·k⁴ nos modos altos: {(amp[kmax > a.kcut] * kmax[kmax > a.kcut] ** 4).sum():.3e}   (Clifford exato: 0)")
        # modo mais forte acima do corte
        idx = np.unravel_index(np.argmax(np.where(kmax > a.kcut, P, 0)), P.shape)
        print(f"modo alto mais forte: (ku, kv) = ({int(KU[idx])}, {int(KV[idx])}), amplitude {amp[idx]:.3e}")
    else:
        print("esfera: sem FFT (v não é periódico); use a imagem")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(H.T, origin="lower", extent=[0, 2 * math.pi, 0, C.DOMAINS[domain]["v"][1]], aspect="auto"); ax[0].set_title("H(u, v)"); fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(K.T, origin="lower", extent=[0, 2 * math.pi, 0, C.DOMAINS[domain]["v"][1]], aspect="auto"); ax[1].set_title("K(u, v)"); fig.colorbar(im1, ax=ax[1])
        for x in ax: x.set_xlabel("u"); x.set_ylabel("v")
        fig.tight_layout(); fig.savefig(a.out, dpi=140)
        print(f"imagem salva: {a.out}")
    except Exception as e:
        print("sem imagem:", e)


if __name__ == "__main__":
    main()
