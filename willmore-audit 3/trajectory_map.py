"""
trajectory_map.py: a trajetória de UM treino como curva no espaço de superfícies
(Lectures 7, 8: PCA / kernel PCA de trajetórias) e, de graça, o certificado C6
(Kuwert-Schätzle: a energia certificada é monótona ao longo do treino?).

Para cada checkpoint por época de uma pasta run_N do WillmorePINN:
  - W_cert em 96² e Gauss-Bonnet (rápido: sem resíduo)
  - o mapa de H numa grade 48² (Möbius-invariante a menos de escala; é a
    "coordenada" da superfície)
Depois:
  - PCA da sequência de mapas de H: espectro (quantas direções o treino usa) e
    projeção 2D colorida pela época
  - energia espectral de H por banda (modos <= kcut e > kcut) ao longo das
    épocas: o treino mexe nos modos altos ou eles ficam parados desde o
    pretraining?
  - W_cert e a loss salva contra a época (C6)

    python trajectory_map.py --repo ~/Downloads/WillmorePINN-main --run ~/Downloads/WillmorePINN-main/checkpoints/run_1
Saída: runs/trajectory__run_1.json e runs/fig3_trajectory_run_1.png
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certificates as C                      # noqa: E402
from methods.published import from_checkpoint  # noqa: E402


def epoch_of(path):
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[-1]) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run", required=True, help="pasta checkpoints/run_N")
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--hgrid", type=int, default=48)
    ap.add_argument("--kcut", type=int, default=6)
    ap.add_argument("--every", type=int, default=1, help="usa 1 checkpoint a cada N")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    run = os.path.expanduser(a.run)
    files = [f for f in glob.glob(os.path.join(run, "*.pt")) if epoch_of(f) >= 0
             and not os.path.basename(f).startswith(("best", "latest"))]
    files = sorted(files, key=epoch_of)[::a.every]
    if not files:
        raise SystemExit(f"nenhum checkpoint por época em {run}; arquivos: {os.listdir(run)[:10]}")
    label = os.path.basename(run.rstrip("/"))
    print(f"{len(files)} checkpoints em {run}")

    epochs, W, GB, loss, Hmaps = [], [], [], [], []
    dom = None
    t0 = time.time()
    for k, f in enumerate(files):
        s = from_checkpoint(a.repo, f)
        dom = C.domain_for_genus(s.genus)
        uv, w = C.quadrature_grid(dom, a.grid, a.grid)
        geo = C.geometry(s.phi, uv)
        W.append(float((geo["H"] ** 2 * geo["dA"] * w).sum()))
        GB.append(float((geo["K"] * geo["dA"] * w).sum() - 4 * math.pi * (1 - s.genus)))
        uvh, _ = C.quadrature_grid(dom, a.hgrid, a.hgrid)
        Hmaps.append(C.geometry(s.phi, uvh)["H"].reshape(a.hgrid, a.hgrid).numpy())
        epochs.append(s.meta.get("epoch", epoch_of(f)))
        loss.append(s.meta.get("reported_loss"))
        if k % 20 == 0:
            print(f"  [{k + 1}/{len(files)}] epoch {epochs[-1]}  W_cert {W[-1]:.4f}  GB {GB[-1]:.1e}  ({time.time() - t0:.0f}s)")
    Wmin = 4 * math.pi if dom == "sphere" else 2 * math.pi ** 2
    X = np.stack([h.ravel() for h in Hmaps])                      # (T, hgrid²)

    # PCA da trajetória
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    proj = Xc @ Vt[:2].T
    n90 = int(np.searchsorted(np.cumsum(var), 0.90) + 1)
    n99 = int(np.searchsorted(np.cumsum(var), 0.99) + 1)

    # energia espectral de H por banda, por época
    F = np.fft.fft2(np.stack(Hmaps))
    ku = np.fft.fftfreq(a.hgrid, 1 / a.hgrid); kv = ku
    KU, KV = np.meshgrid(ku, kv, indexing="ij")
    high = (np.abs(KU) > a.kcut) | (np.abs(KV) > a.kcut)
    P = np.abs(F) ** 2
    P[:, 0, 0] = 0.0                                              # sem a média
    frac_high = P[:, high].sum(1) / P.sum((1, 2))
    amp_high = np.sqrt(P[:, high].sum(1)) / a.hgrid ** 2

    # C6: monotonicidade de W_cert
    dW = np.diff(W)
    ups = int((dW > 1e-6 * np.abs(W[:-1])).sum())

    res = dict(kind="trajectory", label=label, n_checkpoints=len(files), epochs=epochs, W_cert=W, GB=GB,
               saved_loss=loss, W_min=Wmin, pca_variance=var[:10].tolist(), n_components_90=n90, n_components_99=n99,
               frac_high_modes=frac_high.tolist(), amp_high_modes=amp_high.tolist(), kcut=a.kcut,
               C6_increases=ups, C6_monotone=bool(ups == 0),
               W_first=W[0], W_last=W[-1], W_best=float(min(W)), epoch_best=int(epochs[int(np.argmin(W))]))
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out = os.path.join(HERE, "runs", f"trajectory__{label}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)

    # figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(epochs, W, "k-", label="W_cert (96²)")
    if all(l is not None for l in loss):
        ax[0].plot(epochs, loss, color="tab:orange", alpha=0.8, label="saved loss (reported)")
    ax[0].axhline(Wmin, color="gray", ls="--", lw=1, label="minimum")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("W"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"C6: W_cert along training ({ups} increases)", fontsize=10)
    sc = ax[1].scatter(proj[:, 0], proj[:, 1], c=epochs, cmap="viridis", s=14)
    ax[1].plot(proj[:, 0], proj[:, 1], color="gray", lw=0.5)
    plt.colorbar(sc, ax=ax[1], label="epoch")
    ax[1].set_title(f"PCA of the H maps: {n90} comp. for 90%, {n99} for 99%", fontsize=10)
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    ax[2].plot(epochs, 100 * frac_high, "k-")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel(f"% of H's spectral energy in modes > {a.kcut}")
    ax[2].set_title("Does training touch the high modes?", fontsize=10)
    fig.suptitle(f"{label}: trajectory of the published flow", fontsize=11)
    fig.tight_layout()
    figp = os.path.join(HERE, "runs", f"fig3_trajectory_{label}.png")
    fig.savefig(figp, dpi=150)
    print(json.dumps({k: v for k, v in res.items() if k not in ("epochs", "W_cert", "GB", "saved_loss", "frac_high_modes", "amp_high_modes")}, indent=2))
    print("saved", out, "and", figp)


if __name__ == "__main__":
    main()
