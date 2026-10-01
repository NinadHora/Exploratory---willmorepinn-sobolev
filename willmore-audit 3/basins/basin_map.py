"""
basins/basin_map.py   (Lectures 7 e 8: agrupamento no quociente pela simetria; kernel PCA)

Entrada: a pasta runs/ com um JSON de auditoria por superfície final.
Coordenadas Möbius-invariantes de cada superfície (seção 13 do protocolo):
  x1 = W_cert
  x2 = ||Å||_L2 = sqrt(2 (W_cert − 4π(1−g)))     (identidade da eq. (1) do plano)
  x3 = R/r ajustado: razão de aspecto do toro de revolução que tem o mesmo W
       (inversa da fórmula fechada π²ρ²/√(ρ²−1), ρ = R/r), definida para W ≥ 2π²
Anotações: faixa do C4 e, quando existir, índice do C7.

Agrupar no quociente pela simetria significa agrupar SÓ nessas coordenadas
invariantes, nunca em coordenadas de R³ (duas superfícies congruentes por
Möbius são o mesmo ponto). DBSCAN e não k-means: o número de bacias é a
resposta, não a entrada. Kernel PCA das trajetórias (série de W_cert por
checkpoint) para desenhar os caminhos.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, List

import numpy as np

PI = math.pi


def aspect_ratio_from_W(W: float) -> float:
    """Resolve π² ρ² / sqrt(ρ² − 1) = W para ρ = R/r ≥ √2 (ramo estável), por bisseção."""
    if W < 2 * PI ** 2:
        return float("nan")
    lo, hi = math.sqrt(2.0), 1e3
    f = lambda rho: PI ** 2 * rho ** 2 / math.sqrt(rho ** 2 - 1) - W
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def features(audit: Dict) -> List[float]:
    W, g = audit["W"], audit["genus"]
    return [W, math.sqrt(max(2 * (W - 4 * PI * (1 - g)), 0.0)), aspect_ratio_from_W(W) if g == 1 else float("nan")]


def load_runs(runs_dir: str) -> List[Dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(p) as f:
            a = json.load(f)
        a["_file"] = p
        out.append(a)
    return out


def cluster(audits: List[Dict], eps: float = 0.5, min_samples: int = 3) -> Dict[str, object]:
    """DBSCAN nas coordenadas invariantes (padronizadas). Devolve rótulos e o número de bacias."""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    X = np.array([features(a)[:2] for a in audits])          # R/r só anota; W e ||Å|| agrupam
    Xs = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
    return dict(labels=labels.tolist(), n_basins=int(len(set(labels)) - (1 if -1 in labels else 0)),
                names=[a.get("surface") for a in audits], X=X.tolist(),
                bands=[a["C4"]["band"] if a.get("C4") else None for a in audits])


def trajectory_kpca(trajectories: List[List[float]], n_components: int = 2):
    """Kernel PCA (RBF) das séries W_cert(época), todas reamostradas ao mesmo comprimento."""
    from sklearn.decomposition import KernelPCA
    L = min(len(t) for t in trajectories)
    T = np.array([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(t)), t) for t in trajectories])
    return KernelPCA(n_components=n_components, kernel="rbf").fit_transform(T)


def figure(audits: List[Dict], out_png: str):
    """Figura 2 do protocolo: W_cert × ||Å||_L2, cor = bacia, marcador = faixa do C4."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = cluster(audits)
    X = np.array(res["X"])
    markers = {"critical": "o", "near-critical": "s", "not critical": "x", None: "."}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (x, lab, band) in enumerate(zip(X, res["labels"], res["bands"])):
        ax.scatter(x[0], x[1], c=[lab], cmap="tab10", vmin=-1, vmax=9, marker=markers.get(band, "."), s=40)
    ax.axvline(2 * PI ** 2, ls=":", color="gray", label="Clifford, 2π²")
    ax.set_xlabel("W certified"); ax.set_ylabel("||Å||_L2"); ax.set_title(f"basins found: {res['n_basins']}")
    ax.legend(); fig.tight_layout(); fig.savefig(out_png, dpi=160)
    return res
