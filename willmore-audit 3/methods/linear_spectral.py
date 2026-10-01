"""
methods/linear_spectral.py   (Lecture 3: modelos lineares em espaços de funções)

Primeiro braço da tabela de ingredientes: troca o MLP tanh por um modelo LINEAR
nas mesmas features espectrais, e mantém o resto (loss de Willmore por Monte
Carlo, regularidade de área, Adam). Se isso chega a 2π², a profundidade do MLP
não era o ingrediente; se não chega, era.

    phi(u,v) = sum_k  c_k · f_k(u,v),   c_k em R³

gênero 1: f_k = {cos(a u) cos(b v), cos(a u) sin(b v), sin(a u) cos(b v), sin(a u) sin(b v)}, 0 <= a,b <= K
gênero 0: harmônicos esféricos reais até grau L (a mesma classe do WillmorePINN),
          aqui via a camada SphericalHarmonicEmbedding do repositório deles.

Escrito e não executado (este ambiente não tem torch). Rodar primeiro com
--epochs 50 e conferir que W desce e Gauss-Bonnet fecha antes de qualquer run longo.
"""
from __future__ import annotations

import itertools
import math
import os
import sys
from typing import Dict, Optional

import torch

from .base import Surface, register

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import certificates as C  # noqa: E402


class FourierTorusFeatures(torch.nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.pairs = list(itertools.product(range(K + 1), repeat=2))

    def forward(self, uv):
        u, v = uv[:, 0:1], uv[:, 1:2]
        cols = []
        for a, b in self.pairs:
            cols += [torch.cos(a * u) * torch.cos(b * v), torch.cos(a * u) * torch.sin(b * v),
                     torch.sin(a * u) * torch.cos(b * v), torch.sin(a * u) * torch.sin(b * v)]
        return torch.cat(cols, 1)


class LinearSpectral(torch.nn.Module):
    def __init__(self, genus: int, K: int, repo: Optional[str] = None):
        super().__init__()
        if genus == 1:
            self.features = FourierTorusFeatures(K)
            nfeat = 4 * (K + 1) ** 2
        else:
            repo = os.path.expanduser(repo)
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from model import SphericalHarmonicEmbedding
            self.features = SphericalHarmonicEmbedding(max_degree=K)
            nfeat = (K + 1) ** 2
        self.coef = torch.nn.Linear(nfeat, 3, bias=False)

    def forward(self, uv):
        return self.coef(self.features(uv))


def _init_to_reference(model: LinearSpectral, genus: int, steps: int = 300):
    """Ajuste supervisionado ao toro de Clifford (ou esfera) de referência, como o
    pretraining do WillmorePINN. Sem isso o Adam parte de uma nuvem sem topologia."""
    ref = C.exact_torus() if genus == 1 else C.exact_sphere()
    dom = C.domain_for_genus(genus)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        uv, _ = C.quadrature_grid(dom, 48, 48)
        loss = ((model(uv) - ref(uv)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()


@register("linear_spectral")
def train(genus: int = 1, K: int = 6, epochs: int = 2000, n_points: int = 5000, lr: float = 2e-4,
          seed: int = 0, h2_clip: float = 50.0, area_floor: float = 0.01, reg_weight: float = 10.0,
          repo: Optional[str] = None, verbose: bool = True) -> Surface:
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    dom = C.domain_for_genus(genus)
    d = C.DOMAINS[dom]
    area_dom = (d["u"][1] - d["u"][0]) * (d["v"][1] - d["v"][0])
    model = LinearSpectral(genus, K, repo)
    _init_to_reference(model, genus)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        u = d["u"][0] + (d["u"][1] - d["u"][0]) * torch.rand(n_points)
        v = d["v"][0] + (d["v"][1] - d["v"][0]) * torch.rand(n_points)
        geo = C.geometry(model, torch.stack([u, v], 1))
        det = torch.clamp(geo["E"] * geo["G"] - geo["F"] ** 2, min=1e-12)
        dA = torch.sqrt(det)
        H = geo["H"]
        h2 = H * H
        s = math.sqrt(h2_clip)
        h2h = torch.where(h2 <= h2_clip, h2, 2 * s * H.abs() - h2_clip)        # o mesmo Huber do treino
        W_loss = (h2h * dA).mean() * area_dom
        reg = torch.relu(area_floor - dA).mean()                                 # piso de área, como no paper
        loss = W_loss + reg_weight * reg
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float((h2 * dA).mean() * area_dom))
        if verbose and ep % 100 == 0:
            print(f"  epoch {ep:5d}  W_rep = {hist[-1]:.4f}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Surface(name=f"linear_spectral/K{K}/seed{seed}", genus=genus, phi=model,
                   meta=dict(source="linear spectral ansatz", K=K, epochs=epochs, seed=seed, lr=lr,
                             history_W_rep=hist, n_params=sum(p.numel() for p in model.parameters())))
