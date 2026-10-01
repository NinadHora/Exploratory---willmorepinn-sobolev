"""
basins/perturb_init.py   (Problema (c): de quais dados iniciais o fluxo chega onde)

Gera as 30 a 50 superfícies iniciais do mapa de bacias, de três famílias:
  (i)  toro de referência + perturbação de Fourier aleatória de amplitude controlada
  (ii) toros torcidos: Re(τ) ≠ 0 (o WillmorePINN já aceita τ complexo no config)
  (iii) tubo em torno do nó toral (2,3)

Como entram no fluxo publicado: o run.py inicializa a rede por pretraining
supervisionado ao embedding de referência (sampling.get_reference_embedding).
Para (ii) basta o override de config em τ. Para (i) e (iii) o alvo do
pretraining tem de ser a superfície perturbada: o gancho abaixo
(install_reference_hook) substitui get_reference_embedding pela função
escolhida antes de chamar published.train. Nenhuma linha do run.py muda.

Cada superfície inicial também é um Surface e é auditada ANTES do treino: a
energia do dado inicial é a linha de base trivial do plano (Lecture 2).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable, Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import certificates as C  # noqa: E402
from methods.base import Surface  # noqa: E402

PI = math.pi


def fourier_perturbed_torus(amplitude: float, seed: int, K: int = 3, R: float = math.sqrt(2.0), r: float = 1.0):
    """(i) phi = toro de referência + amplitude · (soma de modos de Fourier aleatórios) · n."""
    g = torch.Generator().manual_seed(seed)
    coef = torch.randn(K + 1, K + 1, 4, generator=g, dtype=torch.float64)
    coef = coef / coef.norm()
    base = C.exact_torus(R, r)
    f = C._single(base)

    def normal(uv):
        return torch.func.vmap(lambda p: C._geometry_point(f, p)["n"])(uv)

    def bump(uv):
        u, v = uv[:, 0], uv[:, 1]
        s = torch.zeros_like(u)
        for a in range(K + 1):
            for b in range(K + 1):
                s = s + (coef[a, b, 0] * torch.cos(a * u) * torch.cos(b * v) + coef[a, b, 1] * torch.cos(a * u) * torch.sin(b * v)
                         + coef[a, b, 2] * torch.sin(a * u) * torch.cos(b * v) + coef[a, b, 3] * torch.sin(a * u) * torch.sin(b * v))
        return s

    def phi(uv):
        return base(uv) + amplitude * bump(uv).unsqueeze(1) * normal(uv)
    return phi


def torus_knot_tube(p: int = 2, q: int = 3, R: float = 2.0, r: float = 0.8, tube: float = 0.3):
    """(iii) tubo de raio `tube` em torno do nó (p,q) sobre o toro (R, r). Gênero 1,
    mas numa classe de isotopia diferente do toro padrão: um dado inicial que
    o fluxo não consegue levar ao Clifford sem cruzar a si mesmo."""
    def curve(t):
        x = (R + r * torch.cos(q * t)) * torch.cos(p * t)
        y = (R + r * torch.cos(q * t)) * torch.sin(p * t)
        z = r * torch.sin(q * t)
        return torch.stack([x, y, z], 1)

    def phi(uv):
        t, s = uv[:, 0], uv[:, 1]
        c = curve(t)
        T = torch.func.vmap(torch.func.jacfwd(lambda tt: curve(tt.unsqueeze(0)).squeeze(0)))(t)
        T = T / T.norm(dim=1, keepdim=True)
        ref = torch.tensor([0.0, 0.0, 1.0], dtype=uv.dtype).expand_as(T)
        N1 = torch.linalg.cross(T, ref); N1 = N1 / N1.norm(dim=1, keepdim=True)
        N2 = torch.linalg.cross(T, N1)
        return c + tube * (torch.cos(s).unsqueeze(1) * N1 + torch.sin(s).unsqueeze(1) * N2)
    return phi


def initial_family(n_fourier: int = 30, amplitudes=(0.05, 0.1, 0.2), taus=("0.1j", "0.2j", "0.05j", "0.05+0.1j", "0.1+0.1j"),
                   knot: bool = True) -> List[Dict]:
    """A lista de dados iniciais, cada um como dicionário {name, kind, surface | tau_override}."""
    items = []
    k = 0
    for amp in amplitudes:
        for _ in range(n_fourier // len(amplitudes)):
            items.append(dict(name=f"fourier/amp{amp}/seed{k}", kind="reference_hook",
                              surface=Surface(f"init/fourier/amp{amp}/seed{k}", 1, fourier_perturbed_torus(amp, k),
                                              meta=dict(amplitude=amp, seed=k))))
            k += 1
    for tau in taus:
        items.append(dict(name=f"twisted/tau{tau}", kind="tau_override", tau=tau))
    if knot:
        items.append(dict(name="knot/2_3", kind="reference_hook",
                          surface=Surface("init/knot_2_3", 1, torus_knot_tube(), meta=dict(p=2, q=3))))
    return items


def install_reference_hook(repo: str, phi: Callable):
    """Faz o pretraining do run.py mirar em `phi` em vez do toro de referência.
    Chamar ANTES de published.train; desfazer com o valor devolvido."""
    repo = os.path.expanduser(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import sampling  # do WillmorePINN
    original = sampling.get_reference_embedding

    def hooked(uv, *args, **kwargs):
        return phi(uv.double()).to(uv.dtype)
    sampling.get_reference_embedding = hooked
    return original
