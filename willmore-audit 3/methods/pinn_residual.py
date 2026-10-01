"""
methods/pinn_residual.py   (Lectures 12, 13: o PINN propriamente dito, em forma fraca)

Segundo braço da tabela de ingredientes: mesmo tipo de ansatz (MLP tanh sobre
features de Fourier), mas o loss é a EQUAÇÃO DE WILLMORE em vez da energia.

Versão forte (ΔH + 2H(H² − K) = 0 pontualmente) precisa de quartas derivadas de
phi e, para o gradiente nos parâmetros, de modo reverso por cima de quatro
níveis de jacfwd aninhados; isso falhou na rodada de 30/09 (erro de operação
in-place no torch usado). Esta versão usa a FORMA FRACA, que só precisa de
segundas derivadas, as mesmas do fluxo publicado:

    δW[ξ e_i] = d/dε W(phi + ε ξ e_i) = ∫ r ξ (e_i · n) dA,   r = ΔH + 2H(H² − K)

(a componente tangencial da variação não muda W, por invariância sob
reparametrização). Com ξ_k funções-teste de baixa frequência e e_i os três
eixos, as funções ξ_k (e_i·n) testam o resíduo r nas direções suaves. O
loss é

    loss = Σ_{k,i} δW[ξ_k e_i]²      (diferença central em ε, grade fixa pequena)
         + λ_W · W                   (sem ele r = 0 não escolhe o mínimo)
         + λ_reg · piso de área      (contra colapso, como no fluxo publicado)

É a medição variacional do certificado C4 (certificates.willmore_residual_variational)
usada como loss. O que a linha responde: descer no resíduo, e não na energia,
produz superfícies mais críticas (C4 menor) ao custo de quê em W?

Custo: por passo, 2 × 3 × n_tests avaliações de W numa grade res_grid²
(default 24², 4 testes: 24 avaliações de 576 pontos), segundas derivadas por
jacfwd aninhado e backward por cima, como no linear_spectral. CPU de laptop:
alguns segundos por passo.
"""
from __future__ import annotations

import math
import os
import sys

import torch

from .base import Surface, register
from .linear_spectral import FourierTorusFeatures

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import certificates as C  # noqa: E402


class SmallMLP(torch.nn.Module):
    def __init__(self, K: int = 4, hidden=(64, 64)):
        super().__init__()
        self.features = FourierTorusFeatures(K)
        layers, d = [], 4 * (K + 1) ** 2
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.Tanh()]
            d = h
        layers.append(torch.nn.Linear(d, 3))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, uv):
        return self.net(self.features(uv))


# as mesmas funções-teste do certificado C4 variacional, mais duas
TESTS = [
    lambda uv: torch.cos(uv[:, 0]) * torch.sin(2 * uv[:, 1]) + 0.5 * torch.cos(uv[:, 1]),
    lambda uv: torch.cos(2 * uv[:, 0]) * torch.sin(uv[:, 1]) + 0.5 * torch.cos(uv[:, 1]),
    lambda uv: torch.cos(uv[:, 0]) * torch.sin(uv[:, 1]) + 0.5 * torch.cos(uv[:, 1]),
    lambda uv: torch.cos(2 * uv[:, 0]) * torch.sin(2 * uv[:, 1]) + 0.5 * torch.cos(uv[:, 1]),
    lambda uv: torch.sin(uv[:, 0]) * torch.cos(uv[:, 1]),
    lambda uv: torch.sin(uv[:, 1]),
]


def _W_grid(phi, uv, w):
    """W = ∫ H² dA na grade fixa, diferenciável nos parâmetros de phi."""
    geo = C.geometry(phi, uv)
    det = torch.clamp(geo["E"] * geo["G"] - geo["F"] ** 2, min=1e-12)
    return (geo["H"] ** 2 * torch.sqrt(det) * w).sum(), torch.sqrt(det)


def _perturbed(model, xi, axis, eps, sign):
    """phi_ε(p) = phi(p) + sign·ε·ξ(p)·e_axis, como função de p (diferenciável em p)."""
    e = torch.zeros(3); e[axis] = 1.0

    def f(p):
        return model(p) + sign * eps * xi(p).unsqueeze(1) * e
    return f


def weak_residual_sq(model, uv, w, tests, eps):
    total = 0.0
    for xi in tests:
        for axis in range(3):
            Wp, _ = _W_grid(_perturbed(model, xi, axis, eps, +1.0), uv, w)
            Wm, _ = _W_grid(_perturbed(model, xi, axis, eps, -1.0), uv, w)
            total = total + ((Wp - Wm) / (2 * eps)) ** 2
    return total


@register("pinn_residual")
def train(genus: int = 1, K: int = 4, hidden=(64, 64), epochs: int = 300, lr: float = 1e-4, seed: int = 0,
          res_grid: int = 24, n_tests: int = 6, eps: float = 1e-3, lam_W: float = 0.05, lam_reg: float = 10.0,
          area_floor: float = 0.01, pretrain_steps: int = 500, verbose: bool = True) -> Surface:
    if genus != 1:
        raise NotImplementedError("pinn_residual: só gênero 1 nesta versão (features de Fourier)")
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    dom = "torus"
    model = SmallMLP(K, hidden)

    # pretraining supervisionado ao Clifford, como o fluxo publicado faz
    ref = C.exact_torus()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(pretrain_steps):
        uv, _ = C.quadrature_grid(dom, 32, 32)
        loss = ((model(uv) - ref(uv)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    uv_res, w_res = C.quadrature_grid(dom, res_grid, res_grid)
    tests = TESTS[:n_tests]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        res = weak_residual_sq(model, uv_res, w_res, tests, eps)
        W, dA = _W_grid(model, uv_res, w_res)
        reg = torch.relu(area_floor - dA).mean()
        loss = res + lam_W * W + lam_reg * reg
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append((float(res), float(W)))
        if verbose and ep % 25 == 0:
            print(f"  epoch {ep:4d}  sum dW^2 = {float(res):.4e}   W_grid = {float(W):.4f}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Surface(name=f"pinn_residual/K{K}/seed{seed}", genus=1, phi=model,
                   meta=dict(source="weak-form residual-loss PINN", K=K, hidden=list(hidden), epochs=epochs,
                             seed=seed, lam_W=lam_W, res_grid=res_grid, n_tests=n_tests, eps=eps,
                             history=hist, n_params=sum(p.numel() for p in model.parameters())))
