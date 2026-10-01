"""
attacks/ladder.py   (Lecture 9: a escada de busca, do aleatório ao Q-learning)

Um ataque é uma superfície que o treino aceita e um certificado rejeita.
Recompensa (seção 12 do protocolo): reward = W_cert − W_Huber, sujeito a
  (a) topologia intacta (C1 dentro da tolerância na grade convergida) e
  (b) limiares de regularidade do config publicado satisfeitos.
Sucesso: reward > 2π (meio quantum).

Degraus, cada um com o que pode tocar e quando para:
  1 random_search   (ε, ρ, centro) de uma bolha + semente        implementado
  2 greedy          passos locais em (ε, ρ, centro)              implementado
  3 annealing       perturbações pequenas dos PESOS da rede      esqueleto
  4 qlearning       ações discretas em (ε, ρ, centro)            só se 3 travar
"""
from __future__ import annotations

import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import certificates as C  # noqa: E402
from methods.base import Surface  # noqa: E402

PI = math.pi
# limiares de regularidade do config_genus1/2.yaml publicado
REG = dict(area_floor=0.01, E_G_min=0.001, E_G_max=5.0, mean_area_floor=0.2)


def accepted_by_training(phi, domain: str, n: int = 5000, seed: int = 0) -> Dict[str, object]:
    """(b): os limiares de regularidade do treino publicado numa amostra de 5000 pontos."""
    d = C.DOMAINS[domain]
    g = torch.Generator().manual_seed(seed)
    u = d["u"][0] + (d["u"][1] - d["u"][0]) * torch.rand(n, generator=g, dtype=torch.float64)
    v = d["v"][0] + (d["v"][1] - d["v"][0]) * torch.rand(n, generator=g, dtype=torch.float64)
    geo = C.geometry(phi, torch.stack([u, v], 1))
    dA = torch.sqrt(torch.clamp(geo["E"] * geo["G"] - geo["F"] ** 2, min=1e-12))
    checks = dict(
        area_floor=bool((dA > REG["area_floor"]).all()),
        metric_range=bool(((geo["E"] > REG["E_G_min"]) & (geo["E"] < REG["E_G_max"]) &
                           (geo["G"] > REG["E_G_min"]) & (geo["G"] < REG["E_G_max"])).all()),
        mean_area=bool(dA.mean() > REG["mean_area_floor"]),
    )
    checks["accepted"] = all(checks.values())
    return checks


def evaluate(base: Surface, eps: float, rho: float, uv0: Tuple[float, float],
             grids=(96, 192, 384), seeds: int = 5) -> Dict[str, object]:
    """Uma avaliação da recompensa para uma bolha. É a função que todos os degraus chamam."""
    phi = C.graft_bump(base.phi, base.domain, uv0, eps, rho)
    rs = C.resolution_study(phi, base.domain, grids=grids, verbose=False)
    est = C.estimator_report(phi, base.domain, rs["W"], seeds=seeds)
    acc = accepted_by_training(phi, base.domain)
    gb = rs["rows"][-1]["gb_residual"]
    reward = rs["W"] - est["W_huber_mean"]
    return dict(eps=eps, rho=rho, uv0=list(uv0), W_cert=rs["W"], grid=rs["n"], converged=rs["converged"],
                gb_residual=gb, W_rep=est["W_rep_mean"], W_rep_sd=est["W_rep_sd"],
                W_huber=est["W_huber_mean"], reward=reward, accepted=acc["accepted"], regularity=acc,
                success=bool(rs["converged"] and gb < C.TOL_GB and acc["accepted"] and reward > 2 * PI))


def random_search(base: Surface, eps_grid=(0.05, 0.1, 0.2, 0.3), rho_grid=(0.5, 0.3, 0.2, 0.15),
                  centres: Optional[List[Tuple[float, float]]] = None, seeds: int = 5, verbose=True) -> List[Dict]:
    """Degrau 1: grade 4 × 4 em (ε, ρ), 3 centros, 5 sementes. Para quando encontra
    sucesso ou esgota a grade."""
    if centres is None:
        centres = [(PI, 0.0), (PI, PI), (PI / 2, PI / 2)] if base.genus == 1 else [(PI, PI / 2), (0.0, PI / 3), (PI, 2 * PI / 3)]
    rows = []
    for uv0 in centres:
        for eps in eps_grid:
            for rho in rho_grid:
                r = evaluate(base, eps, rho, uv0, seeds=seeds)
                rows.append(r)
                if verbose:
                    print(f"  ε={eps:.2f} ρ={rho:.2f} uv0={uv0}  reward={r['reward']:+.3f}  accepted={r['accepted']}  success={r['success']}")
                if r["success"]:
                    return rows
    return rows


def greedy(base: Surface, start: Dict, steps: int = 200, patience: int = 20, seeds: int = 5, verbose=True) -> List[Dict]:
    """Degrau 2: a partir do melhor ponto do degrau 1, passos locais em (ε, ρ, u0, v0),
    aceitando só melhorias da recompensa entre pontos aceitos pelo treino."""
    rng = random.Random(0)
    cur = dict(start)
    best, since = cur, 0
    rows = [cur]
    for _ in range(steps):
        cand = dict(eps=max(1e-3, best["eps"] * math.exp(rng.gauss(0, 0.15))),
                    rho=max(1e-3, best["rho"] * math.exp(rng.gauss(0, 0.15))),
                    uv0=(best["uv0"][0] + rng.gauss(0, 0.1), best["uv0"][1] + rng.gauss(0, 0.1)))
        r = evaluate(base, cand["eps"], cand["rho"], cand["uv0"], seeds=seeds)
        rows.append(r)
        if r["accepted"] and r["reward"] > best["reward"]:
            best, since = r, 0
            if verbose:
                print(f"  improved: reward={r['reward']:+.3f} at ε={r['eps']:.3f} ρ={r['rho']:.3f}")
        else:
            since += 1
        if since >= patience or best["success"]:
            break
    return rows


def annealing(base: Surface, layers=("last", "second_last"), rel_size=(1e-3, 1e-2), evaluations: int = 2000):
    """Degrau 3, A ESCREVER. Perturba os pesos das duas últimas camadas da rede
    treinada (tamanho relativo entre 1e-3 e 1e-2), aceita pela regra de
    Metropolis sobre a recompensa, para com taxa de aceitação < 5% por 200
    passos. É o degrau que responde se o ESPAÇO DE PARÂMETROS do método
    publicado contém, perto do treinado, superfícies que enganam o loss.
    Precisa que base.phi seja o nn.Module (published.from_checkpoint dá isso)."""
    raise NotImplementedError


def qlearning(base: Surface):
    """Degrau 4, só se o 3 travar. Ações discretas em (ε, ρ, centro); tabela Q.
    O plano permite parar antes deste degrau."""
    raise NotImplementedError
