"""
methods/base.py

A interface comum. Toda estratégia do plano (fluxo publicado, ansatz linear,
PINN de resíduo, otimizadores, CNN periódica, set transformer, fluxo em malha)
termina num objeto Surface, e um Surface é a única coisa que o harness audita.

    class Surface:
        name   : str                 rótulo do run ("published/run_1", "linear_spectral/K6/seed0")
        genus  : int                 0 ou 1 (2 quando a quadratura de duas cartas existir)
        phi    : (N,2)->(N,3) torch  diferenciável; float64
        meta   : dict                tudo que explica de onde veio (config, seeds, commit, épocas)

Regra: quem produz um Surface não calcula certificado nenhum; quem audita não
sabe de onde o Surface veio. Essa separação é o que faz a tabela de ingredientes
(Problema (d)) valer alguma coisa.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import certificates as C  # noqa: E402


@dataclass
class Surface:
    name: str
    genus: int
    phi: Callable[[torch.Tensor], torch.Tensor]
    meta: Dict = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return C.domain_for_genus(self.genus)

    def audit(self, grid: int = 96, residual: bool = True, residual_grid: int = 32,
              resolution_study: bool = False, seeds: int = 5, verbose: bool = True) -> Dict:
        """Roda o protocolo inteiro neste Surface e devolve o dicionário que vira
        uma linha da tabela de auditoria. É a única porta de entrada para os
        certificados; nenhum método chama certificates.py diretamente."""
        rs = None
        if resolution_study:
            rs = C.resolution_study(self.phi, self.domain, verbose=verbose)
            grid = rs["n"]
        cert = C.certify(self.phi, self.genus, nu=grid, nv=grid, residual=residual,
                         residual_grid=(residual_grid, residual_grid), verbose=verbose)
        cert["estimator"] = C.estimator_report(self.phi, self.domain, cert["W"], seeds=seeds)
        cert["resolution_study"] = rs
        cert["surface"] = self.name
        cert["meta"] = self.meta
        return cert


# Registro: cada módulo em methods/ registra um construtor "nome -> Surface",
# para que os scripts de SLURM chamem qualquer método pelo nome.
REGISTRY: Dict[str, Callable[..., Surface]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def exact(name: str) -> Surface:
    """As superfícies exatas também são Surfaces: são a linha de base trivial
    (Lecture 2) e os testes unitários de toda a tabela."""
    table = {
        "sphere": (0, C.exact_sphere()),
        "ellipsoid_112": (0, C.exact_ellipsoid(1, 1, 2)),
        "clifford": (1, C.exact_torus()),
        "torus_R3": (1, C.exact_torus(3, 1)),
    }
    g, phi = table[name]
    return Surface(name=f"exact/{name}", genus=g, phi=phi, meta={"source": "exact formula"})
