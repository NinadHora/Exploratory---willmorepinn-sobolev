"""
methods/others.py   Esqueletos dos braços restantes da tabela de ingredientes.
Cada um tem a interface fechada (assinatura, o que devolve, de qual aula vem,
o que a linha responde) e o corpo a escrever. pinn_residual vive em methods/pinn_residual.py. Ordem do plano: cnn_periodic, set_transformer, mesh_flow só se sobrar tempo.
"""
from __future__ import annotations

from typing import Optional

from .base import Surface, register


@register("cnn_periodic")
def cnn_periodic(genus: int = 1, epochs: int = 2000, seed: int = 0) -> Surface:
    """Lecture 5. Ansatz com simetria embutida: uma CNN 1D/2D com padding circular
    sobre a grade (u, v) do toro produz phi numa grade fixa; a superfície
    contínua vem de interpolação trigonométrica da saída (que mantém a
    periodicidade e dá derivadas exatas por FFT).
    Pergunta da linha: a equivariância a translações em (u,v) muda a bacia
    que o fluxo alcança?"""
    raise NotImplementedError


@register("set_transformer")
def set_transformer(genus: int = 1, epochs: int = 2000, seed: int = 0) -> Surface:
    """Lecture 6. Ansatz sobre pontos amostrados sem ordem: atenção sobre o
    conjunto de (u,v) com features espectrais como query. O Surface resultante
    precisa continuar sendo uma função (N,2)->(N,3) diferenciável em cada
    ponto, então a saída por ponto não pode depender dos outros pontos do lote
    (usar apenas cross-attention de cada ponto a um conjunto FIXO de âncoras)."""
    raise NotImplementedError


@register("mesh_flow")
def mesh_flow(genus: int = 1, steps: int = 5000, seed: int = 0) -> Surface:
    """A demo de Plateau da disciplina (plateau_solver.py) estendida de área
    para ∫H² em malha triangular: fluxo de Willmore discreto (Bobenko e
    Schröder 2005). O Surface aqui não é uma rede: phi é a interpolação da
    malha, e os certificados C1 e C8 são exatos em malha. É o único braço sem
    autodiff, portanto a única linha da tabela em que 'computação de
    curvatura' é um ingrediente trocado de verdade. Rígido (passo explícito
    pequeno); cortado primeiro se faltar tempo."""
    raise NotImplementedError
