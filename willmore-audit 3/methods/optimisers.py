"""
methods/optimisers.py   (Lecture 2: o fluxo é o gradiente na métrica do otimizador)

Terceiro braço da tabela: mesmo ansatz, mesmo loss, fluxo diferente.

  adam / adamw / sgd   já existem no run.py deles: basta o override de config
  sobolev              o pré-condicionador de willmorepinn-sobolev (gênero 1),
                       aplicado pelo patch_runpy.py; híbrido 70% Sobolev + 30% Adam
  lbfgs                não existe no run.py: esqueleto abaixo, a escrever

Cada braço devolve um Surface e é auditado pelo mesmo protocolo. A pergunta que
a linha responde: o certificado C4 (ponto crítico) e a monotonicidade C6
dependem do otimizador ou só do ansatz?
"""
from __future__ import annotations

from typing import Dict, Optional

from .base import Surface, register
from . import published


@register("optimiser/adam")
def adam(repo: str, genus: int = 1, seed: int = 0, epochs: Optional[int] = None) -> Surface:
    ov = {"optimizer": {"type": "adam"}, "seed": seed}
    if epochs:
        ov["training"] = {"num_epochs": epochs}
    return published.train(repo, genus, ov, tag=f"adam_seed{seed}")


@register("optimiser/adamw")
def adamw(repo: str, genus: int = 1, seed: int = 0, weight_decay: float = 1e-5) -> Surface:
    return published.train(repo, genus, {"optimizer": {"type": "adamw"}, "training": {"weight_decay": weight_decay},
                                         "seed": seed}, tag=f"adamw_seed{seed}")


@register("optimiser/sgd")
def sgd(repo: str, genus: int = 1, seed: int = 0, momentum: float = 0.9) -> Surface:
    return published.train(repo, genus, {"optimizer": {"type": "sgd", "momentum": momentum}, "seed": seed},
                           tag=f"sgd_seed{seed}")


@register("optimiser/sobolev")
def sobolev(repo: str, genus: int = 1, seed: int = 0) -> Surface:
    """Pré-requisito: sobolev_optim.py copiado para o repo e patch_runpy.py aplicado
    (README_genus2_patch.md do willmorepinn-sobolev). O patch liga o
    pré-condicionador nos primeiros 70% das épocas (flag _NG_ACTIVE) e deixa o
    Adam polir o resto; foi isso que deu W = 20.13 no gênero 1.
    Só gênero 1: no gênero 2 o bloco-diagonal descola a costura (resultado
    negativo já documentado)."""
    if genus != 1:
        raise NotImplementedError("Sobolev multi-carta é incompatível com a colagem; ver relatório técnico")
    return published.train(repo, genus, {"seed": seed}, tag=f"sobolev_seed{seed}")


@register("optimiser/lbfgs")
def lbfgs(repo: str, genus: int = 1, seed: int = 0) -> Surface:
    """A ESCREVER. O run.py não tem L-BFGS. Caminho: reutilizar create_embedding_model
    e create_embedding_loss do repo deles e escrever um laço próprio com
    torch.optim.LBFGS(history_size=20, line_search_fn='strong_wolfe') sobre uma
    amostra FIXA de 5000 pontos por época (L-BFGS supõe loss determinístico
    dentro do closure; reamostrar a cada chamada quebra a busca de linha)."""
    raise NotImplementedError
