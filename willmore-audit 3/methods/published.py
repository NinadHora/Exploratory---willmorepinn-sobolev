"""
methods/published.py

O fluxo publicado (Hirst, Sá Earp, Silva) como Surface, de dois jeitos:

  from_checkpoint(repo, ckpt)            carrega best_model.pt e devolve o Surface
  train(repo, genus, overrides, tag)     roda o treino deles com um config modificado
                                         (τ, épocas, seed, pesos do loss) e devolve o
                                         Surface do melhor checkpoint

O segundo é o que a frente das bacias e o braço dos otimizadores usam: cada
run é o código deles intacto com um dicionário de config diferente. Nada do
run.py é editado; o Sobolev entra pelo patch que já existe no repositório
willmorepinn-sobolev.
"""
from __future__ import annotations

import copy
import glob
import os
import sys
from typing import Dict, Optional

import torch
import yaml

from .base import Surface, register


def _import_repo(repo: str):
    repo = os.path.expanduser(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import model as wp_model      # noqa
    import run as wp_run          # noqa
    return wp_model, wp_run


@register("published/checkpoint")
def from_checkpoint(repo: str, ckpt: str, name: Optional[str] = None) -> Surface:
    wp_model, _ = _import_repo(repo)
    ck = torch.load(os.path.expanduser(ckpt), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    genus = cfg.get("topology", {}).get("genus", 1)
    if genus == 2:
        raise NotImplementedError("gênero 2: quadratura de duas cartas ainda não existe no harness")
    m = wp_model.create_embedding_model(cfg, torch.device("cpu"), skip_init=True)
    m.load_state_dict(ck.get("model", ck.get("model_state_dict")))
    m = m.double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return Surface(name=name or f"published/{os.path.basename(os.path.dirname(ckpt))}", genus=genus, phi=m,
                   meta=dict(source="WillmorePINN checkpoint", ckpt=ckpt, epoch=ck.get("epoch"),
                             reported_loss=ck.get("loss"), config=cfg))


@register("published/train")
def train(repo: str, genus: int, overrides: Optional[Dict] = None, tag: str = "") -> Surface:
    """Treina com o run.py deles a partir de configs/config_genus{g}.yaml + overrides.
    overrides é um dicionário aninhado, por exemplo
        {"topology": {"torus": {"tau": "0.8j"}}, "training": {"num_epochs": 600}, "seed": 7}
    Devolve o Surface do best_model.pt do run recém-criado."""
    wp_model, wp_run = _import_repo(repo)
    repo = os.path.expanduser(repo)
    with open(os.path.join(repo, "configs", f"config_genus{genus}.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg = _deep_update(copy.deepcopy(cfg), overrides or {})
    before = set(glob.glob(os.path.join(repo, cfg["output"]["checkpoint_dir"], "run_*")))
    cwd = os.getcwd()
    os.chdir(repo)                       # run.py escreve checkpoints/ e logs/ relativos
    try:
        wp_run.train(config_dict=cfg)
    finally:
        os.chdir(cwd)
    after = set(glob.glob(os.path.join(repo, cfg["output"]["checkpoint_dir"], "run_*")))
    new = sorted(after - before, key=os.path.getmtime)
    if not new:
        raise RuntimeError("o treino não criou um run_N novo; veja os logs do run.py")
    ckpt = os.path.join(new[-1], "best_model.pt")
    s = from_checkpoint(repo, ckpt, name=f"published/{tag or os.path.basename(new[-1])}")
    s.meta["overrides"] = overrides
    return s


def _deep_update(base: Dict, upd: Dict) -> Dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base
