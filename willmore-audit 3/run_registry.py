"""
run_registry.py

Todo resultado do projeto é um JSON em runs/, com o hash do commit do código
que o produziu (exigência do plano: dados sintéticos, semeados, com commit).

    python run_registry.py audit   --surface exact/clifford
    python run_registry.py audit   --surface published/checkpoint --repo ~/WillmorePINN-main --ckpt .../best_model.pt
    python run_registry.py train   --method linear_spectral --kw genus=1 K=6 epochs=2000 seed=0
    python run_registry.py attack  --surface published/checkpoint --repo ... --ckpt ... --rung 1
    python run_registry.py basins  --runs runs/

O nome do arquivo é <método>__<rótulo>__<hash8>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import methods  # noqa: E402
from methods.base import REGISTRY, Surface, exact  # noqa: E402


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "nogit"


def save(kind: str, label: str, payload: dict, runs_dir: str = "runs") -> str:
    os.makedirs(os.path.join(HERE, runs_dir), exist_ok=True)
    payload = dict(payload)
    payload.update(kind=kind, commit=commit_hash(), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   torch=torch.__version__)
    safe = label.replace("/", "_")
    path = os.path.join(HERE, runs_dir, f"{kind}__{safe}__{payload['commit']}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"saved {path}")
    return path


def _kw(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        out[k] = v
    return out


def get_surface(a) -> Surface:
    if a.surface.startswith("exact/"):
        return exact(a.surface.split("/", 1)[1])
    if a.surface == "published/checkpoint":
        return REGISTRY[a.surface](a.repo, a.ckpt)
    return REGISTRY[a.surface](**_kw(a.kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "train", "attack", "basins"])
    ap.add_argument("--surface"); ap.add_argument("--method")
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--kw", nargs="*", help="pares chave=valor para o método")
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--resolution-study", action="store_true")
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    if a.cmd == "audit":
        s = get_surface(a)
        cert = s.audit(grid=a.grid, resolution_study=a.resolution_study)
        save("audit", s.name, cert, a.runs)

    elif a.cmd == "train":
        s = REGISTRY[a.method](**_kw(a.kw))
        cert = s.audit(grid=a.grid, resolution_study=a.resolution_study)
        save("audit", s.name, cert, a.runs)

    elif a.cmd == "attack":
        from attacks import ladder
        s = get_surface(a)
        rows = ladder.random_search(s)
        best = max(rows, key=lambda r: r["reward"])
        if a.rung >= 2:
            rows += ladder.greedy(s, best)
        save("attack", s.name, dict(surface=s.name, rung=a.rung, rows=rows,
                                    best=max(rows, key=lambda r: r["reward"])), a.runs)

    elif a.cmd == "basins":
        from basins import basin_map
        audits = [x for x in basin_map.load_runs(os.path.join(HERE, a.runs)) if x.get("kind") == "audit"]
        res = basin_map.figure(audits, os.path.join(HERE, a.runs, "fig2_basins.png"))
        save("basins", "map", res, a.runs)


if __name__ == "__main__":
    main()
