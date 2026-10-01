"""
mesh_check.py: segunda metade do certificado C3 (Li-Yau e mergulho).

Constrói uma malha triangular da superfície numa grade regular (u,v) e testa se
algum par de triângulos NÃO vizinhos se intersecta (Möller 1997, vetorizado em
numpy; fase larga por caixas numa grade uniforme). Nenhuma dependência além de
numpy e torch (torch só para avaliar phi).

Leitura, junto com a cota de Li-Yau (W_cert < 8π ⇒ mergulhada):
    não cruza, W < 8π   : mergulhada e certificada; teorema e malha concordam
    não cruza, W ≥ 8π   : mergulhada, mas só a malha fala (o caso do nó treinado, e de gênero 2)
    cruza,     W ≥ 8π   : imersa com auto-interseção; consistente com o teorema
    cruza,     W < 8π   : contradição: a energia certificada está errada, ou a malha
                          achou um quase-toque numérico; refinar os dois e repetir

Uso:
    python mesh_check.py --surface exact_clifford
    python mesh_check.py --surface exact_torus --R 0.5 --r 1.0        # toro-fuso: cruza
    python mesh_check.py --surface knot_init                           # tubo do nó (2,3)
    python mesh_check.py --repo ~/Downloads/WillmorePINN-main --ckpt .../run_1/best_model.pt --genus 1
    python mesh_check.py --bump 0.05 0.15 --repo ... --ckpt ...        # bolha enxertada na rede
Opções: --n 96 (grade), --out runs/mesh__label.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ----------------------------------------------------------------------------
# 1. malha de uma grade (u,v)
# ----------------------------------------------------------------------------
def grid_mesh(phi_np, domain: str, n: int):
    """phi_np: (N,2) numpy -> (N,3) numpy. Devolve vértices V (m,3), faces F (k,3) e as (u,v) de cada vértice."""
    if domain == "torus":
        u = (np.arange(n) + 0.5) * 2 * math.pi / n
        v = (np.arange(n) + 0.5) * 2 * math.pi / n
        U, Vv = np.meshgrid(u, v, indexing="ij")
        uv = np.stack([U.ravel(), Vv.ravel()], 1)
        V = phi_np(uv)
        idx = np.arange(n * n).reshape(n, n)
        i0 = idx
        i1 = np.roll(idx, -1, axis=0)        # u + du (periódico)
        j1 = np.roll(idx, -1, axis=1)        # v + dv (periódico)
        ij = np.roll(i1, -1, axis=1)
        F = np.concatenate([np.stack([i0.ravel(), i1.ravel(), ij.ravel()], 1),
                            np.stack([i0.ravel(), ij.ravel(), j1.ravel()], 1)], 0)
        return V, F, uv
    if domain == "sphere":
        u = (np.arange(n) + 0.5) * 2 * math.pi / n
        v = (np.arange(1, n)) * math.pi / n              # sem os polos
        U, Vv = np.meshgrid(u, v, indexing="ij")
        uv = np.stack([U.ravel(), Vv.ravel()], 1)
        V = phi_np(uv)
        m = n - 1
        idx = np.arange(n * m).reshape(n, m)
        i1 = np.roll(idx, -1, axis=0)
        # faixas entre anéis
        a, b, c, d = idx[:, :-1], i1[:, :-1], i1[:, 1:], idx[:, 1:]
        F = np.concatenate([np.stack([a.ravel(), b.ravel(), c.ravel()], 1),
                            np.stack([a.ravel(), c.ravel(), d.ravel()], 1)], 0)
        # polos: média de phi nos anéis extremos
        north = phi_np(np.stack([u, np.full(n, 1e-6)], 1)).mean(0, keepdims=True)
        south = phi_np(np.stack([u, np.full(n, math.pi - 1e-6)], 1)).mean(0, keepdims=True)
        V = np.concatenate([V, north, south], 0)
        iN, iS = n * m, n * m + 1
        capN = np.stack([np.full(n, iN), idx[:, 0], i1[:, 0]], 1)
        capS = np.stack([np.full(n, iS), i1[:, -1], idx[:, -1]], 1)
        F = np.concatenate([F, capN, capS], 0)
        uv = np.concatenate([uv, [[0, 0]], [[0, math.pi]]], 0)
        return V, F, uv
    raise ValueError(domain)


# ----------------------------------------------------------------------------
# 2. interseção triângulo-triângulo (Möller 1997), vetorizada sobre pares
# ----------------------------------------------------------------------------
def _tri_tri(P, Q, eps=1e-12):
    """P, Q: (k,3,3). Devolve bool (k,) : os triângulos se intersectam (não coplanares)."""
    def plane_dist(T, X):
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        d = -np.einsum("ij,ij->i", n, T[:, 0])
        return n, np.einsum("ij,ikj->ik", n, X) + d[:, None]

    nQ, dP = plane_dist(Q, P)          # distâncias dos vértices de P ao plano de Q
    nP, dQ = plane_dist(P, Q)
    dP = np.where(np.abs(dP) < eps, 0.0, dP)
    dQ = np.where(np.abs(dQ) < eps, 0.0, dQ)
    ok = ~(((dP > 0).all(1)) | ((dP < 0).all(1)) | ((dQ > 0).all(1)) | ((dQ < 0).all(1)))
    coplanar = (dP == 0).all(1)
    ok &= ~coplanar                     # coplanares: ignorados (medida zero numa superfície lisa)
    D = np.cross(nP, nQ)
    ax = np.argmax(np.abs(D), axis=1)
    pP = np.take_along_axis(P, ax[:, None, None].repeat(3, 1), 2)[:, :, 0]   # (k,3) projeções
    pQ = np.take_along_axis(Q, ax[:, None, None].repeat(3, 1), 2)[:, :, 0]

    def interval(p, d):
        # vértice isolado do lado oposto: reordena para que v0 seja o isolado
        s = np.sign(d)
        iso = np.where(s[:, 0] == s[:, 1], 2, np.where(s[:, 0] == s[:, 2], 1, 0))
        # casos com zeros: se um vértice está no plano, trata-o como isolado
        z = (d == 0)
        iso = np.where(z.any(1) & ~z.all(1), np.argmax(z, 1), iso)
        k = np.arange(len(d))
        o = [(iso + 0) % 3, (iso + 1) % 3, (iso + 2) % 3]
        p0, p1, p2 = p[k, o[0]], p[k, o[1]], p[k, o[2]]
        d0, d1, d2 = d[k, o[0]], d[k, o[1]], d[k, o[2]]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = p1 + (p0 - p1) * d1 / (d1 - d0)
            t2 = p2 + (p0 - p2) * d2 / (d2 - d0)
        lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
        return lo, hi

    loP, hiP = interval(pP, dP)
    loQ, hiQ = interval(pQ, dQ)
    overlap = (hiP >= loQ) & (hiQ >= loP)
    return ok & overlap & np.isfinite(loP) & np.isfinite(loQ)


def self_intersections(V, F, cell: float | None = None, max_pairs: int = 50_000_000):
    """Pares (i,j) de faces não vizinhas que se intersectam."""
    T = V[F]                                     # (m,3,3)
    lo, hi = T.min(1), T.max(1)
    if cell is None:
        cell = 2.0 * float(np.median(hi - lo))
    g_lo = np.floor(lo / cell).astype(np.int64)
    g_hi = np.floor(hi / cell).astype(np.int64)
    # cada triângulo entra em todas as células que a sua caixa toca
    keys, owners = [], []
    for i in range(len(F)):
        xs = range(g_lo[i, 0], g_hi[i, 0] + 1); ys = range(g_lo[i, 1], g_hi[i, 1] + 1); zs = range(g_lo[i, 2], g_hi[i, 2] + 1)
        for x in xs:
            for y in ys:
                for z in zs:
                    keys.append((x, y, z)); owners.append(i)
    keys = np.array(keys); owners = np.array(owners)
    order = np.lexsort(keys.T[::-1]); keys, owners = keys[order], owners[order]
    _, start, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    pairs = []
    for s, c in zip(start, counts):
        if c < 2:
            continue
        o = owners[s:s + c]
        ii, jj = np.triu_indices(c, 1)
        pairs.append(np.stack([o[ii], o[jj]], 1))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.unique(np.concatenate(pairs, 0), axis=0)
    # exclui vizinhos (compartilham vértice) e caixas que não se tocam
    Fa, Fb = F[pairs[:, 0]], F[pairs[:, 1]]
    share = (Fa[:, :, None] == Fb[:, None, :]).any((1, 2))
    box = (lo[pairs[:, 0]] <= hi[pairs[:, 1]]).all(1) & (lo[pairs[:, 1]] <= hi[pairs[:, 0]]).all(1)
    pairs = pairs[~share & box]
    if len(pairs) > max_pairs:
        raise RuntimeError(f"{len(pairs)} pares candidatos; aumente cell ou reduza n")
    hit = np.zeros(len(pairs), dtype=bool)
    for s in range(0, len(pairs), 200_000):
        p = pairs[s:s + 200_000]
        hit[s:s + 200_000] = _tri_tri(T[p[:, 0]], T[p[:, 1]])
    return pairs[hit]


# ----------------------------------------------------------------------------
# 3. superfícies
# ----------------------------------------------------------------------------
def numpy_surface(args):
    """Devolve (phi_np, domain, label)."""
    if args.surface in ("exact_clifford", "exact_torus"):
        R, r = (math.sqrt(2.0), 1.0) if args.surface == "exact_clifford" else (args.R, args.r)

        def phi(uv):
            u, v = uv[:, 0], uv[:, 1]
            return np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], 1)
        return phi, "torus", f"{args.surface}_R{R:.3g}_r{r:.3g}"
    if args.surface == "exact_sphere":
        def phi(uv):
            u, v = uv[:, 0], uv[:, 1]
            return np.stack([np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)], 1)
        return phi, "sphere", "exact_sphere"
    # daqui para baixo precisa de torch
    import torch
    torch.set_default_dtype(torch.float64)
    import certificates as C
    if args.surface == "knot_init":
        from basins.perturb_init import torus_knot_tube
        t_phi, dom, label = torus_knot_tube(), "torus", "init_knot_2_3"
    elif args.ckpt:
        from methods.published import from_checkpoint
        s = from_checkpoint(args.repo, args.ckpt)
        t_phi, dom, label = s.phi, C.domain_for_genus(s.genus), os.path.basename(os.path.dirname(args.ckpt))
    else:
        raise SystemExit("escolha --surface ou --ckpt")
    if args.bump:
        eps, rho = args.bump
        t_phi = C.graft_bump(t_phi, dom, (math.pi, math.pi), eps, rho)
        label += f"_bump{eps}_{rho}"

    def phi(uv):
        with torch.no_grad():
            return t_phi(torch.as_tensor(uv, dtype=torch.float64)).cpu().numpy()
    return phi, dom, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default=None, help="exact_clifford | exact_torus | exact_sphere | knot_init")
    ap.add_argument("--R", type=float, default=3.0); ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--repo"); ap.add_argument("--ckpt"); ap.add_argument("--genus", type=int, default=1)
    ap.add_argument("--bump", type=float, nargs=2, metavar=("EPS", "RHO"))
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.repo:
        a.repo = os.path.expanduser(a.repo)
    if a.ckpt:
        a.ckpt = os.path.expanduser(a.ckpt)
    phi, dom, label = numpy_surface(a)
    t0 = time.time()
    V, F, uv = grid_mesh(phi, dom, a.n)
    pairs = self_intersections(V, F)
    dt = time.time() - t0
    res = dict(kind="mesh", label=label, domain=dom, grid=a.n, n_vertices=int(len(V)), n_faces=int(len(F)),
               n_intersecting_pairs=int(len(pairs)), self_intersects=bool(len(pairs) > 0), seconds=round(dt, 1))
    if len(pairs):
        where = uv[F[pairs[:, 0]][:, 0]]
        res["examples_uv"] = where[:5].round(3).tolist()
    print(json.dumps(res, indent=2))
    out = a.out or os.path.join(HERE, "runs", f"mesh__{label}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
