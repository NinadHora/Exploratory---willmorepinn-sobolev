"""
certificates.py

Recebe qualquer mapa phi: (N,2) -> (N,3) diferenciável em torch e verifica
identidades que toda superfície fechada regular tem de satisfazer.

Blocos, na ordem em que se leem:
  1. domínio e quadratura        onde integrar e com que pesos
  2. geometria num ponto         E F G L M N, H, K, dA, normal, por autodiff
  3. integrais certificadas      W, área, ∫K dA
  4. os quatro certificados      C1 Gauss-Bonnet, C2 Möbius, C3 Li-Yau, C4 equação de Willmore
  5. estimador de treino         réplica do loss do WillmorePINN (Monte Carlo + Huber)
  6. bolha enxertada             o ataque
  7. estudo de resolução         a grade resolve a superfície?
  8. relatório                   tudo junto, com vereditos
  9. superfícies exatas          os testes unitários

Tudo em float64. Só torch e numpy.
"""
import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.func import jacfwd, vmap

PI = math.pi
Phi = Callable[[torch.Tensor], torch.Tensor]   # (N,2) -> (N,3)


# =============================================================================
# 1. DOMÍNIO E QUADRATURA
#    Mesmas convenções do WillmorePINN:
#    gênero 0: u em [0,2π] (azimute, periódico), v em [0,π] (polar, não periódico)
#    gênero 1: u e v em [0,2π], ambos periódicos
# =============================================================================
DOMAINS = {
    "sphere": dict(u=(0.0, 2 * PI), v=(0.0, PI), periodic=(True, False), chi=2, genus=0),
    "torus":  dict(u=(0.0, 2 * PI), v=(0.0, 2 * PI), periodic=(True, True), chi=0, genus=1),
}


def domain_for_genus(genus: int) -> str:
    return {0: "sphere", 1: "torus"}[genus]


def _nodes_1d(a: float, b: float, n: int, periodic: bool):
    """Nós e pesos numa direção.
    Periódica: regra do ponto médio (para função periódica suave converge
    exponencialmente, e os nós nunca caem em 0 ou 2π).
    Não periódica: Gauss-Legendre (os nós nunca tocam os polos)."""
    if periodic:
        h = (b - a) / n
        return a + h * (np.arange(n) + 0.5), np.full(n, h)
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def quadrature_grid(domain: str, nu: int = 96, nv: int = 96) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grade tensorial nu x nv. Retorna uv (N,2) e pesos w (N,) tais que
    sum(w * f(uv)) aproxima ∫∫ f du dv."""
    d = DOMAINS[domain]
    xu, wu = _nodes_1d(*d["u"], nu, d["periodic"][0])
    xv, wv = _nodes_1d(*d["v"], nv, d["periodic"][1])
    U, V = np.meshgrid(xu, xv, indexing="ij")
    uv = torch.tensor(np.stack([U.ravel(), V.ravel()], 1), dtype=torch.float64)
    w = torch.tensor(np.outer(wu, wv).ravel(), dtype=torch.float64)
    return uv, w


# =============================================================================
# 2. GEOMETRIA NUM PONTO
#    Forward-mode (jacfwd), como no WillmorePINN. A função phi recebe lotes,
#    então _single a transforma numa função de um ponto (2,) -> (3,).
# =============================================================================
def _single(phi: Phi):
    return lambda p: phi(p.unsqueeze(0)).squeeze(0)


def _geometry_point(f, p: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Formas fundamentais, H, K, dA e normal num único ponto p = (u, v)."""
    J = jacfwd(f)(p)              # (3,2): colunas phi_u, phi_v
    Hs = jacfwd(jacfwd(f))(p)     # (3,2,2): segundas derivadas
    pu, pv = J[:, 0], J[:, 1]
    E, F, G = pu @ pu, pu @ pv, pv @ pv
    cross = torch.linalg.cross(pu, pv)
    n = cross / torch.linalg.norm(cross)
    L, M, N = Hs[:, 0, 0] @ n, Hs[:, 0, 1] @ n, Hs[:, 1, 1] @ n
    det = E * G - F * F                       # sem clamp: NaN aqui é achado, não bug
    H = (E * N - 2 * F * M + G * L) / (2 * det)
    K = (L * N - M * M) / det
    return dict(E=E, F=F, G=G, H=H, K=K, dA=torch.sqrt(det), n=n)


def geometry(phi: Phi, uv: torch.Tensor) -> Dict[str, torch.Tensor]:
    """O mesmo, vetorizado sobre todos os pontos de uv (N,2)."""
    f = _single(phi)

    def g(p):
        d = _geometry_point(f, p)
        return d["E"], d["F"], d["G"], d["H"], d["K"], d["dA"], d["n"]

    E, F, G, H, K, dA, n = vmap(g)(uv)
    return dict(E=E, F=F, G=G, H=H, K=K, dA=dA, n=n)


# =============================================================================
# 3. INTEGRAIS CERTIFICADAS
# =============================================================================
def integrate(phi: Phi, domain: str, integrand, nu: int = 96, nv: int = 96, chunk: int = 4096) -> float:
    """sum(w * integrand(geometria)) em pedaços, para não estourar memória."""
    uv, w = quadrature_grid(domain, nu, nv)
    total = 0.0
    for i in range(0, uv.shape[0], chunk):
        geo = geometry(phi, uv[i:i + chunk])
        total += float((integrand(geo) * w[i:i + chunk]).sum())
    return total


def willmore(phi, domain, nu=96, nv=96) -> float:
    """W = ∫ H² dA. A energia CERTIFICADA."""
    return integrate(phi, domain, lambda g: g["H"] ** 2 * g["dA"], nu, nv)


def area(phi, domain, nu=96, nv=96) -> float:
    return integrate(phi, domain, lambda g: g["dA"], nu, nv)


def total_curvature(phi, domain, nu=96, nv=96) -> float:
    """∫ K dA. Gauss-Bonnet diz que vale 2πχ = 4π(1 − g)."""
    return integrate(phi, domain, lambda g: g["K"] * g["dA"], nu, nv)


# =============================================================================
# 4. OS QUATRO CERTIFICADOS
# =============================================================================

# ---- C1 Gauss-Bonnet --------------------------------------------------------
def gauss_bonnet(phi, domain, nu=96, nv=96) -> Dict[str, float]:
    tk = total_curvature(phi, domain, nu, nv)
    expected = 2 * PI * DOMAINS[domain]["chi"]
    return dict(total_curvature=tk, chi_estimate=tk / (2 * PI), expected=expected, residual=abs(tk - expected))


# ---- C2 Möbius --------------------------------------------------------------
def mobius_inversion(phi: Phi, center: torch.Tensor, radius: float) -> Phi:
    """Ψ(x) = c + r² (x − c)/|x − c|². A composição Ψ∘phi continua diferenciável,
    então o mesmo autodiff calcula W na superfície invertida."""
    c = center.reshape(1, 3)

    def psi_phi(uv):
        x = phi(uv) - c
        return c + radius ** 2 * x / (x * x).sum(1, keepdim=True)
    return psi_phi


def mobius_test(phi, domain, nu=96, nv=96) -> Dict[str, object]:
    """Dois centros, como manda o protocolo: um fora da caixa envolvente e um no
    centroide. Cada um distorce a superfície de um jeito; dois acertos é mais
    difícil de ser coincidência."""
    uv, _ = quadrature_grid(domain, 64, 64)
    with torch.no_grad():
        x = phi(uv)
    lo, hi = x.min(0).values, x.max(0).values
    ctr = 0.5 * (lo + hi)
    ext = float((hi - lo).max())
    centers = {
        "outside": ctr + torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype) * (0.5 * float(hi[2] - lo[2]) + 1.5 * ext),
        "inside": ctr,
    }
    W0 = willmore(phi, domain, nu, nv)
    out = dict(W=W0, runs={})
    for name, c in centers.items():
        with torch.no_grad():
            r = float(torch.linalg.norm(phi(uv) - c, dim=1).mean())
        psi = mobius_inversion(phi, c, r)
        W1 = willmore(psi, domain, nu, nv)
        out["runs"][name] = dict(W_inverted=W1, rel_diff=abs(W1 - W0) / max(W0, 1e-12),
                                 total_curvature=total_curvature(psi, domain, nu, nv),
                                 center=c.tolist(), radius=r)
    out["rel_diff_max"] = max(r["rel_diff"] for r in out["runs"].values())
    return out


# ---- C3 Li-Yau --------------------------------------------------------------
def li_yau(W: float) -> Dict[str, object]:
    """W ≥ 4π k, k = multiplicidade máxima. Logo W < 8π força mergulho."""
    return dict(W=W, embedded_certified=W < 8 * PI, max_multiplicity_bound=int(math.floor(W / (4 * PI) + 1e-12)))


# ---- C4 equação de Willmore ------------------------------------------------
BANDS = [(0.05, "critical"), (0.5, "near-critical"), (float("inf"), "not critical")]


def band(rel: float) -> str:
    for cut, name in BANDS:
        if rel < cut:
            return name
    return "not critical"


def willmore_residual(phi, domain, nu=32, nv=32, chunk=512) -> Dict[str, object]:
    """r = ΔH + 2H(H² − K), com ΔH = (1/√g) ∂_i(√g g^{ij} ∂_j H).
    Quartas derivadas de phi, tudo por jacfwd aninhado:
      H_at      usa 2ª derivada de phi
      jacfwd(H) usa 3ª
      jacfwd(flux) usa 4ª
    Reportado em L² com peso dA e relativo a max(escala dos termos, H_rms³)."""
    f = _single(phi)

    def H_at(p):
        return _geometry_point(f, p)["H"]

    def flux(p):                              # √g g^{ij} ∂_j H
        d = _geometry_point(f, p)
        E, F, G, dA = d["E"], d["F"], d["G"], d["dA"]
        gradH = jacfwd(H_at)(p)
        ginv = torch.stack([torch.stack([G, -F]), torch.stack([-F, E])]) / (E * G - F * F)
        return dA * (ginv @ gradH)

    def residual_point(p):
        d = _geometry_point(f, p)
        lapH = torch.trace(jacfwd(flux)(p)) / d["dA"]
        r = lapH + 2 * d["H"] * (d["H"] ** 2 - d["K"])
        return r, d["H"], torch.abs(lapH) + torch.abs(2 * d["H"] * (d["H"] ** 2 - d["K"])), d["dA"]

    uv, w = quadrature_grid(domain, nu, nv)
    R, Hs, S, W_ = [], [], [], []
    for i in range(0, uv.shape[0], chunk):
        r, H, s, dA = vmap(residual_point)(uv[i:i + chunk])
        R.append(r); Hs.append(H); S.append(s); W_.append(w[i:i + chunk] * dA)
    r, H, s, w = torch.cat(R), torch.cat(Hs), torch.cat(S), torch.cat(W_)
    l2 = float(torch.sqrt((r ** 2 * w).sum() / w.sum()))
    scale = float(torch.sqrt((s ** 2 * w).sum() / w.sum()))
    h_rms3 = float(torch.sqrt((H ** 2 * w).sum() / w.sum())) ** 3
    rel = l2 / max(scale, h_rms3, 1e-300)
    return dict(residual_L2=l2, residual_relative=rel, band=band(rel))


def willmore_residual_variational(phi, domain, nu=96, nv=96, eps=1e-3, n_tests=4) -> Dict[str, object]:
    """Fallback do C4, só com segundas derivadas.
    Identidade: para phi_ε = phi + ε ξ n,  dW/dε em 0 = ∫ r ξ dA.
    (validada em numpy: toro R=3, ξ = cos(u)sin(2v) + 0.5cos(v), dá 4.57985 pelos dois lados;
    com 0.5cos(2v) no lugar dá −1.16711 pelos dois lados: torch e numpy concordam.)
    Usa n_tests funções-teste ξ de baixa frequência e devolve a raiz da média
    de (dW/dε)², dividida pela norma L² média de ξ: uma estimativa por baixo
    de ||r||_L2 restrita a essas direções."""
    d = DOMAINS[domain]
    f = _single(phi)

    def normal_point(p):
        return _geometry_point(f, p)["n"]

    tests = []
    for a, b in [(1, 2), (2, 1), (1, 1), (2, 2)][:n_tests]:   # (1,2) é a ξ usada na validação numpy
        if d["periodic"][1]:
            tests.append(lambda uv, a=a, b=b: torch.cos(a * uv[:, 0]) * torch.sin(b * uv[:, 1]) + 0.5 * torch.cos(uv[:, 1]))
        else:
            tests.append(lambda uv, a=a, b=b: torch.sin(uv[:, 1]) ** 2 * torch.cos(a * uv[:, 0]) + 0.5 * torch.cos(uv[:, 1]))

    def perturbed(xi, e):
        def g(uv):
            return phi(uv) + e * xi(uv).unsqueeze(1) * vmap(normal_point)(uv)
        return g

    slopes = []
    for xi in tests:
        Wp = willmore(perturbed(xi, eps), domain, nu, nv)
        Wm = willmore(perturbed(xi, -eps), domain, nu, nv)
        slopes.append((Wp - Wm) / (2 * eps))
    # norma L² de cada ξ com peso dA
    uv, w = quadrature_grid(domain, nu, nv)
    dA = geometry(phi, uv)["dA"]
    norms = [float(torch.sqrt((xi(uv) ** 2 * dA * w).sum())) for xi in tests]
    est = math.sqrt(sum(s * s for s in slopes) / len(slopes)) / (sum(norms) / len(norms))
    return dict(slopes=slopes, residual_L2_lower_bound=est)


# =============================================================================
# 5. ESTIMADOR DE TREINO (réplica do EmbeddingWillmoreLoss do WillmorePINN)
#    Monte Carlo uniforme no domínio; √g com piso 1e-6; Huber em H² acima de
#    h2_clip (exato abaixo, linear em |H| acima). Devolve as duas versões.
# =============================================================================
def willmore_mc(phi, domain, n=5000, seed=0, h2_clip: Optional[float] = 50.0, eps=1e-6) -> Dict[str, float]:
    d = DOMAINS[domain]
    g = torch.Generator().manual_seed(seed)
    u = d["u"][0] + (d["u"][1] - d["u"][0]) * torch.rand(n, generator=g, dtype=torch.float64)
    v = d["v"][0] + (d["v"][1] - d["v"][0]) * torch.rand(n, generator=g, dtype=torch.float64)
    geo = geometry(phi, torch.stack([u, v], 1))
    area_dom = (d["u"][1] - d["u"][0]) * (d["v"][1] - d["v"][0])
    dA = torch.sqrt(torch.clamp(geo["E"] * geo["G"] - geo["F"] ** 2, min=eps))
    H = geo["H"]
    h2 = H * H
    uncapped = float((h2 * dA).mean() * area_dom)
    if h2_clip is None:
        return dict(W_mc=uncapped, W_mc_huber=uncapped, n=n, seed=seed)
    s = h2_clip ** 0.5
    h2h = torch.where(h2 <= h2_clip, h2, 2 * s * H.abs() - h2_clip)
    return dict(W_mc=uncapped, W_mc_huber=float((h2h * dA).mean() * area_dom), n=n, seed=seed)


def estimator_report(phi, domain, W_cert: float, seeds: int = 5, n: int = 5000, h2_clip=50.0) -> Dict[str, object]:
    """Protocolo da seção 11: média, desvio, viés e se o viés é significativo
    (|viés| > 2 sd/√seeds); se o Huber está ativo (diferença > 10% de W_cert)."""
    runs = [willmore_mc(phi, domain, n=n, seed=s, h2_clip=h2_clip) for s in range(seeds)]
    rep = [r["W_mc"] for r in runs]
    hub = [r["W_mc_huber"] for r in runs]
    mean = lambda x: sum(x) / len(x)
    sd = lambda x: (sum((v - mean(x)) ** 2 for v in x) / max(len(x) - 1, 1)) ** 0.5
    bias = mean(rep) - W_cert
    return dict(W_rep_mean=mean(rep), W_rep_sd=sd(rep), W_huber_mean=mean(hub), W_huber_sd=sd(hub),
                bias=bias, bias_significant=abs(bias) > 2 * sd(rep) / math.sqrt(seeds),
                huber_active=abs(mean(hub) - mean(rep)) > 0.1 * W_cert, seeds=seeds, n=n)


# =============================================================================
# 6. BOLHA ENXERTADA (o ataque)
#    phi_ε(uv) = phi(uv) + ε exp(−d²/ρ²) n(uv), d = distância periódica a uv0.
# =============================================================================
def graft_bump(phi: Phi, domain: str, uv0, eps: float, rho: float) -> Phi:
    d = DOMAINS[domain]
    f = _single(phi)
    uv0_t = torch.tensor(uv0, dtype=torch.float64)

    def normal_point(p):
        return _geometry_point(f, p)["n"]

    def dist2(uv):
        diff = uv - uv0_t
        total = 0.0
        for k, key in enumerate(("u", "v")):
            dk = diff[:, k]
            if d["periodic"][k]:
                Lk = d[key][1] - d[key][0]
                dk = torch.remainder(dk + Lk / 2, Lk) - Lk / 2
            total = total + dk ** 2
        return total

    def phi_eps(uv):
        return phi(uv) + eps * torch.exp(-dist2(uv) / rho ** 2).unsqueeze(1) * vmap(normal_point)(uv)
    return phi_eps


# =============================================================================
# 7. ESTUDO DE RESOLUÇÃO (seção 10 do protocolo)
#    Para em: |∫K dA − 2πχ| < tol_gb  E  |ΔW| < tol_w · W.
#    Não convergiu até a última grade: reporta como não resolvido.
# =============================================================================
def resolution_study(phi, domain, grids=(48, 96, 192, 384), tol_gb=1e-6, tol_w=1e-6, verbose=True):
    rows, prev = [], None
    for n in grids:
        W = willmore(phi, domain, n, n)
        gb_res = abs(total_curvature(phi, domain, n, n) - 2 * PI * DOMAINS[domain]["chi"])
        dW = None if prev is None else abs(W - prev)
        rows.append(dict(n=n, W=W, gb_residual=gb_res, dW=dW))
        if verbose:
            print(f"  grade {n:4d}²  W = {W:.8f}   |∫K dA − 2πχ| = {gb_res:.2e}" + ("" if dW is None else f"   ΔW = {dW:.2e}"))
        if gb_res < tol_gb and dW is not None and dW < tol_w * max(1.0, W):
            return dict(converged=True, n=n, W=W, rows=rows)
        prev = W
    return dict(converged=False, n=grids[-1], W=rows[-1]["W"], rows=rows)


# =============================================================================
# 8. RELATÓRIO
# =============================================================================
KNOWN_MINIMA = {0: ("round sphere", 4 * PI), 1: ("Clifford torus", 2 * PI ** 2)}
TOL_GB = 1e-3 * 2 * PI      # seção 4
TOL_MOBIUS = 1e-3           # seção 5


def certify(phi, genus, nu=96, nv=96, residual=True, residual_grid=(32, 32), verbose=True) -> Dict[str, object]:
    domain = domain_for_genus(genus)
    c = dict(domain=domain, genus=genus, grid=(nu, nv))
    c["W"] = willmore(phi, domain, nu, nv)
    c["area"] = area(phi, domain, nu, nv)
    c["C1"] = gauss_bonnet(phi, domain, nu, nv)
    c["C2"] = mobius_test(phi, domain, nu, nv)
    c["C3"] = li_yau(c["W"])
    c["C4"] = willmore_residual(phi, domain, *residual_grid) if residual else None
    c["verdicts"] = {
        "C1 Gauss-Bonnet": c["C1"]["residual"] < TOL_GB,
        "C2 Mobius (both centres)": c["C2"]["rel_diff_max"] < TOL_MOBIUS,
        "C3 Li-Yau (W < 8π)": c["C3"]["embedded_certified"],
        "C5 lower bound": c["W"] >= KNOWN_MINIMA[genus][1] * (1 - 1e-9),
    }
    if c["C4"]:
        c["verdicts"]["C4 Willmore equation (critical band)"] = c["C4"]["band"] == "critical"
    if verbose:
        print(report(c))
    return c


def report(c: Dict[str, object]) -> str:
    name, wmin = KNOWN_MINIMA[c["genus"]]
    lines = [
        f"domain={c['domain']}  genus={c['genus']}  grid={c['grid'][0]}x{c['grid'][1]}",
        f"W certified            = {c['W']:.10f}   [{name}: {wmin:.10f};  W/W_min = {c['W'] / wmin:.6f}]",
        f"area                   = {c['area']:.10f}",
        f"∫K dA                  = {c['C1']['total_curvature']:+.10f}   (expected {c['C1']['expected']:+.10f};  chi_hat = {c['C1']['chi_estimate']:.6f})",
    ]
    for k, r in c["C2"]["runs"].items():
        lines.append(f"W after inversion ({k:7s}) = {r['W_inverted']:.10f}   rel diff {r['rel_diff']:.2e}")
    lines.append(f"Li-Yau: multiplicity ≤ {c['C3']['max_multiplicity_bound']}   embedded certified = {c['C3']['embedded_certified']}")
    if c["C4"]:
        lines.append(f"Willmore residual L2 = {c['C4']['residual_L2']:.3e}   relative = {c['C4']['residual_relative']:.3e}   band = {c['C4']['band']}")
    lines.append("--- verdicts ---")
    for k, ok in c["verdicts"].items():
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    return "\n".join(lines)


# =============================================================================
# 9. SUPERFÍCIES EXATAS (testes unitários), nas convenções de domínio acima
# =============================================================================
def exact_sphere(r: float = 1.0) -> Phi:
    def phi(uv):
        u, v = uv[:, 0], uv[:, 1]
        return r * torch.stack([torch.sin(v) * torch.cos(u), torch.sin(v) * torch.sin(u), torch.cos(v)], 1)
    return phi


def exact_ellipsoid(a: float, b: float, c: float) -> Phi:
    def phi(uv):
        u, v = uv[:, 0], uv[:, 1]
        return torch.stack([a * torch.sin(v) * torch.cos(u), b * torch.sin(v) * torch.sin(u), c * torch.cos(v)], 1)
    return phi


def exact_torus(R: float = math.sqrt(2.0), r: float = 1.0) -> Phi:
    """R = √2, r = 1 é o toro de Clifford projetado: W = 2π²."""
    def phi(uv):
        u, v = uv[:, 0], uv[:, 1]
        return torch.stack([(R + r * torch.cos(v)) * torch.cos(u), (R + r * torch.cos(v)) * torch.sin(u), r * torch.sin(v)], 1)
    return phi


def torus_willmore_exact(R: float, r: float) -> float:
    """W do toro de revolução: π² R² / (r √(R² − r²))."""
    return PI ** 2 * R ** 2 / (r * math.sqrt(R ** 2 - r ** 2))
