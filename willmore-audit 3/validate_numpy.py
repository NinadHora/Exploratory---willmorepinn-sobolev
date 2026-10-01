"""
validate_numpy.py — espelho em numpy do certificates.py, para validar a MATEMÁTICA
(quadratura, fórmulas de H/K/dA, Gauss–Bonnet, Möbius, resíduo de Willmore, ataque)
num ambiente sem torch. Derivadas por diferenças finitas centrais de 4ª ordem em
float64 (≈1e-9 de precisão nas segundas derivadas), o suficiente para checar que
as identidades fecham e que a quadratura atinge precisão de máquina nos casos exatos.

Uso:  python validate_numpy.py
"""
import math
import numpy as np

PI = math.pi
DOMAINS = {
    "sphere": dict(u=(0.0, 2 * PI), v=(0.0, PI), periodic=(True, False), chi=2),
    "torus": dict(u=(0.0, 2 * PI), v=(0.0, 2 * PI), periodic=(True, True), chi=0),
}


def _nodes_1d(a, b, n, periodic):
    if periodic:
        h = (b - a) / n
        return a + h * (np.arange(n) + 0.5), np.full(n, h)
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def quadrature_grid(domain, nu, nv):
    d = DOMAINS[domain]
    xu, wu = _nodes_1d(*d["u"], nu, d["periodic"][0])
    xv, wv = _nodes_1d(*d["v"], nv, d["periodic"][1])
    U, V = np.meshgrid(xu, xv, indexing="ij")
    return np.stack([U.ravel(), V.ravel()], 1), np.outer(wu, wv).ravel()


# ---- derivadas por diferenças finitas (4ª ordem) --------------------------
def _d1(f, uv, k, h):
    e = np.zeros(2); e[k] = h
    return (-f(uv + 2 * e) + 8 * f(uv + e) - 8 * f(uv - e) + f(uv - 2 * e)) / (12 * h)


def jacobian(f, uv, h=1e-3):
    return np.stack([_d1(f, uv, 0, h), _d1(f, uv, 1, h)], -1)  # (N,3,2)


def hessian(f, uv, h=1e-3):
    fu = lambda p: _d1(f, p, 0, h)
    fv = lambda p: _d1(f, p, 1, h)
    puu, puv, pvv = _d1(fu, uv, 0, h), _d1(fu, uv, 1, h), _d1(fv, uv, 1, h)
    return puu, puv, pvv


def geometry(f, uv, h=1e-3):
    J = jacobian(f, uv, h)
    pu, pv = J[..., 0], J[..., 1]
    puu, puv, pvv = hessian(f, uv, h)
    E, F, G = (pu * pu).sum(-1), (pu * pv).sum(-1), (pv * pv).sum(-1)
    nrm = np.cross(pu, pv)
    n = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)
    L, M, N = (puu * n).sum(-1), (puv * n).sum(-1), (pvv * n).sum(-1)
    det = E * G - F * F
    H = (E * N - 2 * F * M + G * L) / (2 * det)
    K = (L * N - M * M) / det
    return dict(E=E, F=F, G=G, H=H, K=K, dA=np.sqrt(det), n=n)


def integrate(f, domain, integrand, nu, nv, h=1e-3):
    uv, w = quadrature_grid(domain, nu, nv)
    return float((integrand(geometry(f, uv, h)) * w).sum())


def willmore(f, domain, nu=96, nv=96):
    return integrate(f, domain, lambda g: g["H"] ** 2 * g["dA"], nu, nv)


def total_curvature(f, domain, nu=96, nv=96):
    return integrate(f, domain, lambda g: g["K"] * g["dA"], nu, nv)


def mobius(f, center, radius):
    c = np.asarray(center)[None]

    def g(uv):
        x = f(uv) - c
        return c + radius ** 2 * x / (x * x).sum(1, keepdims=True)
    return g


def willmore_residual_fd(f, domain, nu=32, nv=32, h=2e-2):
    """ΔH + 2H(H²−K) via Laplace–Beltrami (1/√g)∂_i(√g g^{ij}∂_j H). Só para validar em superfícies exatas."""
    def H_at(uv):
        return geometry(f, uv, 1e-3)["H"]

    def flux(uv):
        g = geometry(f, uv, 1e-3)
        gu, gv = _d1(H_at, uv, 0, h), _d1(H_at, uv, 1, h)
        det = g["E"] * g["G"] - g["F"] ** 2
        fu = g["dA"] * (g["G"] * gu - g["F"] * gv) / det
        fv = g["dA"] * (-g["F"] * gu + g["E"] * gv) / det
        return np.stack([fu, fv], -1)

    uv, w = quadrature_grid(domain, nu, nv)
    if not DOMAINS[domain]["periodic"][1]:  # só no espelho FD: estêncil não pode cruzar os polos
        keep = (uv[:, 1] > 4 * h) & (uv[:, 1] < PI - 4 * h)
        uv, w = uv[keep], w[keep]
    g = geometry(f, uv, 1e-3)
    div = _d1(flux, uv, 0, h)[:, 0] + _d1(flux, uv, 1, h)[:, 1]
    lap = div / g["dA"]
    r = lap + 2 * g["H"] * (g["H"] ** 2 - g["K"])
    scale = np.abs(lap) + np.abs(2 * g["H"] * (g["H"] ** 2 - g["K"]))
    wA = w * g["dA"]
    l2 = math.sqrt((r ** 2 * wA).sum() / wA.sum())
    sc = math.sqrt((scale ** 2 * wA).sum() / wA.sum())
    return l2, l2 / max(sc, 1e-300)


def graft_bump(f, domain, uv0, eps, rho):
    d = DOMAINS[domain]
    uv0 = np.asarray(uv0)

    def g(uv):
        n = geometry(f, uv, 1e-3)["n"]
        diff = uv - uv0
        for k in range(2):
            if d["periodic"][k]:
                Lk = 2 * PI
                diff[:, k] = np.remainder(diff[:, k] + Lk / 2, Lk) - Lk / 2
        b = np.exp(-(diff ** 2).sum(1) / rho ** 2)[:, None]
        return f(uv) + eps * b * n
    return g


def willmore_mc(f, domain, n=5000, seed=0, h2_clip=50.0, eps=1e-6):
    d = DOMAINS[domain]
    rng = np.random.default_rng(seed)
    u = d["u"][0] + (d["u"][1] - d["u"][0]) * rng.random(n)
    v = d["v"][0] + (d["v"][1] - d["v"][0]) * rng.random(n)
    g = geometry(f, np.stack([u, v], 1))
    A = (d["u"][1] - d["u"][0]) * (d["v"][1] - d["v"][0])
    dA = np.sqrt(np.maximum(g["E"] * g["G"] - g["F"] ** 2, eps))
    h2 = g["H"] ** 2
    unc = (h2 * dA).mean() * A
    s = h2_clip ** 0.5
    h2h = np.where(h2 <= h2_clip, h2, 2 * s * np.abs(g["H"]) - h2_clip)
    return unc, (h2h * dA).mean() * A


# ---- superfícies exatas -------------------------------------------------------
def sphere(r=1.0):
    return lambda uv: r * np.stack([np.sin(uv[:, 1]) * np.cos(uv[:, 0]), np.sin(uv[:, 1]) * np.sin(uv[:, 0]), np.cos(uv[:, 1])], 1)


def ellipsoid(a, b, c):
    return lambda uv: np.stack([a * np.sin(uv[:, 1]) * np.cos(uv[:, 0]), b * np.sin(uv[:, 1]) * np.sin(uv[:, 0]), c * np.cos(uv[:, 1])], 1)


def torus(R=math.sqrt(2.0), r=1.0):
    return lambda uv: np.stack([(R + r * np.cos(uv[:, 1])) * np.cos(uv[:, 0]), (R + r * np.cos(uv[:, 1])) * np.sin(uv[:, 0]), r * np.sin(uv[:, 1])], 1)


def torus_W(R, r):
    return PI ** 2 * R ** 2 / (r * math.sqrt(R ** 2 - r ** 2))


if __name__ == "__main__":
    np.set_printoptions(precision=12)
    ok = True

    def check(name, got, exp, tol):
        global ok
        err = abs(got - exp)
        flag = err < tol
        ok &= flag
        print(f"  [{'PASS' if flag else 'FAIL'}] {name:50s} {got:+.10f}  (esperado {exp:+.10f}, erro {err:.1e})")

    print("== C0: quadratura e fórmulas nas superfícies exatas")
    check("esfera: W", willmore(sphere(), "sphere"), 4 * PI, 1e-7)
    check("esfera r=2.5: W (invariância de escala)", willmore(sphere(2.5), "sphere"), 4 * PI, 1e-7)
    check("esfera: ∫K dA", total_curvature(sphere(), "sphere"), 4 * PI, 1e-7)
    check("elipsoide (1,1,2): ∫K dA", total_curvature(ellipsoid(1, 1, 2), "sphere"), 4 * PI, 1e-7)
    We = willmore(ellipsoid(1, 1, 2), "sphere")
    flag = We > 4 * PI; ok &= flag
    print(f"  [{'PASS' if flag else 'FAIL'}] {'elipsoide (1,1,2): W > 4π':50s} {We:+.10f}  (= {We/(4*PI):.4f}·4π)")
    check("Clifford: W", willmore(torus(), "torus"), 2 * PI ** 2, 1e-7)
    check("Clifford: ∫K dA", total_curvature(torus(), "torus"), 0.0, 1e-7)
    check("toro R=3,r=1: W (fórmula fechada)", willmore(torus(3, 1), "torus"), torus_W(3, 1), 1e-7)
    check("toro R=3,r=1: ∫K dA", total_curvature(torus(3, 1), "torus"), 0.0, 1e-7)

    print("== C2: invariância de Möbius (centro fora da superfície)")
    for name, f, dom in [("esfera", sphere(), "sphere"), ("elipsoide", ellipsoid(1, 1, 2), "sphere"),
                         ("Clifford", torus(), "torus"), ("toro R=3", torus(3, 1), "torus")]:
        W0 = willmore(f, dom)
        g = mobius(f, [0.0, 0.0, 6.0], 3.0)
        W1 = willmore(g, dom, 128, 128)
        check(f"{name}: W(Ψ∘φ) = W(φ)", W1, W0, 2e-6)
        check(f"{name}: ∫K dA preservado", total_curvature(g, dom, 128, 128), 2 * PI * DOMAINS[dom]["chi"], 2e-6)

    print("== C4: resíduo da equação de Willmore (esfera e Clifford são críticos; R=3 não é)")
    for name, f, dom in [("esfera", sphere(), "sphere"), ("Clifford", torus(), "torus"), ("toro R=3", torus(3, 1), "torus")]:
        l2, rel = willmore_residual_fd(f, dom)
        print(f"        {name:10s} resíduo L2 = {l2:.3e}   relativo = {rel:.3e}")
    print("        (esfera: ΔH=0 e H²=K ⇒ resíduo 0; Clifford: resíduo 0; toro R=3: resíduo O(1))")

    print("== ataque: bolha enxertada no toro de Clifford")
    f = torus()
    print("        (bolha no equador externo v=0; grade 192²; ∫K dA ≈ 0 certifica que a grade resolve a bolha)")
    for eps, rho in [(0.1, 0.5), (0.3, 0.5), (0.2, 0.15)]:
        g = graft_bump(f, "torus", (PI, 0.0), eps, rho)
        Wq = willmore(g, "torus", 192, 192)
        gb = total_curvature(g, "torus", 192, 192)
        Wmc0, Wh0 = willmore_mc(g, "torus", seed=0)
        Wmc1, Wh1 = willmore_mc(g, "torus", seed=1)
        print(f"        ε={eps:.2f} ρ={rho:.2f}:  W_quad={Wq:.4f}  ∫K dA={gb:+.1e}  W_MC(seed0/1)={Wmc0:.3f}/{Wmc1:.3f}  Huber={Wh0:.3f}/{Wh1:.3f}   (Clifford: {2*PI**2:.4f})")

    print("\nTUDO OK" if ok else "\nHÁ FALHAS")
