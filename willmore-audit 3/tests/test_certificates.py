"""
Testes unitários dos certificados nas superfícies exatas.
Rode:  python -m pytest tests -v        (CPU, ~1–3 min)

Os valores esperados aqui foram reproduzidos independentemente em numpy
(validate_numpy.py) com erro < 1e-9; se um teste falhar no torch, o problema
está no autodiff/φ, não na quadratura.
"""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

import certificates as C

PI = math.pi
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- C0: quadratura
@pytest.mark.parametrize("r", [1.0, 2.5])
def test_sphere_willmore_scale_invariant(r):
    assert abs(C.willmore(C.exact_sphere(r), "sphere") - 4 * PI) < 1e-9


def test_sphere_gauss_bonnet():
    gb = C.gauss_bonnet(C.exact_sphere(), "sphere")
    assert gb["residual"] < 1e-9 and abs(gb["chi_estimate"] - 2) < 1e-9


def test_ellipsoid_gauss_bonnet_and_lower_bound():
    phi = C.exact_ellipsoid(1.0, 1.0, 2.0)
    assert C.gauss_bonnet(phi, "sphere")["residual"] < 1e-9
    W = C.willmore(phi, "sphere")
    assert abs(W - 15.4516066443) < 1e-6   # valor reproduzido em numpy
    assert W > 4 * PI


def test_clifford_willmore_and_gauss_bonnet():
    phi = C.exact_torus()  # R=√2, r=1
    assert abs(C.willmore(phi, "torus") - 2 * PI ** 2) < 1e-9
    assert abs(C.total_curvature(phi, "torus")) < 1e-9


def test_generic_torus_closed_form():
    R, r = 3.0, 1.0
    assert abs(C.willmore(C.exact_torus(R, r), "torus") - C.torus_willmore_exact(R, r)) < 1e-9


# ---------------------------------------------------------------- C2: Möbius
@pytest.mark.parametrize("name,phi,dom", [
    ("esfera", C.exact_sphere(), "sphere"),
    ("elipsoide", C.exact_ellipsoid(1, 1, 2), "sphere"),
    ("clifford", C.exact_torus(), "torus"),
    ("toro R=3", C.exact_torus(3, 1), "torus"),
])
def test_mobius_invariance(name, phi, dom):
    m = C.mobius_test(phi, dom, 128, 128)
    assert m["rel_diff_max"] < 1e-7, name
    for r in m["runs"].values():
        assert abs(r["total_curvature"] - 2 * PI * C.DOMAINS[dom]["chi"]) < 1e-7, name


# ---------------------------------------------------------------- C4: equação de Willmore
def test_willmore_residual_vanishes_on_critical_surfaces():
    for phi, dom in [(C.exact_sphere(), "sphere"), (C.exact_torus(), "torus")]:
        r = C.willmore_residual(phi, dom, 32, 32)
        assert r["residual_relative"] < 1e-6 and r["band"] == "critical"


def test_willmore_residual_nonzero_off_critical():
    r = C.willmore_residual(C.exact_torus(3, 1), "torus", 32, 32)
    assert r["residual_relative"] > 0.5 and r["band"] == "not critical"


def test_variational_fallback_matches_direct_residual():
    # identidade validada em numpy: toro R=3, dW/deps = 4.57985 = ∫ r ξ dA
    var = C.willmore_residual_variational(C.exact_torus(3, 1), "torus", 96, 96, n_tests=1)
    assert abs(var["slopes"][0] - 4.5798) < 2e-3
    var0 = C.willmore_residual_variational(C.exact_torus(), "torus", 96, 96, n_tests=1)
    assert abs(var0["slopes"][0]) < 1e-3


# ---------------------------------------------------------------- C3: Li–Yau
def test_li_yau():
    assert C.li_yau(2 * PI ** 2)["embedded_certified"]
    assert not C.li_yau(8 * PI + 0.1)["embedded_certified"]
    assert C.li_yau(8 * PI + 0.1)["max_multiplicity_bound"] == 2


# ---------------------------------------------------------------- estimador de treino
def test_mc_estimator_matches_quadrature_on_clifford():
    mc = C.willmore_mc(C.exact_torus(), "torus", n=20000, seed=0)
    assert abs(mc["W_mc"] - 2 * PI ** 2) < 0.3   # ruído de Monte-Carlo, não erro sistemático
    assert mc["W_mc_huber"] == pytest.approx(mc["W_mc"])  # H² < 50 no Clifford: Huber inativo


# ---------------------------------------------------------------- ataque
def test_bump_keeps_topology_and_raises_energy():
    phi = C.graft_bump(C.exact_torus(), "torus", (PI, 0.0), eps=0.3, rho=0.5)
    W = C.willmore(phi, "torus", 192, 192)
    assert abs(W - 20.3511) < 1e-3                       # valor reproduzido em numpy
    assert abs(C.total_curvature(phi, "torus", 192, 192)) < 1e-6
    assert W > 2 * PI ** 2


# ---------------------------------------------------------------- relatório
def test_certify_runs_end_to_end():
    c = C.certify(C.exact_torus(), genus=1, nu=64, nv=64, residual_grid=(24, 24), verbose=False)
    assert all(c["verdicts"].values()), c["verdicts"]


def test_estimator_report_on_clifford():
    e = C.estimator_report(C.exact_torus(), "torus", 2 * PI ** 2, seeds=3, n=5000)
    assert e["W_rep_sd"] < 1.0 and not e["huber_active"]
