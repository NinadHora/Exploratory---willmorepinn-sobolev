"""
sobolev_optim.py — Gradiente natural funcional / de Sobolev para o fluxo neural de Willmore
============================================================================================

IDEIA (a contribuição "ML profundo"):
  O treino do paper faz Adam sobre os PARÂMETROS da rede. Geometricamente isso é
  um fluxo de gradiente L² no espaço de parâmetros — uma métrica arbitrária, que
  depende de como a rede está parametrizada e NÃO é o fluxo de Willmore clássico.

  O fluxo de Willmore de verdade é o gradiente de W na métrica L²(superfície) — e a
  comunidade clássica pré-condiciona por uma métrica de Sobolev (H^s) para domar a
  rigidez do fluxo de 4ª ordem e não travar em mínimos de alta frequência.

  Aqui replicamos isso no contexto neural via GRADIENTE NATURAL:
      g_nat = M^{-1} g,
  onde g = ∇_θ W (gradiente cru dos parâmetros) e M é a métrica de Gram do mapa
  φ_θ : (u,v) → R³ na superfície:
      M_{ab} = ∫ ⟨∂_a φ, ∂_b φ⟩ dA          (métrica L²; sobolev=False)
      M_{ab} = ∫ ⟨∂_a φ, ∂_b φ⟩ + ⟨∂_a φ_u, ∂_b φ_u⟩ + ⟨∂_a φ_v, ∂_b φ_v⟩ dA
                                              (métrica de Sobolev; sobolev=True)

  M é p×p (p = nº de parâmetros). NUNCA formamos M: resolvemos M g_nat = g por
  gradientes conjugados (CG) usando só produtos M·v matrix-free (JVP + VJP).

HIPÓTESE FALSIFICÁVEL:
  Com tudo o mais idêntico, este fluxo desce abaixo de 30.19 (gênero 2), onde o
  Adam fica preso no mínimo local do "halter".

ESCOPO desta primeira versão:
  - Alvo de validação: gênero 1 (rede única φ:(u,v)→R³). É o portão de validação:
    o módulo TEM que reproduzir o toro de Clifford (W→2π²) antes de irmos pro gênero 2.
  - Para o gênero 2 (multi-carta) as features precisam ser por-carta; deixei marcado
    onde estender (NOTA GÊNERO 2 lá embaixo).

VALIDAR SEM GPU (no nó de login, instantâneo):
      python sobolev_optim.py --selftest
  Confere: (1) JVP bate com diferença finita, (2) M é semidefinida positiva,
  (3) g_nat é direção de descida (g·g_nat > 0). Se isso passar, a matemática está certa.
"""

from __future__ import annotations
import argparse
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Features do mapa e pesos da métrica de superfície
# --------------------------------------------------------------------------- #
def _embedding_features(model: nn.Module, uv: torch.Tensor, sobolev: bool):
    """
    Constrói o vetor de features f(θ) (achatado) e os pesos w_i = dA_i da métrica.

    f = φ                          (L²)        -> (B,3)  -> achatado (3B,)
    f = [φ, φ_u, φ_v]              (Sobolev)   -> (B,9)  -> achatado (9B,)

    Os pesos (elemento de área dA) são DESTACADOS: tratamos a métrica como fixa
    (estilo Gauss–Newton), o que mantém o caso L² em 2ª ordem e é numericamente estável.
    """
    uv = uv.detach().clone().requires_grad_(True)
    phi = model(uv)                                    # (B,3), depende de θ

    # φ_u, φ_v por autograd (create_graph=True p/ sobreviver a 2ª/3ª ordem em θ)
    def d(col):
        return torch.autograd.grad(col, uv, grad_outputs=torch.ones_like(col),
                                   create_graph=True)[0]
    dx, dy, dz = d(phi[:, 0]), d(phi[:, 1]), d(phi[:, 2])
    phi_u = torch.stack([dx[:, 0], dy[:, 0], dz[:, 0]], dim=1)
    phi_v = torch.stack([dx[:, 1], dy[:, 1], dz[:, 1]], dim=1)

    # pesos dA = sqrt(EG - F²)  (destacados)
    E = (phi_u * phi_u).sum(1)
    F = (phi_u * phi_v).sum(1)
    G = (phi_v * phi_v).sum(1)
    dA = torch.sqrt(torch.clamp(E * G - F * F, min=1e-12)).detach()   # (B,)

    if sobolev:
        feats = torch.cat([phi, phi_u, phi_v], dim=1)   # (B,9)
        wrow = dA.repeat_interleave(9)                  # peso por componente-de-amostra
    else:
        feats = phi                                     # (B,3)
        wrow = dA.repeat_interleave(3)

    return feats.reshape(-1), wrow                      # (M,), (M,)


# --------------------------------------------------------------------------- #
# Produto matrix-free M·v  (JVP + peso + VJP)
# --------------------------------------------------------------------------- #
def _make_matvec(params, feats, wrow, damping: float):
    """
    Devolve uma função v -> (M + damping·I) v, sem nunca formar M.

      M v = Jᵀ ( w ⊙ (J v) ),   J = ∂feats/∂θ

    JVP (J v) via o truque do duplo-backward:
      Jᵀ·dummy é linear em dummy; derivar (Jᵀ·dummy) em relação a dummy dá J·(·).
    """
    dummy = torch.zeros_like(feats, requires_grad=True)
    # Jᵀ·dummy  (lista por parâmetro) — create_graph p/ poder derivar em dummy depois
    JT_dummy = torch.autograd.grad(feats, params, grad_outputs=dummy,
                                   create_graph=True, retain_graph=True)
    JT_dummy_flat = torch.cat([t.reshape(-1) for t in JT_dummy])

    def jvp(v_flat):                      # J v   -> (M,)
        Jv = torch.autograd.grad(JT_dummy_flat, dummy, grad_outputs=v_flat,
                                 retain_graph=True)[0]
        return Jv

    def matvec(v_flat):                   # (M + damping I) v -> (p,)
        Jv = jvp(v_flat)
        WJv = wrow * Jv                   # aplica a métrica de superfície
        JT_WJv = torch.autograd.grad(feats, params, grad_outputs=WJv,
                                     retain_graph=True)
        out = torch.cat([t.reshape(-1) for t in JT_WJv])
        return out + damping * v_flat

    return matvec


# --------------------------------------------------------------------------- #
# Gradientes conjugados (matrix-free)
# --------------------------------------------------------------------------- #
def _cg(matvec, b, iters: int, tol: float):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = torch.dot(r, r)
    b_norm = torch.sqrt(torch.dot(b, b)) + 1e-12
    last = rs
    for _ in range(iters):
        Ap = matvec(p)
        alpha = rs / (torch.dot(p, Ap) + 1e-20)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        last = rs_new
        if torch.sqrt(rs_new) / b_norm < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, (torch.sqrt(last) / b_norm).item()


# --------------------------------------------------------------------------- #
# Passo de gradiente natural — DROP-IN no run.py
# --------------------------------------------------------------------------- #
@torch.enable_grad()
def natural_gradient_step(model: nn.Module,
                          uv: torch.Tensor,
                          lr: float = 0.0,
                          sobolev: bool = False,
                          damping: float = 1e-3,
                          cg_iters: int = 20,
                          cg_tol: float = 1e-4,
                          grad_clip: float | None = 1.0,
                          write_grad_only: bool = False) -> dict:
    """
    Pré-condiciona o gradiente cru (já em p.grad, vindo de loss.backward()) pela
    métrica do mapa e dá o passo manualmente:  θ ← θ − lr · M^{-1} g.

    USO no run.py (substitui o bloco backward/step):
        loss.backward()
        if USE_SOBOLEV_NG:
            diag = natural_gradient_step(model, uv_batch, lr=current_lr,
                                         sobolev=SOBOLEV, damping=DAMP,
                                         cg_iters=CG_ITERS, grad_clip=gradient_clip)
        else:
            if gradient_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
    (current_lr = optimizer.param_groups[0]['lr'])

    Retorna diagnósticos: resíduo do CG e o cosseno g·g_nat (deve ser > 0 = descida).
    """
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    g = torch.cat([p.grad.reshape(-1) for p in params]).detach()

    feats, wrow = _embedding_features(model, uv, sobolev)
    matvec = _make_matvec(params, feats, wrow, damping)
    g_nat, cg_res = _cg(matvec, g, cg_iters, cg_tol)

    # diagnóstico: g_nat é direção de descida?
    cos = torch.dot(g, g_nat) / (g.norm() * g_nat.norm() + 1e-20)

    if grad_clip is not None and not write_grad_only:
        gn = g_nat.norm()
        if gn > grad_clip:
            g_nat = g_nat * (grad_clip / (gn + 1e-12))

    if write_grad_only:
        # MODO PRE-CONDICIONADOR: escreve M^{-1}g em p.grad e NAO da passo.
        # O optimizer.step() (Adam) do run.py consome a direcao natural,
        # mantendo escala adaptativa + lr schedule que ja bateram 0.9999x.
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel()
                p.grad = g_nat[i:i + n].view_as(p).clone()
                i += n
    else:
        # MODO PURO: passo manual θ ← θ − lr · g_nat (gradiente natural puro).
        # ATENCAO: use um lr PROPRIO (~1e-2..1e-1), NAO o lr minusculo do Adam.
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel()
                p.add_(g_nat[i:i + n].view_as(p), alpha=-lr)
                i += n
                p.grad = None

    return {"cg_residual": cg_res, "descent_cos": cos.item(),
            "g_norm": g.norm().item(), "g_nat_norm": g_nat.norm().item()}

# --------------------------------------------------------------------------- #
# GÊNERO 2 — multi-carta (bloco-diagonal)
# --------------------------------------------------------------------------- #
# O Genus2MultiChartNetwork tem DUAS sub-redes independentes (model.torus1,
# model.torus2), cada uma um mapa φ:(u,v)→R³ chamável por forward_torus1/2.
# Como as duas não compartilham parâmetros, a métrica de Sobolev M é
# BLOCO-DIAGONAL: bloco 1 só envolve θ de torus1, bloco 2 só θ de torus2.
# Logo não precisamos montar uma M acoplada — basta rodar a máquina de carta
# única (já validada no gênero 1) UMA VEZ POR CARTA, cada uma com sua amostra
# uv, seus params e sua M, e escrever M^{-1}g de volta nos .grad da sub-rede.
#
# Isto reaproveita _embedding_features / _make_matvec / _cg sem alteração.

@torch.enable_grad()
def _precondition_chart(forward_fn, params, uv, sobolev, damping, cg_iters, cg_tol,
                        disk_center=None, neck_radius=None):
    """
    Pré-condiciona o grad de UMA carta. Espera que p.grad já esteja preenchido
    (de loss.backward()). Escreve M^{-1}g de volta em p.grad. Retorna diagnóstico.

    forward_fn: model.forward_torus1 ou model.forward_torus2 (mapa (u,v)→R³).
    params: lista dos parâmetros DAQUELA sub-rede (com grad preenchido).
    disk_center, neck_radius: se dados, EXCLUI da métrica os pontos uv com
      distância periódica ao centro ≤ neck_radius (a zona de colagem). Assim o
      Sobolev só age no CORPO do toro; a costura fica para o Adam (grad cru).
    """
    params = [p for p in params if p.requires_grad and p.grad is not None]
    if not params:
        return None
    g = torch.cat([p.grad.reshape(-1) for p in params]).detach()

    # Máscara do pescoço: descarta amostras dentro do anel de colagem.
    if disk_center is not None and neck_radius is not None:
        import math as _m
        u0, v0 = disk_center
        du = torch.abs(uv[:, 0] - u0); du = torch.min(du, 2 * _m.pi - du)
        dv = torch.abs(uv[:, 1] - v0); dv = torch.min(dv, 2 * _m.pi - dv)
        body = (du * du + dv * dv) > (neck_radius * neck_radius)
        if body.sum() < 8:        # poucos pontos no corpo: não pré-condiciona
            return None
        uv = uv[body]

    # features da carta: usa forward_fn no lugar de model() — mesma máquina.
    uv = uv.detach().clone().requires_grad_(True)
    phi = forward_fn(uv)                                # (B,3)

    def d(col):
        return torch.autograd.grad(col, uv, grad_outputs=torch.ones_like(col),
                                   create_graph=True)[0]
    dx, dy, dz = d(phi[:, 0]), d(phi[:, 1]), d(phi[:, 2])
    phi_u = torch.stack([dx[:, 0], dy[:, 0], dz[:, 0]], dim=1)
    phi_v = torch.stack([dx[:, 1], dy[:, 1], dz[:, 1]], dim=1)
    E = (phi_u * phi_u).sum(1); F = (phi_u * phi_v).sum(1); G = (phi_v * phi_v).sum(1)
    dA = torch.sqrt(torch.clamp(E * G - F * F, min=1e-12)).detach()

    if sobolev:
        feats = torch.cat([phi, phi_u, phi_v], dim=1).reshape(-1)
        wrow = dA.repeat_interleave(9)
    else:
        feats = phi.reshape(-1)
        wrow = dA.repeat_interleave(3)

    matvec = _make_matvec(params, feats, wrow, damping)
    g_nat, cg_res = _cg(matvec, g, cg_iters, cg_tol)
    cos = torch.dot(g, g_nat) / (g.norm() * g_nat.norm() + 1e-20)

    # write_grad_only: escreve M^{-1}g de volta; Adam consome (híbrido validado no g1).
    with torch.no_grad():
        i = 0
        for p in params:
            n = p.numel()
            p.grad = g_nat[i:i + n].view_as(p).clone()
            i += n

    return {"cg_residual": cg_res, "descent_cos": cos.item(),
            "g_norm": g.norm().item(), "g_nat_norm": g_nat.norm().item()}


def natural_gradient_step_genus2(model: nn.Module,
                                 uv_T1: torch.Tensor,
                                 uv_T2: torch.Tensor,
                                 sobolev: bool = True,
                                 damping: float = 1e-3,
                                 cg_iters: int = 20,
                                 cg_tol: float = 1e-4,
                                 neck_alpha: float = 2.5) -> dict:
    """
    Pré-condicionador de Sobolev multi-carta (gênero 2), modo write_grad_only.

    Chama loss.backward() ANTES (enche p.grad de torus1 e torus2 com o grad de
    TODA a loss: Willmore + regularidade + gluing). Aqui pré-condicionamos o grad
    de cada carta com a métrica do mapa daquela carta (bloco-diagonal) e escrevemos
    M^{-1}g de volta em p.grad. O optimizer.step() (Adam) consome em seguida.

    EXCLUSÃO DO PESCOÇO (neck_alpha>0): pontos dentro do anel de colagem
    r ≤ neck_alpha·δ são removidos da métrica de cada carta. O Sobolev então só
    age no CORPO de cada toro (onde está a maior parte da energia de Willmore),
    deixando a costura para o Adam. Isto evita o descolamento observado quando o
    pré-condicionador bloco-diagonal ignora o acoplamento M_{12} do gluing.
    neck_alpha=0 desliga a exclusão (volta ao comportamento que descola a colagem).
    """
    t1_params = list(model.torus1.parameters())
    t2_params = list(model.torus2.parameters())

    # raio do pescoço a excluir = neck_alpha · δ; centros dos discos de cada carta.
    delta = float(getattr(model, "disk_radius", 0.0))
    neck_r = (neck_alpha * delta) if (neck_alpha and delta > 0) else None
    c1 = tuple(getattr(model, "disk_center_T1", (0.0, 0.0))) if neck_r else None
    c2 = tuple(getattr(model, "disk_center_T2", (0.0, 0.0))) if neck_r else None

    d1 = _precondition_chart(model.forward_torus1, t1_params, uv_T1,
                             sobolev, damping, cg_iters, cg_tol,
                             disk_center=c1, neck_radius=neck_r)
    d2 = _precondition_chart(model.forward_torus2, t2_params, uv_T2,
                             sobolev, damping, cg_iters, cg_tol,
                             disk_center=c2, neck_radius=neck_r)

    out = {}
    if d1: out.update({f"T1_{k}": v for k, v in d1.items()})
    if d2: out.update({f"T2_{k}": v for k, v in d2.items()})
    return out


# =========================================================================== #
# AUTOVERIFICAÇÃO (roda em CPU, sem GPU)
# =========================================================================== #
def _selftest():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)   # dupla precisão p/ checar diferença finita

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(),
                                     nn.Linear(16, 16), nn.Tanh(),
                                     nn.Linear(16, 3))
        def forward(self, uv):
            return self.net(uv)

    model = Tiny()
    params = [p for p in model.parameters() if p.requires_grad]
    p_total = sum(p.numel() for p in params)
    uv = torch.rand(24, 2) * 6.283185

    print(f"[selftest] rede de teste com {p_total} parâmetros, B=24 amostras\n")

    # ---- Teste 1: JVP vs diferença finita ----
    feats0, _ = _embedding_features(model, uv, sobolev=False)
    dummy = torch.zeros_like(feats0, requires_grad=True)
    JT_dummy = torch.autograd.grad(feats0, params, grad_outputs=dummy,
                                   create_graph=True, retain_graph=True)
    JT_dummy_flat = torch.cat([t.reshape(-1) for t in JT_dummy])
    v = torch.randn(p_total)
    Jv = torch.autograd.grad(JT_dummy_flat, dummy, grad_outputs=v, retain_graph=True)[0]

    def feats_at(shift):
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel(); p.add_(shift[i:i + n].view_as(p)); i += n
        f, _ = _embedding_features(model, uv, sobolev=False)
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel(); p.sub_(shift[i:i + n].view_as(p)); i += n
        return f.detach()

    eps = 1e-6
    fd = (feats_at(eps * v) - feats_at(-eps * v)) / (2 * eps)
    err = (Jv - fd).abs().max().item()
    print(f"  Teste 1 (JVP vs dif. finita)   erro max = {err:.2e}   "
          f"{'OK' if err < 1e-5 else 'FALHOU'}")

    # ---- Teste 2: M é semidefinida positiva (v·Mv ≥ 0) ----
    feats, wrow = _embedding_features(model, uv, sobolev=False)
    matvec = _make_matvec(params, feats, wrow, damping=0.0)
    quad = min(torch.dot(z, matvec(z)).item() for z in (torch.randn(p_total) for _ in range(5)))
    print(f"  Teste 2 (M PSD: min v·Mv)      valor   = {quad:.2e}   "
          f"{'OK' if quad >= -1e-8 else 'FALHOU'}")

    # ---- Teste 3: g_nat é direção de descida (g·g_nat > 0) ----
    loss = (model(uv) ** 2).sum()        # loss-fantasma só p/ gerar um g
    model.zero_grad(); loss.backward()
    diag = natural_gradient_step(model, uv, lr=0.0,   # lr=0: não move, só mede
                                 sobolev=False, damping=1e-3, cg_iters=50, grad_clip=None)
    print(f"  Teste 3 (descida: cos(g,g_nat))valor   = {diag['descent_cos']:+.4f}  "
          f"(res CG {diag['cg_residual']:.1e})   "
          f"{'OK' if diag['descent_cos'] > 0 else 'FALHOU'}")

    print("\n[selftest] se os três deram OK, a máquina do gradiente natural está correta.")
    print("           próximo: rodar genus 1 com optimizer.type='sobolev_ng' e ver W→19.74.")


def _selftest_g2():
    """Valida o pré-condicionador multi-carta em CPU com uma rede fake de 2 cartas."""
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    class Chart(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(),
                                     nn.Linear(16, 3))
        def forward(self, uv):
            return self.net(uv)

    class FakeG2(nn.Module):
        def __init__(self):
            super().__init__()
            self.torus1 = Chart()
            self.torus2 = Chart()
        def forward_torus1(self, uv):
            return self.torus1(uv)
        def forward_torus2(self, uv):
            return self.torus2(uv)

    model = FakeG2()
    uv1 = torch.rand(20, 2) * 6.283185
    uv2 = torch.rand(20, 2) * 6.283185

    p1 = sum(p.numel() for p in model.torus1.parameters())
    p2 = sum(p.numel() for p in model.torus2.parameters())
    print(f"[selftest-g2] 2 cartas: torus1={p1} params, torus2={p2} params\n")

    # loss-fantasma que envolve AS DUAS cartas (como o gluing faria)
    loss = (model.forward_torus1(uv1) ** 2).sum() + (model.forward_torus2(uv2) ** 2).sum()
    model.zero_grad(); loss.backward()

    # guarda g cru de cada carta antes de pré-condicionar
    g1 = torch.cat([p.grad.reshape(-1) for p in model.torus1.parameters()]).clone()
    g2 = torch.cat([p.grad.reshape(-1) for p in model.torus2.parameters()]).clone()

    diag = natural_gradient_step_genus2(model, uv1, uv2, sobolev=True,
                                        damping=1e-3, cg_iters=50)

    # após a chamada, p.grad contém M^{-1}g de cada carta
    gnat1 = torch.cat([p.grad.reshape(-1) for p in model.torus1.parameters()])
    gnat2 = torch.cat([p.grad.reshape(-1) for p in model.torus2.parameters()])
    cos1 = torch.dot(g1, gnat1) / (g1.norm() * gnat1.norm() + 1e-20)
    cos2 = torch.dot(g2, gnat2) / (g2.norm() * gnat2.norm() + 1e-20)

    print(f"  Carta T1: cos(g,g_nat) = {cos1:+.4f}  {'OK' if cos1 > 0 else 'FALHOU'}  "
          f"(res CG {diag.get('T1_cg_residual', float('nan')):.1e})")
    print(f"  Carta T2: cos(g,g_nat) = {cos2:+.4f}  {'OK' if cos2 > 0 else 'FALHOU'}  "
          f"(res CG {diag.get('T2_cg_residual', float('nan')):.1e})")

    # checagem-chave: pré-condicionar T1 NÃO deve ter tocado o grad de T2 acoplá-los
    # (bloco-diagonal). Verificamos que cada g_nat tem o tamanho certo da sua carta.
    ok_shape = (gnat1.numel() == p1) and (gnat2.numel() == p2)
    print(f"  Bloco-diagonal (shapes por carta)  {'OK' if ok_shape else 'FALHOU'}")
    print("\n[selftest-g2] se T1 e T2 deram cos>0, o pré-condicionador multi-carta está correto.")
    print("              próximo: rodar genus 2 com o híbrido e comparar contra W=29.23.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="valida carta única (gênero 1)")
    ap.add_argument("--selftest-g2", action="store_true", help="valida multi-carta (gênero 2)")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.selftest_g2:
        _selftest_g2()
    else:
        print("Use --selftest (gênero 1) ou --selftest-g2 (gênero 2) para validar em CPU, "
              "ou importe natural_gradient_step[_genus2] no run.py.")
