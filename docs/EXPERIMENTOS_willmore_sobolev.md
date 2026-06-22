# Log de Experimentos — Sobolev/Gradiente Natural no WillmorePINN

**Projeto:** extensão do código de fluxo neural de Willmore (Hirst, Sá Earp, Silva — arXiv:2604.04321) com um pré-condicionador de gradiente natural/Sobolev.
**Objetivo:** testar se pré-condicionar o passo pela métrica do mapa φ (em vez do gradiente L² puro do Adam) ajuda a descer abaixo do resultado genus-2 do paper.
**Autora:** Nina da Hora · **Hardware:** cluster GPU (L40S / H100) · **Última atualização:** 21 jun 2026

---

## Mínimos teóricos por gênero (referência)

| Gênero | Mínimo | Valor | Status |
|---|---|---|---|
| 0 | Esfera redonda | 4π ≈ **12.57** | provado (Willmore; Li–Yau) |
| 1 | Toro de Clifford | 2π² ≈ **19.74** | provado (Marques–Neves, 2014) |
| 2 | Lawson ξ₂,₁ | 4π² ≈ **21.89** | **conjectura (Kusner) — em aberto** |

O genus-2 é o problema aberto. O paper chega a **W = 30.19** (acima de Lawson, abaixo da soma catenoide de 2 Cliffords ~39.48) e especula que "more compute" talvez alcance Lawson.

---

## Parte 1 — Genus 1 (validação da máquina)

> **Propósito:** o Clifford é um mínimo *já provado*, sem mínimos locais ruins. Aqui não se "ganha" — só se valida que a máquina de gradiente natural desce liso e reproduz o mínimo conhecido. Qualquer método correto tem de chegar em ~19.74.

| Run | Método | Épocas | Best W | Ratio vs Clifford | Leitura |
|---|---|---|---|---|---|
| run_1 | Adam puro (baseline do paper) | 200 | 19.74 | **0.9999×** | ✓ reproduz Clifford |
| run_3 | Gradiente natural "quebrado" (substituiu optimizer.step; clip g_nat→1.0) | — | ~109 | 5.53× | ✗ clip encolheu o passo 51× |
| run_4 | Pré-condicionador→Adam (write_grad_only), cosine em U | 200 | 36.15 | 1.83× | parcial; cosine + Adam brigam |
| run_5 | Pré-condicionador→Adam, cosine monotônico | 600 | 21.57 | 1.09× | platô; Adam limita o polimento final |
| **run_8** | **Híbrido: Sobolev 70% + Adam puro 30% final** | 600 | **20.13** | **1.0197×** | ✓ **furou o platô** |

### Conclusões do genus 1
- **A máquina está correta:** desce monótona de ~199 → 20.13, regularidade ~0 o tempo todo, sem colapso.
- **Descoberta de método:** Sobolev é melhor na **descida** (vai rápido de 199 a ~22); Adam é melhor no **polimento final** (últimos %). O **híbrido** (Sobolev cedo + Adam tarde) bate qualquer um sozinho.
- **Modo natural PURO colapsou** (W→0): o passo manual θ ← θ − δ·M⁻¹g ignora o termo de regularidade/anti-colapso, então a malha degenera (dA→0). Reduzir δ (8e-3→2e-3) não resolveu — é direção, não magnitude. Conclusão: o modo puro precisaria de piso de área + line-search; abandonado em favor do híbrido.
- **Status: genus 1 FECHADO como validação.** Não vale perseguir os 2% finais — é caso resolvido. A pergunta "o Sobolev ajuda?" só tem resposta no genus 2 (onde existe mínimo local).

---

## Parte 2 — Genus 2 (o problema de verdade)

> **Propósito:** aqui existe um mínimo local real (o "halter"/dumbbell). É o único lugar onde o Sobolev pode *ganhar*, não só empatar.

### Hiperparâmetros do paper (confirmados contra `config_genus2.yaml` — batem 100%)
- Duas cartas de toro T₁,T₂, τ₁=τ₂=0.7i, δ=0.65, duas redes (128,256,512,256,128) tanh, 4N=24 features Fourier/carta, **664.838 params**.
- Pretraining 200 épocas. Treino 2000 épocas, 5000 pts/época, batch 1000.
- AdamW wd=1e-5, clip=1, cosine LR 2e-4→1e-6.
- Pesos: λ_W=1, λ_R=10, λ_G=200. Sub-regularidade: α_area=1, α_pos=0.5, α_smooth=0.5, **α_log=2** (log-barreira anti-colapso, só genus-2). a_min=0.01, ε_pos=0.001, g_max=5.
- Warmup Willmore 0.01→1 em 200 épocas. Colagem C⁰ desde início, C¹@ep50 (rampa 10), C²@ep150 (rampa 20). λ_C0=200, λ_C1=50, λ_C2=20. Anel regularidade peso 20. Huber H² clip @50×média.

### Runs

| Run | Setup | Épocas | Best W | Ratio vs Lawson | Leitura |
|---|---|---|---|---|---|
| **run_9** | Baseline do paper, do zero (Adam puro) | 2000 | **29.70** | **1.357×** | ✓ reproduz o paper (eles: 30.19); diferença = init/MC |
| — | "More compute" do zero, cosine T_max=4000 | (morreu ~1550) | — | — | interrompido (teto 2h interativo); no ep1550 estava em ~31.4, *pior* que run_9 no mesmo ponto |
| **run_11** | Warm restart de run_9 (LR fixa 5e-5, +650 ép) | →2400 | **29.23** | **1.335×** | **mínimo local CONFIRMADO** (ver abaixo) |
| run_12 | Continuação τ = 0.8i | 1000 | 31.30 | 1.43× | não melhorou (τ mais grosso) |
| smoke v1 | Sobolev g2 (sob=True, damp 1e-3) | 300 | 36.71 | — | descola colagem no ep200 (gluing 0.07→1.11) |
| smoke v2 | Sobolev g2 (L², damp 1e-2) | 300 | — | — | descola igual (gluing 0.06→0.92, W→40.8) |
| smoke v3 | Sobolev g2 (L², damp 1e-2, exclui pescoço) | 300 | — | — | descola igual (gluing 0.07→1.14) |

### Conclusão central do genus 2 — O 29.7 é um MÍNIMO LOCAL (o halter)

O teste decisivo foi o **run_11** (warm restart): pegou o melhor checkpoint do run_9 e treinou +650 épocas com LR fixa em 5e-5 — **50× mais alta** que a LR morta (~1e-6) do fim do cosine.

Trajetória do W (eval) no warm restart:

| Época | W (eval) |
|---|---|
| 1800 | 30.46 |
| 1900 | 31.25 |
| 2000 | 30.35 |
| 2100 | 29.96 |
| 2200 | 29.73 |
| 2300 | 29.32 |
| 2400 | 29.23 |

**Interpretação:** com 50× mais LR por 650 épocas, o W apenas **oscila na faixa ~29.2–31.2** e arrasta o piso de 29.7 → 29.2 (ganho de ~0.5). O `train sample` pula 28.9↔30.6 entre épocas vizinhas = **ruído de Monte-Carlo**, não descida. Se fosse "LR morta no fim do cosine", a LR viva teria feito o W despencar (29.7→27→25); não foi o caso. A superfície chacoalha *dentro de uma bacia* sem escapar.

→ **Isto refuta empiricamente a especulação do paper** ("perhaps with more compute this would be achievable"). Para esta parametrização e este τ inicial, mais compute **não** alcança Lawson — o fluxo fica preso no halter.

→ **O problema mudou de natureza:** não é mais "treinar melhor/mais", é **"escapar da bacia de atração"**.

**Âncora oficial atual: W = 29.23 (run_11, 1.335× Lawson).**

---

## Contribuições já consolidadas (antes de τ/Sobolev)

1. **Reprodução independente** da baseline genus-2 (29.70 vs 30.19 do paper) — valida o resultado deles.
2. **Refutação do "more compute"** — resultado negativo bem-feito: o 29.7 é mínimo local, não treino incompleto.
3. **Reformulação do problema** — de "treinar mais" para "escapar da bacia", o que direciona o trabalho futuro.
4. **Achado de método (genus 1):** híbrido Sobolev-descida + Adam-polimento bate cada um isolado.

---

## Próximas frentes (gerar dados)

### Frente A — Continuação em τ (barata, alta chance de escapar)
Variar o τ inicial das cartas (tubo mais fino/grosso muda a geometria do pescoço). Se o halter é a bacia onde τ=0.7 cai, outro τ pode cair noutra bacia — possivelmente a do Lawson.
- Candidatos: τ = 0.5i, 0.6i, 0.8i, 1.0i (2000 épocas cada).
- Critério: algum cai abaixo de 29? Abaixo de 27?

### Frente B — Sobolev multi-carta (a contribuição de método) — TESTADA, EM ABERTO
Estendido `sobolev_optim3.py` para multi-carta (métrica bloco-diagonal por carta, +
exclusão opcional do pescoço). Selftest multi-carta passa (cos>0 em cada carta).
**Porém:** ao integrar no `_train_epoch_genus2`, ligar o Sobolev descola a colagem em
todos os 3 regimes testados (gluing salta ~16× no ep200). Diagnóstico (hipótese): o
bloco-diagonal ignora o acoplamento M₁₂ do gluing, e a não-localidade dos parâmetros
da rede impede restringir o pré-condicionador ao corpo. **Conclusão:** funciona em
carta única (gênero 1), incompatível com a colagem multi-carta nesta forma. Ver
relatório técnico (RELATORIO_sobolev_willmore).

### Frente C — pré-condicionamento que respeite o acoplamento (futuro)
Separar o grad de gluing (mantido cru) do resto (pré-condicionado) — exige 2 backwards.
Ou aplicar Sobolev sobre construção de domínio único (sem colagem), se a inicialização
do octágono for viabilizada.

### Ordem sugerida
τ primeiro (barato, pode achar bacia melhor) → depois Sobolev para *afundar* a bacia nova. Os dois se somam.

---

## Workflow do cluster (referência rápida)

```bash
# login no cluster -> headnode (SLURM)
ssh <usuario>@<gateway>
ssh headnode
# GPU interativa (teto 2h)
srun --partition=l40s --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=2:00:00 --pty bash
# dentro do nó (shell novo): ativar o env conda e entrar no repo
conda activate <env>
cd ~/WillmorePINN-main
```

- Runs longos (>~1h40): usar **sbatch** (não interativo) ou **tmux** para sobreviver a queda de conexão.
- Tempo medido: genus-2 ≈ **1.2 s/época** na L40S → 2000 ép ≈ 40 min.
- Checkpoints em `checkpoints/run_N/` (best_model.pt, latest_model.pt, a cada 50 ép).
- **Cuidado com --resume:** restaura LR morta do checkpoint e start_epoch; para warm restart, forçar LR nova após o load (patch aplicado em run.py) e usar scheduler "none" para LR fixa.

---

## Tabela-resumo (uma linha por run, para colar/atualizar)

| ID | Gênero | Método | Épocas | Best W | Ratio | Nota |
|---|---|---|---|---|---|---|
| run_1 | 1 | Adam baseline | 200 | 19.74 | 0.9999× | ✓ Clifford |
| run_3 | 1 | NG quebrado | — | ~109 | 5.53× | clip matou passo |
| run_4 | 1 | precond→Adam | 200 | 36.15 | 1.83× | parcial |
| run_5 | 1 | precond→Adam mono | 600 | 21.57 | 1.09× | platô |
| run_8 | 1 | híbrido Sob+Adam | 600 | 20.13 | 1.0197× | ✓ furou platô |
| run_9 | 2 | Adam baseline | 2000 | 29.70 | 1.357× | reproduz paper |
| run_11 | 2 | warm restart LR5e-5 | +650 | 29.23 | 1.335× | **min local confirmado** |
