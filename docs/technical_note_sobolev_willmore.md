# Technical note: exploring Sobolev preconditioning for the neural Willmore flow

**Project:** an exploratory extension of *Minimising Willmore Energy via Neural Flow* (Hirst, Sá Earp & Silva, arXiv:2604.04321), adding a natural-gradient / Sobolev preconditioner to the existing codebase.
**Author:** Nina da Hora, IMECC/UNICAMP · **Hardware:** GPU cluster (L40S / H100)
**Status:** work in progress; partial results, one open direction
**Date:** 22 June 2026

---

## 1. Purpose and scope

This note documents an attempt to engage with and contribute to the ongoing WillmorePINN project. It is not a finished result, nor a claim to have improved on the published work. The aim was modest: to take the existing code, understand it from the inside, and test whether one specific idea, namely preconditioning the optimisation step by the metric of the map φ rather than relying on plain Adam in parameter space, could help in the genus-2 case, which the authors themselves flag as open.

I report what I tried, what worked, and what did not, in the hope that it is useful to the people already working on this. Where I offer an explanation for an observed behaviour, I mark it as a hypothesis; the observations themselves are what I am more confident about.

## 2. Background

The published training optimises the Willmore objective with Adam over the network parameters. A natural geometric question, well within the "neural flow" framing the paper adopts (it cites Halverson and Ruehle's metric flows), is whether replacing or modifying this parameter-space update with a surface-metric-aware preconditioner could better approximate a Sobolev-type Willmore flow, of the kind standard in the classical Willmore-flow literature for taming the stiffness of the fourth-order flow.

For reference, the relevant minima:

| Genus | Minimiser | W | Status |
|---|---|---|---|
| 0 | Round sphere | 4π ≈ 12.57 | proved |
| 1 | Clifford torus | 2π² ≈ 19.74 | proved (Marques and Neves, 2014) |
| 2 | Lawson ξ₂,₁ | 4π² ≈ 21.89 | conjectured (Kusner), open |

The paper reaches W = 30.19 at genus 2, well below the naive catenoid sum of two Cliffords (≈39.48), but above Lawson (21.89). The authors note it "has not yet reached" Lawson, "perhaps with more compute this would be achievable."

## 3. Implementation

### 3.1 The preconditioner

The preconditioner computes g_nat = M⁻¹ g, where g = ∇_θ W and M is the Gram metric of the map φ_θ : (u,v) → R³:

- L²: M_ab = ∫ ⟨∂_a φ, ∂_b φ⟩ dA
- Sobolev: adds the ⟨∂_a φ_u, ∂_b φ_u⟩ + ⟨∂_a φ_v, ∂_b φ_v⟩ terms.

M (size p×p, with p the number of parameters) is never formed explicitly; M·g_nat = g is solved matrix-free by conjugate gradients, using JVP and VJP products. The code is in `sobolev_optim3.py`, with CPU self-tests that check the JVP against finite differences, that M is positive semidefinite, and that g_nat is a descent direction. These pass for both the single-chart and the two-chart versions.

### 3.2 How it is applied

In the genus-1 experiments the preconditioner descends quickly but does not polish as finely as Adam at the end. The mode used is therefore a hybrid: precondition during the first ~70% of epochs (writing M⁻¹g into the gradient for Adam to consume), then plain Adam. A pure natural-gradient step was also tried and abandoned, since it bypasses the anti-collapse regularity term and degenerates the surface.

## 4. Genus 1: a sanity check

Genus 1 is a solved geometric case, so this is only a sanity check that the machinery can recover the known minimiser in a favourable setting. It does: the hybrid reaches W = 20.13 (1.02× the Clifford value), descending smoothly from W ≈ 199 with regularity staying near zero throughout. I read this as confirmation that the single-chart preconditioner is sound, not as a result in itself, since there is nothing to improve on a solved case.

| Run | Method | Best W | Ratio vs Clifford |
|---|---|---|---|
| Adam baseline (paper) | 200 ep | 19.74 | 0.9999× |
| Hybrid Sobolev + Adam | 600 ep | 20.13 | 1.0197× |

## 5. Genus 2: findings

### 5.1 Reproducing the baseline

Running the authors' configuration unchanged, I reproduced their genus-2 result: W = 29.70 (their reported value is 30.19; the small difference is initialisation and Monte-Carlo noise). The purpose was to obtain a reliable anchor to compare against.

### 5.2 Does more training help?

The paper suggests more compute might reach Lawson. I checked this with a warm restart: resuming from the best checkpoint and training a further 650 epochs with the learning rate held at 5e-5, roughly 50× higher than the nearly dead end-of-cosine rate. The energy did not escape; it oscillated in the W ≈ 29.2 to 31.2 band and the best value moved from 29.7 to 29.23, with the variation dominated by Monte-Carlo noise rather than a genuine descent.

I would not over-claim from a single experiment, but this is at least suggestive that, for this initialisation and parametrisation, W ≈ 29.7 behaves like a local minimum (the "dumbbell"), and that simply training longer, without changing the parametrisation or initialisation, may be unlikely to reach Lawson. If so, the useful question shifts from "train more" to "escape the basin", which is what motivated trying the preconditioner and varying the initial τ.

### 5.3 Varying the initial τ (partial)

A small sweep of the initial modulus τ. So far I have only completed the τ = 0.8i continuation run, which gives W = 31.30 (1000 epochs), i.e. no improvement over the τ = 0.7 baseline; the τ = 0.9i and τ = 1.1i runs are pending. This is only preliminary.

### 5.4 The Sobolev preconditioner at genus 2: open

This is the part I could not get to work, and I want to be clear about that.

At genus 2 the model is two independent sub-networks (one per torus chart), so the metric M is naturally block-diagonal and the preconditioner can run per chart, reusing the validated single-chart machinery. The self-test passes. But when the preconditioner is switched on during training (after the gluing has settled, epoch ≥ 200), it pulls the two charts apart at the gluing seam: the gluing loss jumps by roughly 16× and the energy worsens. Three regimes were tried to avoid this:

| Regime | Gluing before (ep ≈ 150) | Gluing when switched on (ep 200) |
|---|---|---|
| Sobolev, damping 1e-3 | 0.068 | 1.11 |
| L², damping 1e-2 | 0.056 | 0.92 |
| L², damping 1e-2, excluding the neck (r ≤ 2.5δ) | 0.072 | 1.14 |

All three behave essentially the same.

My tentative explanation, a hypothesis rather than something proved: the gluing loss couples the two charts at the neck (it involves φ from both), and the block-diagonal scheme ignores the cross term M₁₂, so the preconditioned step does not account for the seam. I also tried excluding the neck samples from the metric so the preconditioner would act only on the body of each torus; this did not help, presumably because the network is a global map, so adjusting θ to reposition the body also moves the neck region by continuity, and one cannot confine the preconditioner to a sub-region when the parameters are shared globally.

What I take from this, cautiously: the Sobolev preconditioner is sound in the single-chart case, but the two-chart construction with gluing introduces a coupling that this block-diagonal scheme does not respect. I do not think this rules out preconditioning at genus 2 in general, only this particular and simplest form of it.

## 6. Relation to the paper's open directions

The paper lists several open threads; this exploration touches a few of them and may suggest one connection:

1. On "perhaps with more compute": the warm-restart experiment is at least evidence that, in this setup, more training alone does not escape the basin (Section 5.2).
2. On "extend training and refine tuning with lower gluing/regularity losses": relevant here, because the gluing is precisely what broke the preconditioner (Section 5.4).
3. On varying the starting point: a small start was made on the τ sweep (Section 5.3).
4. On the single connected fundamental domain (the octagon the authors attempted but could not initialise): if the gluing seam is what obstructs preconditioning, then a seamless domain might remove that obstruction. The authors' difficulty there (no analytic initial embedding) and the difficulty here (the seam breaks the preconditioner) look like two sides of the same trade-off: gluable charts are easy to initialise but carry a seam, while a single domain has no seam but is hard to initialise. This is offered as an observation worth keeping in mind, not as a conclusion.

### Possible next steps

1. Finish the τ sweep (0.9i, 1.1i) to see whether the basin is robust to τ.
2. Try a preconditioner that respects the coupling, for example keeping the gluing gradient raw while preconditioning the rest, which requires two backward passes.
3. If the single-domain construction becomes initialisable, test the preconditioner there, where the absence of a seam should remove the obstruction of Section 5.4.

---

## Appendix: code and setup

- **Code:** `sobolev_optim3.py` (single- and multi-chart preconditioner, optional neck exclusion). Self-tests: `--selftest` (genus 1), `--selftest-g2` (genus 2).
- **Integration:** a patch to `run.py` (`_train_epoch_genus2`), parametrised by environment variables `NG_SOBOLEV`, `NG_DAMPING`, `NG_CGITERS`.
- **Hardware:** GPU cluster, L40S and H100 nodes; conda env `ml_env`. Genus 2 runs at ≈ 1.2 s/epoch (L40S), so 2000 epochs take ≈ 40 min.
- **Current genus-2 anchor:** W = 29.23.

*This note records an honest attempt to contribute to an existing effort. The parts that worked are modest; the part that did not is documented so that the next attempt, mine or someone else's, can start from it.*
