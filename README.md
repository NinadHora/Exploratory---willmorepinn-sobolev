# willmore-sobolev

Sobolev / natural-gradient preconditioning for the neural Willmore flow, an exploratory
extension of [WillmorePINN](https://github.com/edhirst/WillmorePINN)
(Hirst, Sá Earp & Silva, *Minimising Willmore Energy via Neural Flow*, arXiv:2604.04321).

This is a work-in-progress contribution: it adds a natural-gradient (Sobolev) preconditioner
to the training step and tests whether it helps minimise the Willmore energy. It works in the
genus-1 (single-chart) case and is incompatible, in its current block-diagonal form, with the
genus-2 gluing construction. See the technical note in `docs/` for the full account.

## What's here

```
code/
  sobolev_optim.py        # the preconditioner: M^{-1}g via matrix-free CG (JVP+VJP).
                          #   single-chart (genus 0/1) + two-chart (genus 2, block-diagonal).
                          #   self-tests: --selftest (genus 1), --selftest-g2 (genus 2)
patches/
  patch_runpy.py          # patch that wires the preconditioner into the genus-0/1 train loop
docs/
  technical_note_sobolev_willmore.pdf   # the write-up: method, results, what worked / what didn't
  technical_note_sobolev_willmore.md
  EXPERIMENTOS_willmore_sobolev.md      # experiment log (run-by-run, PT-BR)
```

## Idea

The published training uses Adam over the network parameters. This replaces/augments that step
with a preconditioned direction `g_nat = M^{-1} g`, where `M` is the Gram metric of the map
`φ : (u,v) → R³`:

- L²:      `M_ab = ∫ ⟨∂_a φ, ∂_b φ⟩ dA`
- Sobolev: `M_ab = ∫ ⟨∂_a φ, ∂_b φ⟩ + ⟨∂_a φ_u, ∂_b φ_u⟩ + ⟨∂_a φ_v, ∂_b φ_v⟩ dA`

`M` (size p×p) is never formed; `M·g_nat = g` is solved matrix-free by conjugate gradients.

## Validate the maths (CPU, no GPU)

```bash
python code/sobolev_optim.py --selftest      # genus 1 (single chart)
python code/sobolev_optim.py --selftest-g2   # genus 2 (two charts, block-diagonal)
```

Both check: JVP vs finite differences, M positive semidefinite, and that `g_nat` is a descent
direction (`cos(g, g_nat) > 0`).

## Use in WillmorePINN

Drop `code/sobolev_optim.py` next to `run.py` in a WillmorePINN checkout. The genus-2 integration
adds, between `loss.backward()` and `optimizer.step()` inside `_train_epoch_genus2`:

```python
from sobolev_optim import natural_gradient_step_genus2
# ... after loss.backward() ...
if globals().get('_NG_ACTIVE_G2', False):
    natural_gradient_step_genus2(model, uv_T1[s1:e1], uv_T2[s2:e2],
                                 sobolev=True, damping=1e-3, cg_iters=20)
# ... then clip + optimizer.step() ...
```

with a flag set in the main loop to enable it only after the gluing has settled.

## Status (summary)

- **Genus 1:** works. Hybrid Sobolev+Adam reaches W = 20.13 (1.02× the Clifford value).
- **Genus 2:** baseline reproduced (W = 29.70). A warm restart suggests W ≈ 29.7 is a local
  minimum, not undertraining. The block-diagonal preconditioner is **incompatible with the
  gluing**: switching it on pulls the two charts apart at the seam. Open problem; see the note.

## License

Code intended for use with WillmorePINN; please cite the original paper. No warranty.
