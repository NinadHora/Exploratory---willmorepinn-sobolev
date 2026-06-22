# Wiring the Sobolev preconditioner into run.py

Two patches are involved.

## Genus 0/1 (single chart) — patch_runpy.py
Applies the hybrid: preconditioner for the first ~70% of epochs, then plain Adam.
Adds, after `loss.backward()` in the genus-0/1 train step:

    if globals().get('_NG_ACTIVE', False):
        natural_gradient_step(model, uv_batch, write_grad_only=True,
                              sobolev=False, damping=1e-3, cg_iters=20)

and a flag in the main loop:

    globals()['_NG_ACTIVE'] = USE_SOBOLEV_NG and (epoch < int(NG_PHASE * num_epochs))

## Genus 2 (two charts) — applied to _train_epoch_genus2
Add the import near the top of run.py:

    from sobolev_optim import natural_gradient_step_genus2
    USE_SOBOLEV_G2 = True
    NG_G2_START    = 200        # enable only after gluing settles (warmup + C2 done)
    NG_G2_END_FRAC = 0.85       # disable for the last 15% so Adam polishes
    NG_G2_KW = dict(sobolev=True, damping=1e-3, cg_iters=20)

Set the phase flag in the main loop, next to the genus-1 flag:

    globals()['_NG_ACTIVE_G2'] = (USE_SOBOLEV_G2 and
        epoch >= NG_G2_START and epoch < int(NG_G2_END_FRAC * num_epochs))

And inside _train_epoch_genus2, between backward and clip/step:

    loss.backward()
    if globals().get('_NG_ACTIVE_G2', False):
        natural_gradient_step_genus2(model, uv_T1[s1:e1], uv_T2[s2:e2], **NG_G2_KW)
    if gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()

NOTE: as documented in the technical note, this genus-2 integration currently
destabilises the gluing seam. It is included for reproducibility of that finding,
not as a working configuration.
