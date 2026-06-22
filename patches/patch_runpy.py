# Aplica o gradiente natural como PRE-CONDICIONADOR do Adam no run.py (genus 0/1).
# Rode na raiz do repo (onde esta run.py), com sobolev_optim.py ao lado.
src = open("run.py").read()

# 1) injeta import + flags logo apos o import do modelo
anchor = "from model import create_embedding_model"
inject = (anchor + "\n"
          "from sobolev_optim import natural_gradient_step\n"
          "USE_SOBOLEV_NG = True\n"
          "SOBOLEV_KW = dict(sobolev=False, damping=1e-3, cg_iters=20)")
assert anchor in src, "ancora do import nao encontrada"
assert "from sobolev_optim import" not in src, "ja parece patcheado — re-extraia o zip limpo"
src = src.replace(anchor, inject, 1)

# 2) no train_epoch genus0/1: pre-condiciona p.grad ANTES do clip + optimizer.step()
old = '''        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Optimizer step
        optimizer.step()'''
new = '''        # Backward pass
        loss.backward()
        
        # Sobolev natural gradient: M^{-1} g -> p.grad; o Adam (optimizer.step) consome
        if USE_SOBOLEV_NG:
            natural_gradient_step(model, uv_batch, write_grad_only=True, **SOBOLEV_KW)
        
        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Optimizer step
        optimizer.step()'''
assert old in src, "bloco do train_epoch genus0/1 nao bateu"
src = src.replace(old, new, 1)

open("run.py","w").write(src)
print("run.py patcheado OK (pre-condicionador-no-Adam, genus 0/1)")
