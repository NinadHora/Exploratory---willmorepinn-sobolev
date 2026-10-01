# sanidade da forma fraca: zero no Clifford exato, não zero no toro R=3
import torch, certificates as C
from methods.pinn_residual import weak_residual_sq, TESTS
torch.set_default_dtype(torch.float64)
uv, w = C.quadrature_grid("torus", 24, 24)
for name, phi in [("Clifford", C.exact_torus()), ("torus R=3", C.exact_torus(3.0, 1.0))]:
    print(name, float(weak_residual_sq(phi, uv, w, TESTS, 1e-3)))
