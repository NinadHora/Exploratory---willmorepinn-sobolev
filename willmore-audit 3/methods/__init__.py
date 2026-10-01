"""methods: cada estratégia do plano vira um Surface; só o Surface é auditado."""
from .base import REGISTRY, Surface, exact, register  # noqa: F401
from . import published, linear_spectral, optimisers, pinn_residual, others  # noqa: F401  (registram-se ao importar)
