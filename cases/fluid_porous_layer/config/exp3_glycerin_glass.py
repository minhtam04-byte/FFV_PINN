"""Beckermann et al. (1988), experiment 3: configuration 2 (central porous layer, s/L = 0.33),
large test cell, glycerin, 6.0 mm glass beads (Table 1).

Measured temperature profiles at y/L = 0.912, 0.554, 0.204: Fig. 3(a).
eps = 0.380 is inferred from C = 0.565 with Eq. (9)-(10) (see ROADMAP.md).
Glycerin viscosity varies by > 2x over the 10 K range of the experiment, so a
constant-property model is expected to agree within ~5% of Delta T (as in the paper).
"""
from cases.fluid_porous_layer.config import base


def get_config():
    config = base.get_config()
    config.wandb.name = "fluid_porous_layer_exp3_glycerin_glass"

    config.physics.Ra = 9.52e5
    config.physics.Pr = 8356.2
    config.physics.Da = 2.248e-6
    config.physics.lambda_k = 2.259
    config.physics.F_eps = 0.565
    config.physics.epsilon = 0.380

    config.diff.scheme = "ffv_compact"
    config.diff.use_continuous_mesh = True
    config.reference.psi_max = None
    return config
