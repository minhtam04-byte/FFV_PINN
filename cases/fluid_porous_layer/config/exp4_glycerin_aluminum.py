"""Beckermann et al. (1988), experiment 4: configuration 2 (central porous layer, s/L = 0.33),
large test cell, glycerin, 6.35 mm aluminium beads (Table 1).

Measured temperature profiles at y/L = 0.912, 0.554, 0.204: Fig. 3(b).
See exp3_glycerin_glass.py for the porosity and the constant-property caveat.
"""
from cases.fluid_porous_layer.config import exp3_glycerin_glass


def get_config():
    config = exp3_glycerin_glass.get_config()
    config.wandb.name = "fluid_porous_layer_exp4_glycerin_aluminum"
    config.physics.Da = 2.518e-6
    config.physics.lambda_k = 17.10
    return config
