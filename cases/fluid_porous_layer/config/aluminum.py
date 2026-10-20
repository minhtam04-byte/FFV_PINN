"""Gao et al. (2017), Sec. 4.3: aluminum beads (Da = 1.534e-5, k_e/k_f = 37.47).

Reference psi_max: 23.21 (LBM, Gao et al.) / 23.9 (Beckermann et al., 1988).
The physics of base.py is already the aluminum case.
"""
from cases.fluid_porous_layer.config import base


def get_config():
    config = base.get_config()
    config.wandb.name = "fluid_porous_layer_aluminum"
    config.reference.psi_max = 23.21  # LBM, Gao et al. (2017)
    return config
