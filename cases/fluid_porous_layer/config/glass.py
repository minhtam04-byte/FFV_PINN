"""Gao et al. (2017), Sec. 4.3: glass beads (Da = 1.37e-5, k_e/k_f = 1.362).

Reference psi_max: 16.52 (LBM, Gao et al.) / 16.4 (Beckermann et al., 1988).
"""
from cases.fluid_porous_layer.config import base


def get_config():
    config = base.get_config()
    config.wandb.name = "fluid_porous_layer_glass"

    # Vật lý
    config.physics.Da = 1.37e-5
    config.physics.lambda_k = 1.362
    config.reference.psi_max = 16.52  # LBM, Gao et al. (2017)

    # Siêu tham số khác với aluminum, giữ nguyên từ config case 1 cũ.
    # Chưa rõ khác biệt nào là cố ý: nếu so sánh hai vật liệu trong bài báo,
    # hãy xóa các dòng không cố ý để hai case chỉ khác nhau ở vật lý.
    config.arch.fourier_emb.embed_scale = 4.0  # aluminum: 2.0
    config.diff.scheme = "ffv"  # aluminum: "ffv_continuous"
    config.diff.n_x = 50  # aluminum: 80
    config.diff.n_y = 50  # aluminum: 80
    config.diff.h_stencil = 0.02  # aluminum: 0.0125 (create_batches đặt lại h = L / n_x)
    config.loss_weighting.loss_weights.T_neumann_bc = 1.0  # aluminum: 10.0
    config.pseudo_time.shrink.end_log_drop = 5.0  # aluminum: 6.0
    return config
