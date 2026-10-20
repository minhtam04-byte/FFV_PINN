"""Batch generation and experiment runner for the 1D conjugate-conduction case.

Moved verbatim from notebooks/conduction_1d.ipynb (outputs_dir added).
"""
import copy
import itertools

import jax.numpy as jnp

from jaxpi.samplers import MeshSampler, UniformSampler
from cases.conduction_1d.train import train_1d


def create_batches_1d(config):
    """
    Bộ sinh batch điểm nội miền cho bài toán 1 chiều.
    Tự động tính toán khoảng đệm biên (safety buffer) phụ thuộc vào stencil_reach của sơ đồ số.
    """
    L = config.physics.L
    scheme = getattr(config.diff, "scheme", "ad").lower().strip()

    if scheme in ["nd", "can", "ffv"]:
        n_x = getattr(config.diff, "n_x", 101)
        dx = L / (n_x - 1)

        # Cập nhật h_stencil đồng bộ chính xác với bước lưới
        config.diff.h_stencil = float(dx)

        # Stencil reach: CAN và FFV 1D cần h/2 hoặc h (1 bước), ND bậc cao có thể cần 2 bước
        stencil_reach = 1.0 if scheme in ["can", "ffv"] else 2.0

        use_continuous_mesh = getattr(config.diff, "use_continuous_mesh", False)

        if use_continuous_mesh:
            # Miền liên tục có đệm biên an toàn để các điểm stencil không văng ra ngoài [0, L]
            safe_x = stencil_reach * dx + 1e-5
            domain_safe = jnp.array([[safe_x, L - safe_x]])
            sampler = UniformSampler(domain_safe, config.training.batch_size)

        else:
            # Lưới rời rạc: loại bỏ đúng số nút sát biên theo stencil_reach
            num_exclude = int(stencil_reach)
            x_inner = jnp.linspace(
                num_exclude * dx, L - num_exclude * dx, n_x - 2 * num_exclude
            )
            # Reshape thành mảng (N, 1) chuẩn đầu vào cho MeshSampler
            mesh_points = x_inner.reshape(-1, 1)

            sampler = MeshSampler(
                mesh=mesh_points, batch_size=config.training.batch_size
            )

    elif scheme == "ad":
        # Vi phân tự động: lấy mẫu ngẫu nhiên trên toàn bộ miền (0, L)
        domain = jnp.array([[0.0, L]])
        sampler = UniformSampler(domain, config.training.batch_size)

    else:
        raise ValueError(f"Sơ đồ không hợp lệ: {scheme}")

    if config.training.random_sampling:
        return sampler
    return itertools.repeat(next(iter(sampler)))


def run_ablation_experiment_1d(exp_name, org_config_mod, modify_config_fn=None,pts=False,use_continuous_mesh=False, outputs_dir=None):
    """
    Hàm chạy thí nghiệm Ablation Study tự động:
    Clone config sâu, áp dụng các tuỳ chỉnh, sinh batch và tiến hành huấn luyện.
    """
    sub_config = copy.deepcopy(org_config_mod.get_config())
    sub_config.wandb.name = f"{sub_config.wandb.name}_{exp_name}"

    if modify_config_fn is not None:
        sub_config = modify_config_fn(sub_config,pts=pts,use_continuous_mesh=use_continuous_mesh)

    batches = create_batches_1d(sub_config)
    
    # Huấn luyện mô hình 1D (hàm train_1d nhận trực tiếp config và samplers)
    model, history = train_1d(config=sub_config, sampler=batches, outputs_dir=outputs_dir)
    return model, history
