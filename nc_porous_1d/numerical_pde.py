import jax
import jax.numpy as jnp


def get_k_local(x, D, k1, k2):
    """Xác định độ dẫn nhiệt cục bộ theo vị trí x."""
    return jnp.where(x <= D, k1, k2)


def compute_can_pde_residuals_1d(T_net_fn, params, x, h, D, k1, k2):
    """Sơ đồ CAN-PDE 1D chuẩn hóa dấu đường chéo Jacobian DƯƠNG (dR/dT_C > 0)."""
    dx = h
    x_all = jnp.stack([x, x + dx, x - dx])  # [C, E, W]

    eval_fn = jax.vmap(T_net_fn, in_axes=(None, 0))
    T_C, T_E, T_W = eval_fn(params, x_all)

    k_e = get_k_local(x + 0.5 * dx, D, k1, k2)
    k_w = get_k_local(x - 0.5 * dx, D, k1, k2)

    flux_diff_w = k_w * (T_C - T_W) / dx
    flux_diff_e = k_e * (T_E - T_C) / dx

    rT = (flux_diff_w - flux_diff_e) / dx
    return {"T": rT}


def compute_ffv_pde_residuals_1d(T_net_fn, params, x, h, D, k1, k2):
    """Sơ đồ FFV 1D dạng tích phân bảo toàn thể tích (cùng bậc O(1) với nghiệm T)."""
    dx = h
    x_all = jnp.stack([x, x + dx, x - dx])  # [C, E, W]

    eval_fn = jax.vmap(T_net_fn, in_axes=(None, 0))
    T_C, T_E, T_W = eval_fn(params, x_all)

    k_e = get_k_local(x + 0.5 * dx, D, k1, k2)
    k_w = get_k_local(x - 0.5 * dx, D, k1, k2)

    rT = k_w * (T_C - T_W) - k_e * (T_E - T_C)
    return {"T": rT}