import jax
import jax.numpy as jnp

def quadratic_vrba_weights(residuals, phi=0.7, eps=1e-6):
    """
    Quadratic vRBA chuẩn giải tích kết hợp nội suy phân bố đều (Eq. 14 Toscano et al.).
    - residuals: shape (n_components, N)
    - phi in [0, 1]: mức độ ưu tiên vùng lỗi cao (mặc định 0.7)
    """
    res_abs = jnp.abs(residuals)
    # 1. Tính r_mean riêng cho từng thành phần (u, v, p, T)
    res_mean = jnp.mean(res_abs, axis=-1, keepdims=True) + eps
    
    # 2. Phân bố nghiêng tối ưu: q(x) = r / r_mean (Eq. 12 & 21-22)
    q = res_abs / res_mean
    
    # 3. Nội suy lồi với phân bố đều p_u = 1.0 (Eq. 14)
    # Tự động thỏa mãn Mean(weights) = 1.0 mà không cần clip nhân tạo
    weights = phi * q + (1.0 - phi) * 1.0
    
    return jax.lax.stop_gradient(weights)