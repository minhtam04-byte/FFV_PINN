import jax
import jax.numpy as jnp
from jaxpi.derivatives import value_and_jacfwd


# =============================================================================
# 1. Auxiliary functions for porous media and buoyancy
# =============================================================================
def is_in_porous_zone(x, physics_cfg):
    """Kiểm tra điểm x có nằm trong lớp môi trường xốp [x_start, x_end] hay không.
    Nếu không khai báo x_start/x_end, mặc định toàn miền [0.0, 1.0] là miền xốp.
    """
    x_start = float(physics_cfg.get("porous_x_start", 0.0))
    x_end = float(physics_cfg.get("porous_x_end", 1.0))
    return jnp.logical_and(x >= x_start, x <= x_end)


def get_eps_local(x, physics_cfg):
    """Lấy độ rỗng cục bộ epsilon(x): trả về eps_p trong miền xốp, 1.0 ngoài miền chất lưu."""
    eps_p = float(physics_cfg.get("epsilon", 0.4))
    return jnp.where(is_in_porous_zone(x, physics_cfg), eps_p, 1.0)


def get_k_local(x, physics_cfg):
    """Lấy hệ số khuếch tán nhiệt hiệu dụng alpha_e(x)."""
    Pr = float(physics_cfg.get("Pr", 6.44))
    Ra = float(physics_cfg.get("Ra", 3.7e6))
    lambda_k = float(physics_cfg.get("lambda_k", 1.0))
    k_fluid = 1.0 / jnp.sqrt(Pr * Ra)
    return jnp.where(is_in_porous_zone(x, physics_cfg), lambda_k * k_fluid, k_fluid)


def compute_porous_and_buoyancy(u, v, T, x, physics_cfg):
    """Tính toán đồng bộ các hệ số khuếch tán, lực cản Darcy-Forchheimer và lực nổi."""
    eps_p = float(physics_cfg.get("epsilon", 0.4))
    Da_p = float(physics_cfg.get("Da", 1.534e-5))
    Pr = float(physics_cfg.get("Pr", 6.44))
    Ra = float(physics_cfg.get("Ra", 3.7e6))
    F_eps = float(physics_cfg.get("F_eps", 0.554))
    T_ref = float(physics_cfg.get("T_ref", 0.5))
    lambda_k = float(physics_cfg.get("lambda_k", 1.0))

    visc_coeff = jnp.sqrt(Pr / Ra)
    diff_T_coeff = 1.0 / jnp.sqrt(Pr * Ra)

    in_porous = is_in_porous_zone(x, physics_cfg)

    eps_local = jnp.where(in_porous, eps_p, 1.0)
    inv_eps = 1.0 / jnp.maximum(eps_local, 1e-6)
    diff_T_local = jnp.where(in_porous, lambda_k * diff_T_coeff, diff_T_coeff)
    inv_K = jnp.where(in_porous, 1.0 / Da_p, 0.0)
    F_eps_local = jnp.where(in_porous, F_eps, 0.0)

    V_mag = jnp.sqrt(u**2 + v**2 + 1e-6)
    drag_coeff = eps_local * (visc_coeff * inv_K + F_eps_local * jnp.sqrt(inv_K) * V_mag)

    f_u_drag = drag_coeff * u
    f_v_drag = drag_coeff * v
    f_buoyancy = eps_local * (T - T_ref)

    return inv_eps, visc_coeff, diff_T_local, f_u_drag, f_v_drag, f_buoyancy


def get_porous_segment_fraction(x_a, x_b, physics_cfg):
    """Tính tỷ lệ chính xác (trong đoạn [0, 1]) của đoạn thẳng [x_a, x_b] (với x_a < x_b)
    nằm bên trong lớp môi trường xốp [x_start, x_end].
    Đúng tuyệt đối cho cả Random Continuous Sampling lẫn Uniform Grid.
    Chỉ đúng với các cấu hình xốp đơn giản (cần chỉnh sửa lại khi hình xốp phức tạp như tròn)
    """
    x_start = float(physics_cfg.get("porous_x_start", 0.0))
    x_end = float(physics_cfg.get("porous_x_end", 1.0))
    length = jnp.maximum(x_b - x_a, 1e-12)

    overlap = jnp.maximum(0.0, jnp.minimum(x_b, x_end) - jnp.maximum(x_a, x_start))
    return jnp.clip(overlap / length, 0.0, 1.0)


def get_exact_harmonic_k(x_a, x_b, physics_cfg):
    """Trung bình điều hòa tổng quát theo khoảng cách thực tế (Distance-Weighted Harmonic Mean):
    k_eff = 1 / ( theta / k_porous + (1 - theta) / k_fluid )
    Khi mặt phân cách nằm chính giữa đoạn [x_a, x_b] (theta = 0.5), công thức tự động
    thu về đúng công thức Harmonic Mean cổ điển của Patankar: 2*k_p*k_f / (k_p + k_f).
    """
    Pr = float(physics_cfg.get("Pr", 6.44))
    Ra = float(physics_cfg.get("Ra", 3.7e6))
    lambda_k = float(physics_cfg.get("lambda_k", 1.0))

    k_fluid = 1.0 / jnp.sqrt(Pr * Ra)
    k_porous = lambda_k * k_fluid

    theta = get_porous_segment_fraction(x_a, x_b, physics_cfg)
    inv_k_eff = (theta / k_porous) + ((1.0 - theta) / k_fluid)
    return 1.0 / inv_k_eff


def get_exact_harmonic_coeff(x_a, x_b, c_fluid, c_porous, physics_cfg):
    """Trung bình điều hòa có trọng số khoảng cách của một hệ số khuếch tán bất kỳ
    (c_fluid trong chất lưu, c_porous trong miền xốp) trên đoạn [x_a, x_b]."""
    theta = get_porous_segment_fraction(x_a, x_b, physics_cfg)
    return 1.0 / (theta / c_porous + (1.0 - theta) / c_fluid)


def compute_pure_porous_drag(u, v, physics_cfg):
    """Tính cường độ lực cản Darcy-Forchheimer thuần túy (khi điểm nằm 100% trong miền xốp).
    Sau đó sẽ được nhân với tỷ lệ thể tích chiếm chỗ chính xác alpha_P của ô FVM.
    """
    eps_p = float(physics_cfg.get("epsilon", 0.4))
    Da_p = float(physics_cfg.get("Da", 1.534e-5))
    Pr = float(physics_cfg.get("Pr", 6.44))
    Ra = float(physics_cfg.get("Ra", 3.7e6))
    F_eps = float(physics_cfg.get("F_eps", 0.554))

    visc_coeff = jnp.sqrt(Pr / Ra)
    inv_K = 1.0 / Da_p
    V_mag = jnp.sqrt(u**2 + v**2 + 1e-6)

    drag_coeff = eps_p * (visc_coeff * inv_K + F_eps * jnp.sqrt(inv_K) * V_mag)
    return drag_coeff * u, drag_coeff * v



# =============================================================================
# 2. FFV-PINN scheme (9-point stencil: [C, E, W, N, S, EE, WW, NN, SS])
# Tham khảo từ bài báo FFV-PINN và chỉnh sửa lại cho bài toán truyền nhiệt liên hợp 
# trong môi trường xốp
# =============================================================================
def compute_ffv_pde_residuals(neural_net_fn, params, x, y, h, physics_cfg):
    """Sơ đồ FFV-PINN 2D 9 điểm:
    - Bảo toàn thông lượng qua các mặt kiểm soát (e, w, n, s).
    - Trung bình điều hòa (Harmonic Mean) chuẩn Patankar cho hệ số dẫn nhiệt k.
    - Tích phân thể tích Simpson 3 điểm [w, P, e] cho nguồn cản Darcy-Forchheimer.
    """
    dx = dy = h
    half_h = 0.5 * h

    x_all = jnp.stack([x, x + half_h, x - half_h, x, x, x + dx, x - dx, x, x])
    y_all = jnp.stack([y, y, y, y + half_h, y - half_h, y, y, y + dy, y - dy])

    # Chỉ cần Forward Pass trên 9 điểm (không dùng đạo hàm AD)
    eval_fn = jax.vmap(neural_net_fn, in_axes=(None, 0, 0))
    u_all, v_all, p_all, T_all = eval_fn(params, x_all, y_all)

    u_P, u_e, u_w, u_n, u_s, u_E, u_W, u_N, u_S = u_all
    v_P, v_e, v_w, v_n, v_s, v_E, v_W, v_N, v_S = v_all
    p_P, p_e, p_w, p_n, p_s, p_E, p_W, p_N, p_S = p_all
    T_P, T_e, T_w, T_n, T_s, T_E, T_W, T_N, T_S = T_all

    # 1. Phương trình liên tục
    div = (u_e - u_w) / dx + (v_n - v_s) / dy
    rc = div

    # 2. Số hạng đối lưu bảo toàn (đồng bộ hệ số 1/eps với Gao et al., 2017)
    inv_eps_e = 1.0 / get_eps_local(x + half_h, physics_cfg)
    inv_eps_w = 1.0 / get_eps_local(x - half_h, physics_cfg)
    inv_eps_n = 1.0 / get_eps_local(x, physics_cfg)
    inv_eps_s = inv_eps_n
    inv_eps_P = inv_eps_n

    conv_u = (
        ((u_e**2) * inv_eps_e - (u_w**2) * inv_eps_w) / dx
        + ((u_n * v_n) * inv_eps_n - (u_s * v_s) * inv_eps_s) / dy
        - u_P * inv_eps_P * div
    )
    conv_v = (
        ((u_e * v_e) * inv_eps_e - (u_w * v_w) * inv_eps_w) / dx
        + ((v_n**2) * inv_eps_n - (v_s**2) * inv_eps_s) / dy
        - v_P * inv_eps_P * div
    )
    conv_T = (
        ((u_e * T_e) - (u_w * T_w)) / dx
        + ((v_n * T_n) - (v_s * T_s)) / dy
        - T_P * div
    )

    # 3. Gradient áp suất trực tiếp từ các mặt kiểm soát e, w, n, s
    dp_dx = (p_e - p_w) / dx
    dp_dy = (p_n - p_s) / dy

    # 4. Khuếch tán động lượng (a_P > 0)
    _, visc_coeff, _, f_u_P, f_v_P, f_buoyancy = compute_porous_and_buoyancy(
        u_P, v_P, T_P, x, physics_cfg
    )
    diff_u_oper = visc_coeff * (4.0 * u_P - u_E - u_W - u_N - u_S) / (dx**2)
    diff_v_oper = visc_coeff * (4.0 * v_P - v_E - v_W - v_N - v_S) / (dy**2)

    # 5. Khuếch tán nhiệt với Trung bình điều hòa (Harmonic Mean - Patankar)
    k_P = get_k_local(x, physics_cfg)
    k_E = get_k_local(x + dx, physics_cfg)
    k_W = get_k_local(x - dx, physics_cfg)

    k_e_harm = (2.0 * k_P * k_E) / (k_P + k_E + 1e-12)
    k_w_harm = (2.0 * k_P * k_W) / (k_P + k_W + 1e-12)

    diff_T_oper = (
        (k_e_harm * (T_P - T_E) + k_w_harm * (T_P - T_W)) / (dx**2)
        + k_P * (2.0 * T_P - T_N - T_S) / (dy**2)
    )

    # 6. Tích phân thể tích lực cản Darcy-Forchheimer qua ô kiểm soát [w, P, e]
    _, _, _, f_u_e, f_v_e, _ = compute_porous_and_buoyancy(
        u_e, v_e, T_e, x + half_h, physics_cfg
    )
    _, _, _, f_u_w, f_v_w, _ = compute_porous_and_buoyancy(
        u_w, v_w, T_w, x - half_h, physics_cfg
    )

    f_u_drag = 0.5 * f_u_P + 0.25 * (f_u_e + f_u_w)
    f_v_drag = 0.5 * f_v_P + 0.25 * (f_v_e + f_v_w)

    # 7. Tổng hợp thặng dư PDE
    ru = conv_u + dp_dx + diff_u_oper + f_u_drag
    rv = conv_v + dp_dy + diff_v_oper + f_v_drag - f_buoyancy
    rT = conv_T + diff_T_oper

    return {"u": ru, "v": rv, "p": rc, "T": rT}


# =============================================================================
# 2B. SƠ ĐỒ FFV_CONTINUOUS 
# Tương tự như sơ đồ FFV, nhưng điều chỉnh lại các công thức để thích hợp với 
# lấy mẫu ngẫu nhiên liên tục
# =============================================================================
def compute_ffv_continuous_pde_residuals(
    neural_net_fn, params, x, y, h, physics_cfg
):
    """Sơ đồ FFV-PINN Continuous 2D (9 điểm: [P, e, w, n, s, E, W, N, S]):
    - Derivative-Free (Không tính bất kỳ đạo hàm AD nào -> Nhanh gấp ~3 lần).
    - Gradient áp suất tính trực tiếp từ hiệu áp suất tại các mặt kiểm soát (p_e - p_w)/dx
      và (p_n - p_s)/dy, khớp tự nhiên với phương trình liên tục (u_e - u_w)/dx.
    - Distance-Weighted Harmonic Mean cho dẫn nhiệt qua mặt phân cách bất kỳ.
    """
    dx = dy = h
    half_h = 0.5 * h

    # 1. Gom tọa độ stencil 9 điểm FVM: [P, e, w, n, s, E, W, N, S]
    x_all = jnp.stack([x, x + half_h, x - half_h, x, x, x + dx, x - dx, x, x])
    y_all = jnp.stack([y, y, y, y + half_h, y - half_h, y, y, y + dy, y - dy])

    # 2. Đánh giá thuần túy Forward Pass trên 9 điểm (Không cần value_and_jacfwd!)
    eval_fn = jax.vmap(neural_net_fn, in_axes=(None, 0, 0))
    u_all, v_all, p_all, T_all = eval_fn(params, x_all, y_all)

    u_P, u_e, u_w, u_n, u_s, u_E, u_W, u_N, u_S = u_all
    v_P, v_e, v_w, v_n, v_s, v_E, v_W, v_N, v_S = v_all
    p_P, p_e, p_w, p_n, p_s, p_E, p_W, p_N, p_S = p_all
    T_P, T_e, T_w, T_n, T_s, T_E, T_W, T_N, T_S = T_all

    # 3. Phương trình liên tục (Continuity)
    div = (u_e - u_w) / dx + (v_n - v_s) / dy
    rc = div

    # 4. Tỷ lệ thể tích ô kiểm soát [x - h/2, x + h/2] nằm trong miền xốp (alpha_P)
    eps_p = float(physics_cfg.get("epsilon", 0.4))
    T_ref = float(physics_cfg.get("T_ref", 0.5))
    Pr = float(physics_cfg.get("Pr", 6.44))
    Ra = float(physics_cfg.get("Ra", 3.7e6))
    visc_coeff = jnp.sqrt(Pr / Ra)

    alpha_P = get_porous_segment_fraction(x - half_h, x + half_h, physics_cfg)
    eps_cell_avg = alpha_P * eps_p + (1.0 - alpha_P) * 1.0
    inv_eps_cell = 1.0 / jnp.maximum(eps_cell_avg, 1e-6)

    # 5. Số hạng đối lưu bảo toàn trực tiếp tại mặt (Không trừ phi_P * div)
    inv_eps_e = 1.0 / get_eps_local(x + half_h, physics_cfg)
    inv_eps_w = 1.0 / get_eps_local(x - half_h, physics_cfg)
    inv_eps_n = inv_eps_cell
    inv_eps_s = inv_eps_cell

    conv_u = (
        ((u_e**2) * inv_eps_e - (u_w**2) * inv_eps_w) / dx
        + ((u_n * v_n) * inv_eps_n - (u_s * v_s) * inv_eps_s) / dy
    )
    conv_v = (
        ((u_e * v_e) * inv_eps_e - (u_w * v_w) * inv_eps_w) / dx
        + ((v_n**2) * inv_eps_n - (v_s**2) * inv_eps_s) / dy
    )
    conv_T = (
        ((u_e * T_e) - (u_w * T_w)) / dx
        + ((v_n * T_n) - (v_s * T_s)) / dy
    )

    # 6. Gradient áp suất trực tiếp từ các mặt kiểm soát e, w, n, s (Không cần CAN-AD)
    dp_dx = (p_e - p_w) / dx
    dp_dy = (p_n - p_s) / dy

    # 7. Khuếch tán động lượng (a_P = +4*visc/h^2 > 0)
    diff_u_oper = visc_coeff * (4.0 * u_P - u_E - u_W - u_N - u_S) / (dx**2)
    diff_v_oper = visc_coeff * (4.0 * v_P - v_E - v_W - v_N - v_S) / (dy**2)

    # 8. Khuếch tán nhiệt với Exact Distance-Weighted Harmonic Mean
    k_e_exact = get_exact_harmonic_k(x, x + dx, physics_cfg)
    k_w_exact = get_exact_harmonic_k(x - dx, x, physics_cfg)

    k_fluid = 1.0 / jnp.sqrt(Pr * Ra)
    lambda_k = float(physics_cfg.get("lambda_k", 1.0))
    k_y_avg = (alpha_P * lambda_k + (1.0 - alpha_P)) * k_fluid

    diff_T_oper = (
        (k_e_exact * (T_P - T_E) + k_w_exact * (T_P - T_W)) / (dx**2)
        + k_y_avg * (2.0 * T_P - T_N - T_S) / (dy**2)
    )

    # 9. Lực cản Darcy-Forchheimer & Lực nổi tích phân trên hai nửa ô [w, P] và [P, e]
    # alpha_left = get_porous_segment_fraction(x - half_h, x, physics_cfg)
    # alpha_right = get_porous_segment_fraction(x, x + half_h, physics_cfg)

    # fu_pure_P, fv_pure_P = compute_pure_porous_drag(u_P, v_P, physics_cfg)
    # fu_pure_e, fv_pure_e = compute_pure_porous_drag(u_e, v_e, physics_cfg)
    # fu_pure_w, fv_pure_w = compute_pure_porous_drag(u_w, v_w, physics_cfg)

    # f_u_drag = 0.5 * alpha_left * (0.5 * fu_pure_w + 0.5 * fu_pure_P) + (
    #     0.5 * alpha_right * (0.5 * fu_pure_P + 0.5 * fu_pure_e)
    # )
    # f_v_drag = 0.5 * alpha_left * (0.5 * fv_pure_w + 0.5 * fv_pure_P) + (
    #     0.5 * alpha_right * (0.5 * fv_pure_P + 0.5 * fv_pure_e)
    # )

    
    # f_buoyancy = eps_cell_avg * (T_P - T_ref)

    # 9. Lực cản Darcy-Forchheimer & Lực nổi Boussinesq tích phân cân bằng trên ô kiểm soát:
    alpha_left = get_porous_segment_fraction(x - half_h, x, physics_cfg)
    alpha_right = get_porous_segment_fraction(x, x + half_h, physics_cfg)

    fu_pure_P, fv_pure_P = compute_pure_porous_drag(u_P, v_P, physics_cfg)
    fu_pure_e, fv_pure_e = compute_pure_porous_drag(u_e, v_e, physics_cfg)
    fu_pure_w, fv_pure_w = compute_pure_porous_drag(u_w, v_w, physics_cfg)
    _, fv_pure_n = compute_pure_porous_drag(u_n, v_n, physics_cfg)
    _, fv_pure_s = compute_pure_porous_drag(u_s, v_s, physics_cfg)

    # - Phương x (ru): Cân bằng trực tiếp với (p_e - p_w) / dx trên [w, P, e]
    f_u_drag = (
        0.5 * alpha_left * (0.5 * fu_pure_w + 0.5 * fu_pure_P)
        + 0.5 * alpha_right * (0.5 * fu_pure_P + 0.5 * fu_pure_e)
    )

    # - Phương y (rv): Kết hợp phân đoạn xốp [w, P, e] và tích phân dọc [s, P, n]
    #   để cân bằng đồng bộ với (p_n - p_s) / dy
    fv_vert_corr = 0.125 * (fv_pure_n + fv_pure_s - 2.0 * fv_pure_P)
    f_v_drag = (
        0.5 * alpha_left * (0.5 * fv_pure_w + 0.5 * fv_pure_P + fv_vert_corr)
        + 0.5 * alpha_right * (0.5 * fv_pure_P + 0.5 * fv_pure_e + fv_vert_corr)
    )

    eps_left = alpha_left * eps_p + (1.0 - alpha_left) * 1.0
    eps_right = alpha_right * eps_p + (1.0 - alpha_right) * 1.0
    T_vert_corr = 0.125 * (T_n + T_s - 2.0 * T_P)
    f_buoyancy = (
        0.5 * eps_left * (0.5 * (T_w + T_P) + T_vert_corr - T_ref)
        + 0.5 * eps_right * (0.5 * (T_P + T_e) + T_vert_corr - T_ref)
    )

    # 10. Tổng hợp thặng dư PDE
    ru = conv_u + dp_dx + diff_u_oper + f_u_drag
    rv = conv_v + dp_dy + diff_v_oper + f_v_drag - f_buoyancy
    rT = conv_T + diff_T_oper

    return {"u": ru, "v": rv, "p": rc, "T": rT}


# =============================================================================
# 2C. SƠ ĐỒ FFV_COMPACT
# Như ffv_continuous nhưng khuếch tán cũng tính từ giá trị tại mặt (cách P h/2):
# stencil 5 điểm [P, e, w, n, s] chỉ vươn tới h/2, nên tâm ô kiểm soát lấy mẫu được
# trong [h/2, L - h/2] và các ô kiểm soát phủ kín cả miền, kể cả sát vách.
# =============================================================================
def compute_ffv_compact_pde_residuals(neural_net_fn, params, x, y, h, physics_cfg):
    """Sơ đồ FFV compact 2D 5 điểm [P, e, w, n, s] (derivative-free, bậc 2):
    - Đối lưu, áp suất, lực cản và lực nổi giống hệt ffv_continuous.
    - Thông lượng khuếch tán tại mặt: k_e * (phi_e - phi_P) / (h/2), với k_e là trung bình
      điều hòa có trọng số khoảng cách trên [x, x + h/2] (chính xác cho nghiệm tuyến tính
      từng khúc có thông lượng liên tục qua mặt phân cách).
    - Với ô sát vách (tâm tại h/2), mặt w nằm trên vách nên thông lượng nhiệt tại vách
      đi trực tiếp vào thặng dư.

    physics.momentum_form chọn cách viết phương trình động lượng một miền:
    - "gao" (mặc định): đúng Eq. (2) của Gao et al. (2017), nhân với ε:
        (1/ε)∇·(uu) + ∇p − ν∇²u + ε·(cản) − ε·(nổi) = 0, ν hằng số qua mặt phân cách
        => tại mặt phân cách: ∂u_t/∂n liên tục và εP liên tục.
    - "hirata": dạng chia cho ε, hệ số gián đoạn nằm trong divergence (Hirata et al. 2009):
        (1/ε)∇·(uu/ε) + ∇P − ∇·((ν/ε)∇u) + (cản) − (nổi) = 0
        => tại mặt phân cách: ν∂u_f/∂n = (ν/ε)∂u_m/∂n và P liên tục, tương đương mô hình
        hai miền Darcy–Brinkman. Trong lòng mỗi miền, hai dạng cho cùng một vật lý.
    """
    dx = dy = h
    half_h = 0.5 * h
    momentum_form = str(physics_cfg.get("momentum_form", "gao")).lower()
    if momentum_form not in ("gao", "hirata"):
        raise ValueError(f"physics.momentum_form='{momentum_form}' phải là 'gao' hoặc 'hirata'.")

    # 1. Stencil 5 điểm: [P, e, w, n, s]
    x_all = jnp.stack([x, x + half_h, x - half_h, x, x])
    y_all = jnp.stack([y, y, y, y + half_h, y - half_h])

    eval_fn = jax.vmap(neural_net_fn, in_axes=(None, 0, 0))
    u_all, v_all, p_all, T_all = eval_fn(params, x_all, y_all)

    u_P, u_e, u_w, u_n, u_s = u_all
    v_P, v_e, v_w, v_n, v_s = v_all
    p_P, p_e, p_w, p_n, p_s = p_all
    T_P, T_e, T_w, T_n, T_s = T_all

    # 2. Phương trình liên tục
    div = (u_e - u_w) / dx + (v_n - v_s) / dy
    rc = div

    # 3. Tỷ lệ thể tích ô kiểm soát [x - h/2, x + h/2] nằm trong miền xốp (alpha_P)
    eps_p = float(physics_cfg.get("epsilon", 0.4))
    T_ref = float(physics_cfg.get("T_ref", 0.5))
    Pr = float(physics_cfg.get("Pr", 6.44))
    Ra = float(physics_cfg.get("Ra", 3.7e6))
    visc_coeff = jnp.sqrt(Pr / Ra)

    alpha_P = get_porous_segment_fraction(x - half_h, x + half_h, physics_cfg)
    eps_cell_avg = alpha_P * eps_p + (1.0 - alpha_P) * 1.0
    inv_eps_cell = 1.0 / jnp.maximum(eps_cell_avg, 1e-6)

    # 4. Số hạng đối lưu bảo toàn tại mặt
    inv_eps_e = 1.0 / get_eps_local(x + half_h, physics_cfg)
    inv_eps_w = 1.0 / get_eps_local(x - half_h, physics_cfg)
    inv_eps_n = inv_eps_cell
    inv_eps_s = inv_eps_cell

    conv_u = (
        ((u_e**2) * inv_eps_e - (u_w**2) * inv_eps_w) / dx
        + ((u_n * v_n) * inv_eps_n - (u_s * v_s) * inv_eps_s) / dy
    )
    conv_v = (
        ((u_e * v_e) * inv_eps_e - (u_w * v_w) * inv_eps_w) / dx
        + ((v_n**2) * inv_eps_n - (v_s**2) * inv_eps_s) / dy
    )
    conv_T = (
        ((u_e * T_e) - (u_w * T_w)) / dx
        + ((v_n * T_n) - (v_s * T_s)) / dy
    )

    # 5. Gradient áp suất từ các mặt kiểm soát
    dp_dx = (p_e - p_w) / dx
    dp_dy = (p_n - p_s) / dy

    # 6. Khuếch tán động lượng từ giá trị tại mặt (khoảng cách h/2)
    diff_u_oper = visc_coeff * (4.0 * u_P - u_e - u_w - u_n - u_s) / (half_h**2)
    diff_v_oper = visc_coeff * (4.0 * v_P - v_e - v_w - v_n - v_s) / (half_h**2)

    # 7. Khuếch tán nhiệt: harmonic mean có trọng số khoảng cách trên nửa ô [P, e] và [w, P]
    k_e_exact = get_exact_harmonic_k(x, x + half_h, physics_cfg)
    k_w_exact = get_exact_harmonic_k(x - half_h, x, physics_cfg)

    k_fluid = 1.0 / jnp.sqrt(Pr * Ra)
    lambda_k = float(physics_cfg.get("lambda_k", 1.0))
    k_y_avg = (alpha_P * lambda_k + (1.0 - alpha_P)) * k_fluid

    diff_T_oper = (
        (k_e_exact * (T_P - T_e) + k_w_exact * (T_P - T_w)) / (half_h**2)
        + k_y_avg * (2.0 * T_P - T_n - T_s) / (half_h**2)
    )

    # 8. Lực cản Darcy-Forchheimer & lực nổi: giống ffv_continuous
    alpha_left = get_porous_segment_fraction(x - half_h, x, physics_cfg)
    alpha_right = get_porous_segment_fraction(x, x + half_h, physics_cfg)

    fu_pure_P, fv_pure_P = compute_pure_porous_drag(u_P, v_P, physics_cfg)
    fu_pure_e, fv_pure_e = compute_pure_porous_drag(u_e, v_e, physics_cfg)
    fu_pure_w, fv_pure_w = compute_pure_porous_drag(u_w, v_w, physics_cfg)
    _, fv_pure_n = compute_pure_porous_drag(u_n, v_n, physics_cfg)
    _, fv_pure_s = compute_pure_porous_drag(u_s, v_s, physics_cfg)

    f_u_drag = (
        0.5 * alpha_left * (0.5 * fu_pure_w + 0.5 * fu_pure_P)
        + 0.5 * alpha_right * (0.5 * fu_pure_P + 0.5 * fu_pure_e)
    )
    fv_vert_corr = 0.125 * (fv_pure_n + fv_pure_s - 2.0 * fv_pure_P)
    f_v_drag = (
        0.5 * alpha_left * (0.5 * fv_pure_w + 0.5 * fv_pure_P + fv_vert_corr)
        + 0.5 * alpha_right * (0.5 * fv_pure_P + 0.5 * fv_pure_e + fv_vert_corr)
    )

    eps_left = alpha_left * eps_p + (1.0 - alpha_left) * 1.0
    eps_right = alpha_right * eps_p + (1.0 - alpha_right) * 1.0
    T_vert_corr = 0.125 * (T_n + T_s - 2.0 * T_P)
    f_buoyancy = (
        0.5 * eps_left * (0.5 * (T_w + T_P) + T_vert_corr - T_ref)
        + 0.5 * eps_right * (0.5 * (T_P + T_e) + T_vert_corr - T_ref)
    )

    if momentum_form == "hirata":
        # Dạng Hirata (chia cho ε): quán tính nhân thêm 1/ε của ô; cản và lực nổi không có ε;
        # nhớt hiệu dụng ν/ε trong miền xốp, trung bình điều hòa tại các mặt e, w
        # (mặt n, s song song với mặt phân cách: trung bình số học theo phần thể tích).
        conv_u = inv_eps_cell * conv_u
        conv_v = inv_eps_cell * conv_v
        f_u_drag = f_u_drag / eps_p
        f_v_drag = f_v_drag / eps_p
        f_buoyancy = (
            0.5 * (0.5 * (T_w + T_P) + T_vert_corr - T_ref)
            + 0.5 * (0.5 * (T_P + T_e) + T_vert_corr - T_ref)
        )
        visc_porous = visc_coeff / eps_p
        mu_e = get_exact_harmonic_coeff(x, x + half_h, visc_coeff, visc_porous, physics_cfg)
        mu_w = get_exact_harmonic_coeff(x - half_h, x, visc_coeff, visc_porous, physics_cfg)
        mu_y = alpha_P * visc_porous + (1.0 - alpha_P) * visc_coeff
        diff_u_oper = (
            (mu_e * (u_P - u_e) + mu_w * (u_P - u_w)) / (half_h**2)
            + mu_y * (2.0 * u_P - u_n - u_s) / (half_h**2)
        )
        diff_v_oper = (
            (mu_e * (v_P - v_e) + mu_w * (v_P - v_w)) / (half_h**2)
            + mu_y * (2.0 * v_P - v_n - v_s) / (half_h**2)
        )

    # 9. Tổng hợp thặng dư PDE
    ru = conv_u + dp_dx + diff_u_oper + f_u_drag
    rv = conv_v + dp_dy + diff_v_oper + f_v_drag - f_buoyancy
    rT = conv_T + diff_T_oper

    return {"u": ru, "v": rv, "p": rc, "T": rT}


# =============================================================================
# 5. SƠ ĐỒ CAN-FFV
# Phiên bản thử nghiệm từ FFV-PINN. Tân dụng cơ chế tính đạo hàm AD tại điểm bất kỳ 
# của PINN dể có thể tính scheme bậc cao trong finite volume truyền thống.
# =============================================================================
def compute_can_ffv_pde_residuals(
    neural_net_fn, params, x, y, h, physics_cfg
):
    """Sơ đồ CAN-FFV 5 điểm tâm ô [P, E, W, N, S] (khoảng cách dx = dy = h):
    - Nội suy giá trị mặt (u_e, v_e, p_e, T_e) bằng đa thức Hermite bậc 3 (O(h^4)):
      phi_e = 0.5*(phi_P + phi_E) - (h/8)*(phi_x_E - phi_x_P).
    - Nội suy đạo hàm mặt cho toán tử khuếch tán bằng đạo hàm Hermite bậc 3 (Qiu & Shu, 2003):
      dphi_dx|_e = 1.5*(phi_E - phi_P)/h - 0.25*(phi_x_P + phi_x_E) -> đạt bậc 4 O(h^4)
      và tăng cường hệ số đường chéo dương a_P = 6*visc/h^2 > 0 cho PTS.
    """
    dx = dy = h
    half_h = 0.5 * h

    # 1. Gom tọa độ 5 TÂM Ô: [0: P, 1: E, 2: W, 3: N, 4: S]
    x_all = jnp.stack([x, x + dx, x - dx, x, x])
    y_all = jnp.stack([y, y, y, y + dy, y - dy])

    # 2. Đánh giá giá trị và đạo hàm bậc 1 Forward-mode AD trên 5 tâm ô
    val_and_grad_single = value_and_jacfwd(neural_net_fn, (1, 2))
    eval_fn = jax.vmap(val_and_grad_single, in_axes=(None, 0, 0))

    (u_all, v_all, p_all, T_all), (d_x_all, d_y_all) = eval_fn(
        params, x_all, y_all
    )
    u_x_all, v_x_all, p_x_all, T_x_all = d_x_all
    u_y_all, v_y_all, p_y_all, T_y_all = d_y_all

    u_P, u_E, u_W, u_N, u_S = u_all
    v_P, v_E, v_W, v_N, v_S = v_all
    p_P, p_E, p_W, p_N, p_S = p_all
    T_P, T_E, T_W, T_N, T_S = T_all

    u_x_P, u_x_E, u_x_W = u_x_all[0], u_x_all[1], u_x_all[2]
    v_x_P, v_x_E, v_x_W = v_x_all[0], v_x_all[1], v_x_all[2]
    p_x_P, p_x_E, p_x_W = p_x_all[0], p_x_all[1], p_x_all[2]
    T_x_P, T_x_E, T_x_W = T_x_all[0], T_x_all[1], T_x_all[2]

    u_y_P, u_y_N, u_y_S = u_y_all[0], u_y_all[3], u_y_all[4]
    v_y_P, v_y_N, v_y_S = v_y_all[0], v_y_all[3], v_y_all[4]
    p_y_P, p_y_N, p_y_S = p_y_all[0], p_y_all[3], p_y_all[4]
    T_y_P, T_y_N, T_y_S = T_y_all[0], T_y_all[3], T_y_all[4]

    # 3. Tái cấu trúc giá trị tại 4 mặt (e, w, n, s) bằng đa thức Hermite bậc 3 (O(h^4))
    u_e = 0.5 * (u_P + u_E) - (dx / 8.0) * (u_x_E - u_x_P)
    u_w = 0.5 * (u_W + u_P) - (dx / 8.0) * (u_x_P - u_x_W)
    u_n = 0.5 * (u_P + u_N) - (dy / 8.0) * (u_y_N - u_y_P)
    u_s = 0.5 * (u_S + u_P) - (dy / 8.0) * (u_y_P - u_y_S)

    v_e = 0.5 * (v_P + v_E) - (dx / 8.0) * (v_x_E - v_x_P)
    v_w = 0.5 * (v_W + v_P) - (dx / 8.0) * (v_x_P - v_x_W)
    v_n = 0.5 * (v_P + v_N) - (dy / 8.0) * (v_y_N - v_y_P)
    v_s = 0.5 * (v_S + v_P) - (dy / 8.0) * (v_y_P - v_y_S)

    T_e = 0.5 * (T_P + T_E) - (dx / 8.0) * (T_x_E - T_x_P)
    T_w = 0.5 * (T_W + T_P) - (dx / 8.0) * (T_x_P - T_x_W)
    T_n = 0.5 * (T_P + T_N) - (dy / 8.0) * (T_y_N - T_y_P)
    T_s = 0.5 * (T_S + T_P) - (dy / 8.0) * (T_y_P - T_y_S)

    p_e = 0.5 * (p_P + p_E) - (dx / 8.0) * (p_x_E - p_x_P)
    p_w = 0.5 * (p_W + p_P) - (dx / 8.0) * (p_x_P - p_x_W)
    p_n = 0.5 * (p_P + p_N) - (dy / 8.0) * (p_y_N - p_y_P)
    p_s = 0.5 * (p_S + p_P) - (dy / 8.0) * (p_y_P - p_y_S)

    # 4. Phương trình liên tục (Continuity)
    div = (u_e - u_w) / dx + (v_n - v_s) / dy
    rc = div

    # 5. Tỷ lệ thể tích ô kiểm soát [x - h/2, x + h/2] nằm trong miền xốp (alpha_P)
    eps_p = float(physics_cfg.get("epsilon", 0.4))
    T_ref = float(physics_cfg.get("T_ref", 0.5))
    Pr = float(physics_cfg.get("Pr", 6.44))
    Ra = float(physics_cfg.get("Ra", 3.7e6))
    visc_coeff = jnp.sqrt(Pr / Ra)

    alpha_P = get_porous_segment_fraction(x - half_h, x + half_h, physics_cfg)
    eps_cell_avg = alpha_P * eps_p + (1.0 - alpha_P) * 1.0
    inv_eps_cell = 1.0 / jnp.maximum(eps_cell_avg, 1e-6)

    # 6. Số hạng đối lưu bảo toàn qua mặt (Không trừ phi_P * div)
    inv_eps_e = 1.0 / get_eps_local(x + half_h, physics_cfg)
    inv_eps_w = 1.0 / get_eps_local(x - half_h, physics_cfg)
    inv_eps_n = inv_eps_cell
    inv_eps_s = inv_eps_cell

    conv_u = (
        ((u_e**2) * inv_eps_e - (u_w**2) * inv_eps_w) / dx
        + ((u_n * v_n) * inv_eps_n - (u_s * v_s) * inv_eps_s) / dy
    )
    conv_v = (
        ((u_e * v_e) * inv_eps_e - (u_w * v_w) * inv_eps_w) / dx
        + ((v_n**2) * inv_eps_n - (v_s**2) * inv_eps_s) / dy
    )
    conv_T = (
        ((u_e * T_e) - (u_w * T_w)) / dx
        + ((v_n * T_n) - (v_s * T_s)) / dy
    )

    # 7. Gradient áp suất từ giá trị áp suất mặt Hermite (Chiu et al., Eq. 12-13)
    dp_dx = (p_e - p_w) / dx
    dp_dy = (p_n - p_s) / dy

    # 8. Khuếch tán động lượng dùng Đạo hàm mặt Hermite bậc 3 (Qiu & Shu, 2003 - Step 2):
    # -d/dx(du/dx) = (du_dx|_w - du_dx|_e) / dx
    # với du_dx|_e = 1.5*(u_E - u_P)/dx - 0.25*(u_x_P + u_x_E)
    diff_u_x = 1.5 * (2.0 * u_P - u_E - u_W) / (dx**2) + 0.25 * (u_x_E - u_x_W) / dx
    diff_u_y = 1.5 * (2.0 * u_P - u_N - u_S) / (dy**2) + 0.25 * (u_y_N - u_y_S) / dy
    diff_u_oper = visc_coeff * (diff_u_x + diff_u_y)

    diff_v_x = 1.5 * (2.0 * v_P - v_E - v_W) / (dx**2) + 0.25 * (v_x_E - v_x_W) / dx
    diff_v_y = 1.5 * (2.0 * v_P - v_N - v_S) / (dy**2) + 0.25 * (v_y_N - v_y_S) / dy
    diff_v_oper = visc_coeff * (diff_v_x + diff_v_y)

    # 9. Khuếch tán nhiệt kết hợp Exact Harmonic Mean và Đạo hàm mặt Hermite:
    # - Khi đoạn [x_P, x_E] nằm trong vùng trơn (theta_e == 0 hoặc 1): dùng đạo hàm Hermite bậc 4.
    # - Khi đoạn [x_P, x_E] bước qua mặt phân cách (0 < theta_e < 1): dùng hiệu số FVM thuần
    #   với k_e_exact để bảo toàn tuyệt đối bước nhảy đạo hàm nhiệt độ!
    theta_e = get_porous_segment_fraction(x, x + dx, physics_cfg)
    theta_w = get_porous_segment_fraction(x - dx, x, physics_cfg)

    smooth_e = jnp.where(jnp.logical_and(theta_e > 1e-5, theta_e < (1.0 - 1e-5)), 0.0, 1.0)
    smooth_w = jnp.where(jnp.logical_and(theta_w > 1e-5, theta_w < (1.0 - 1e-5)), 0.0, 1.0)

    k_e_exact = get_exact_harmonic_k(x, x + dx, physics_cfg)
    k_w_exact = get_exact_harmonic_k(x - dx, x, physics_cfg)

    # Đạo hàm nhiệt độ hướng vào tâm P tại mặt e: (-dT/dx)|_e
    neg_dT_dx_e = (1.0 + 0.5 * smooth_e) * (T_P - T_E) / dx + 0.25 * smooth_e * (T_x_P + T_x_E)
    # Đạo hàm nhiệt độ hướng vào tâm P tại mặt w: (+dT/dx)|_w
    pos_dT_dx_w = (1.0 + 0.5 * smooth_w) * (T_P - T_W) / dx - 0.25 * smooth_w * (T_x_W + T_x_P)

    k_fluid = 1.0 / jnp.sqrt(Pr * Ra)
    lambda_k = float(physics_cfg.get("lambda_k", 1.0))
    k_y_avg = (alpha_P * lambda_k + (1.0 - alpha_P)) * k_fluid

    diff_T_x = (k_e_exact * neg_dT_dx_e + k_w_exact * pos_dT_dx_w) / dx
    diff_T_y = k_y_avg * (
        1.5 * (2.0 * T_P - T_N - T_S) / (dy**2) + 0.25 * (T_y_N - T_y_S) / dy
    )
    diff_T_oper = diff_T_x + diff_T_y

    # 10. Lực cản Darcy-Forchheimer & Lực nổi tích phân trên hai nửa ô [w, P] và [P, e]
    alpha_left = get_porous_segment_fraction(x - half_h, x, physics_cfg)
    alpha_right = get_porous_segment_fraction(x, x + half_h, physics_cfg)

    fu_pure_P, fv_pure_P = compute_pure_porous_drag(u_P, v_P, physics_cfg)
    fu_pure_e, fv_pure_e = compute_pure_porous_drag(u_e, v_e, physics_cfg)
    fu_pure_w, fv_pure_w = compute_pure_porous_drag(u_w, v_w, physics_cfg)

    f_u_drag = 0.5 * alpha_left * (0.5 * fu_pure_w + 0.5 * fu_pure_P) + (
        0.5 * alpha_right * (0.5 * fu_pure_P + 0.5 * fu_pure_e)
    )
    f_v_drag = 0.5 * alpha_left * (0.5 * fv_pure_w + 0.5 * fv_pure_P) + (
        0.5 * alpha_right * (0.5 * fv_pure_P + 0.5 * fv_pure_e)
    )
    f_buoyancy = eps_cell_avg * (T_P - T_ref)

    # 11. Tổng hợp thặng dư PDE
    ru = conv_u + dp_dx + diff_u_oper + f_u_drag
    rv = conv_v + dp_dy + diff_v_oper + f_v_drag - f_buoyancy
    rT = conv_T + diff_T_oper

    return {"u": ru, "v": rv, "p": rc, "T": rT}


# =============================================================================
# 6. Scheme selection function to compute PDE residuals 
# =============================================================================
def compute_pde_residuals_nd(neural_net_fn, params, x, y, h, physics_cfg, scheme="ffv"):
    scheme_clean = str(scheme).lower().strip()
    if scheme_clean == "ffv":
        return compute_ffv_pde_residuals(neural_net_fn, params, x, y, h, physics_cfg)
    elif scheme_clean == "can_ffv":
        return compute_can_ffv_pde_residuals(neural_net_fn, params, x, y, h, physics_cfg)
    elif scheme_clean == "ffv_continuous":
        return compute_ffv_continuous_pde_residuals(neural_net_fn, params, x, y, h, physics_cfg)
    elif scheme_clean == "ffv_compact":
        return compute_ffv_compact_pde_residuals(neural_net_fn, params, x, y, h, physics_cfg)
    else:
        raise ValueError(
            f"Invalid scheme: '{scheme}'. Please choose 'ffv', 'ffv_continuous', 'ffv_compact', or 'can_ffv'."
        )