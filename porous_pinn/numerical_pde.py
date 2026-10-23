"""Finite-volume PDE residuals of the one-domain fluid/porous natural-convection model.

A residual at a collocation point (x, y) is the discrete balance over the square control
volume [x - h/2, x + h/2] x [y - h/2, y + h/2], divided by its area h^2. Face values are
evaluated directly by the network instead of being interpolated from cell centres (the
"simplified FVM" of Wei et al. 2025, Sec. 2.3.1, Eqs. (15)-(16)), so no automatic
differentiation is needed and control volumes may be placed anywhere.

Model (steady and dimensionless: length L, velocity sqrt(g beta dT L), so that
nu = sqrt(Pr/Ra) and alpha_f = 1/sqrt(Ra Pr)). This is Gao et al. (2017), Eqs. (1)-(6),
with effective viscosity nu_e = nu and pressure variable p = eps * p_Gao:
    div u = 0
    (1/eps) div(u u) + grad p - nu lap u
        + eps (nu/K + F_eps |u| / sqrt(K)) u - eps (T - T_ref) e_y = 0
    div(u T) - div(alpha_e grad T) = 0
eps, K (= Da), F_eps and alpha_e (= lambda_k alpha_f) take their porous values for
porous_x_start <= x <= porous_x_end; elsewhere eps = 1, 1/K = 0 and alpha_e = alpha_f.

Interfaces. The coefficients jump at the fluid/porous interfaces. A one-domain model is
equivalent to the two-domain (Darcy-Brinkman) model only if these jumps are differentiated
in the sense of distributions (Hirata et al. 2009). The integral (FV) form does this
naturally: the face fluxes impose continuity of T and of the normal heat flux
(Gao et al. 2017, Eqs. (8)-(9)) without explicit interface conditions. Pointwise AD
residuals do not (see PorousNaturalConvection2D.r_net).

Schemes (config.diff.scheme): "ffv", "ffv_continuous", "ffv_compact" (recommended)
and "can_ffv"; see compute_pde_residuals_nd.

References (PDFs in references/ unless noted):
    Gao, Chen, Chen & Zhang (2017). A modified lattice Boltzmann model for conjugate heat
        transfer in porous media. Int. J. Heat Mass Transfer 105, 673-683. [lbm_ref.pdf]
    Wei, Fan, Wong, Ooi, Wang & Chiu (2025). FFV-PINN: A fast physics-informed neural
        network with simplified finite volume discretization and residual correction.
        Comput. Methods Appl. Mech. Engrg. 444, 118139. [FFV-PINN.pdf]
    Hirata, Goyeau, Gobin, Chandesris & Jamet (2009). Stability of natural convection in
        superposed fluid and porous layers: equivalence of the one- and two-domain
        approaches. Int. J. Heat Mass Transfer 52, 533-536. [fvm_porus_ref.pdf]
    Beckermann, Viskanta & Ramadhyani (1988). Natural convection in vertical enclosures
        containing simultaneously fluid and porous layers. J. Fluid Mech. 186, 257-284.
        Source of the case parameters (Table 1). [JFM1988.pdf]
    Patankar (1980). Numerical Heat Transfer and Fluid Flow, Ch. 4 (harmonic-mean
        interface conductivity). [not in references/]
"""
import jax
import jax.numpy as jnp
from jaxpi.derivatives import value_and_jacfwd


# =============================================================================
# 1. Auxiliary functions for porous media and buoyancy
# =============================================================================
def is_in_porous_zone(x, physics_cfg):
    """Mask of the points that lie in the porous layer.

    Args:
        x: x-coordinates, any shape.
        physics_cfg: config.physics. Uses porous_x_start and porous_x_end; their defaults
            (0 and 1) describe a cavity filled entirely with porous medium.

    Returns:
        Boolean array, True where porous_x_start <= x <= porous_x_end. Points exactly on an
        interface count as porous.
    """
    x_start = float(physics_cfg.get("porous_x_start", 0.0))
    x_end = float(physics_cfg.get("porous_x_end", 1.0))
    return jnp.logical_and(x >= x_start, x <= x_end)


def get_eps_local(x, physics_cfg):
    """Porosity eps(x): physics.epsilon in the porous layer, 1 in the clear fluid.

    Args:
        x: x-coordinates, any shape.
        physics_cfg: config.physics.

    Returns:
        Array of eps values with the shape of x.
    """
    eps_p = float(physics_cfg.get("epsilon", 0.4))
    return jnp.where(is_in_porous_zone(x, physics_cfg), eps_p, 1.0)


def get_k_local(x, physics_cfg):
    """Dimensionless effective thermal diffusivity alpha_e(x).

    alpha_f = 1/sqrt(Ra Pr) in the fluid and lambda_k * alpha_f in the porous layer, where
    lambda_k = k_e/k_f is the effective-to-fluid conductivity ratio (Gao et al. 2017,
    Eq. (6); values from Beckermann et al. 1988, Table 1). The heat-capacity ratio sigma of
    Gao Eq. (7) multiplies only dT/dt and drops out of the steady problem.

    Args:
        x: x-coordinates, any shape.
        physics_cfg: config.physics (Pr, Ra, lambda_k and the layer bounds).

    Returns:
        Array of alpha_e values with the shape of x.
    """
    Pr = float(physics_cfg.get("Pr", 6.44))
    Ra = float(physics_cfg.get("Ra", 3.7e6))
    lambda_k = float(physics_cfg.get("lambda_k", 1.0))
    k_fluid = 1.0 / jnp.sqrt(Pr * Ra)
    return jnp.where(is_in_porous_zone(x, physics_cfg), lambda_k * k_fluid, k_fluid)


def compute_porous_and_buoyancy(u, v, T, x, physics_cfg):
    """Pointwise porous-medium coefficients and body forces.

    Evaluates the source terms of Gao et al. (2017), Eqs. (2), (4) and (5), at a single
    point without cell averaging, in the scaling of the module docstring:
        drag     = eps (nu/K + F_eps |u| / sqrt(K)) u    (1/K = 1/Da in the layer, 0 elsewhere)
        buoyancy = eps (T - T_ref)                       (acts on the y-momentum equation)
    |u| is regularised as sqrt(u^2 + v^2 + 1e-6), which keeps its derivative finite at u = 0.
    Used by the AD residual and by the "ffv" scheme.

    Args:
        u, v, T: network outputs at the point.
        x: x-coordinate; selects the porous or fluid coefficients.
        physics_cfg: config.physics (epsilon, Da, Pr, Ra, F_eps, T_ref, lambda_k).

    Returns:
        Tuple (inv_eps, visc_coeff, diff_T_local, f_u_drag, f_v_drag, f_buoyancy): 1/eps(x),
        nu = sqrt(Pr/Ra), alpha_e(x), the x and y drag components and the buoyancy force.
    """
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
    """Fraction of the segment [x_a, x_b] that lies in the porous layer.

    Exact for any x_a < x_b, so it serves both grid-aligned and randomly placed control
    volumes. Valid only for a layer bounded by the vertical planes x = porous_x_start and
    x = porous_x_end; other porous geometries need a different overlap computation.

    Args:
        x_a, x_b: segment end points with x_a < x_b; broadcastable arrays.
        physics_cfg: config.physics.

    Returns:
        theta in [0, 1], with the broadcast shape of x_a and x_b.
    """
    x_start = float(physics_cfg.get("porous_x_start", 0.0))
    x_end = float(physics_cfg.get("porous_x_end", 1.0))
    length = jnp.maximum(x_b - x_a, 1e-12)

    overlap = jnp.maximum(0.0, jnp.minimum(x_b, x_end) - jnp.maximum(x_a, x_start))
    return jnp.clip(overlap / length, 0.0, 1.0)


def get_porous_subsegment(x_a, x_b, physics_cfg):
    """Porous part of the segment [x_a, x_b], for one-point Gauss quadrature on it.

    The drag of Gao et al. (2017), Eq. (4), is zero in the fluid and smooth inside the layer.
    Integrating it over [x_a, x_b] with the midpoint rule on the porous part alone is
    therefore second-order accurate, even when an interface cuts the segment
    (tests/test_fv_schemes.py::TestDragCellAverage).

    Args:
        x_a, x_b: segment end points with x_a < x_b; broadcastable arrays.
        physics_cfg: config.physics.

    Returns:
        Tuple (length, midpoint): the length of the porous sub-segment (0 if there is none)
        and its midpoint. When the length is 0 the midpoint of [x_a, x_b] is returned, so
        the network is still evaluated at a valid point in the domain; its quadrature
        weight is then zero.
    """
    x_start = float(physics_cfg.get("porous_x_start", 0.0))
    x_end = float(physics_cfg.get("porous_x_end", 1.0))
    lo = jnp.maximum(x_a, x_start)
    hi = jnp.minimum(x_b, x_end)
    length = jnp.maximum(hi - lo, 0.0)
    midpoint = jnp.where(length > 0.0, 0.5 * (lo + hi), 0.5 * (x_a + x_b))
    return length, midpoint


def get_exact_harmonic_k(x_a, x_b, physics_cfg):
    """Thermal diffusivity for the flux between x_a and x_b, exact across an interface.

    Steady 1-D conduction through [x_a, x_b], a fraction theta of which is porous, is a
    series of two thermal resistances. The equivalent diffusivity is therefore the
    distance-weighted harmonic mean
        k_eff = 1 / (theta / k_porous + (1 - theta) / k_fluid),
    and the flux k_eff (T_b - T_a) / (x_b - x_a) is exact for a piecewise-linear profile
    whose normal flux is continuous at the interface (Gao et al. 2017, Eqs. (8)-(9)).
    With the interface midway (theta = 1/2) this reduces to the harmonic mean
    2 k_p k_f / (k_p + k_f) of Patankar (1980, Ch. 4).

    Args:
        x_a, x_b: segment end points with x_a < x_b; broadcastable arrays.
        physics_cfg: config.physics (Pr, Ra, lambda_k and the layer bounds).

    Returns:
        k_eff with the broadcast shape of x_a and x_b.
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
    """Distance-weighted harmonic mean of any piecewise-constant diffusion coefficient.

    Generalises get_exact_harmonic_k to a coefficient c that equals c_fluid in the fluid and
    c_porous in the layer. Used for the effective viscosity nu/eps of the "hirata"
    momentum form.

    Args:
        x_a, x_b: segment end points with x_a < x_b; broadcastable arrays.
        c_fluid, c_porous: coefficient values in the fluid and in the porous layer.
        physics_cfg: config.physics.

    Returns:
        1 / (theta / c_porous + (1 - theta) / c_fluid), theta from
        get_porous_segment_fraction.
    """
    theta = get_porous_segment_fraction(x_a, x_b, physics_cfg)
    return 1.0 / (theta / c_porous + (1.0 - theta) / c_fluid)


def compute_pure_porous_drag(u, v, physics_cfg):
    """Darcy-Forchheimer drag for a point fully inside the porous layer.

    Returns eps (nu/K + F_eps |u| / sqrt(K)) u with the porous values of eps and K = Da
    (Gao et al. 2017, Eq. (4)). The caller weights it by the porous fraction of the
    control volume, so the drag vanishes in the fluid.

    Args:
        u, v: velocity at the point.
        physics_cfg: config.physics (epsilon, Da, Pr, Ra, F_eps).

    Returns:
        Tuple (f_u, f_v) of drag components.
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
    """"ffv" residuals on the 9-point stencil [P, e, w, n, s, E, W, N, S].

    Adapted from the simplified FVM of Wei et al. (2025), Sec. 2.3.1. Face values
    (e, w, n, s, at distance h/2) come from the network. Diffusion uses the cell-centre
    neighbours (E, W, N, S, at distance h) as in their Eq. (6).
        - Convection: face fluxes minus phi_P * div u.
        - Pressure gradient: (p_e - p_w)/h and (p_n - p_s)/h.
        - Heat conduction: harmonic mean of the centre values k_P and k_E (Patankar 1980,
          Ch. 4). This is exact only when the interface lies midway between P and E.
        - Drag: trapezoidal rule (weights 1/4, 1/2, 1/4) over [w, P, e] of the pointwise drag.
          It is first order in cells cut by an interface.
        - Buoyancy: evaluated at P.
    The stencil reaches h, so control-volume centres must stay at least h away from the
    walls (porous_pinn.batching.create_batches).

    Args:
        neural_net_fn: callable (params, x, y) -> (u, v, p, T).
        params: network parameters.
        x, y: scalar centre of the control volume (the caller vmaps over points).
        h: control-volume width, equal to the stencil spacing.
        physics_cfg: config.physics.

    Returns:
        Dict of residuals {"u": x-momentum, "v": y-momentum, "p": continuity, "T": energy}.
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
    """"ffv_continuous" residuals: the 9-point "ffv" stencil, valid for any control-volume position.

    Differs from compute_ffv_pde_residuals so that a control volume may be cut anywhere
    by an interface:
        - Convection in conservative form (no phi_P * div u term). 1/eps is taken at the e
          and w faces, and as the cell average on the n and s faces.
        - Pressure gradient from face pressures, (p_e - p_w)/h and (p_n - p_s)/h, which is
          consistent with the continuity residual (u_e - u_w)/h + (v_n - v_s)/h.
        - Heat conduction through e and w with the distance-weighted harmonic mean over
          [P, E] and [W, P] (get_exact_harmonic_k). The n and s faces, parallel to the
          interface, use the volume-fraction average.
        - Drag and buoyancy: trapezoidal rule on the half cells [w, P] and [P, e], each weighted
          by its porous fraction, plus a correction in y from the n and s values.
    Diffusion still uses neighbours at distance h, so centres must stay at least h away from
    the walls. The drag in a cut cell is first order, because each half-cell rule mixes
    points from both sides of the interface (tests/test_fv_schemes.py::TestDragCellAverage,
    marked as a strict xfail).

    Args:
        neural_net_fn: callable (params, x, y) -> (u, v, p, T).
        params: network parameters.
        x, y: scalar centre of the control volume.
        h: control-volume width, equal to the stencil spacing.
        physics_cfg: config.physics.

    Returns:
        Dict of residuals {"u": x-momentum, "v": y-momentum, "p": continuity, "T": energy}.
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
def compute_ffv_compact_pde_residuals(neural_net_fn, params, x, y, h, physics_cfg,
                                      return_stencil=False):
    """"ffv_compact" residuals: 5-point stencil [P, e, w, n, s] plus two drag quadrature points.

    Every term uses values at distance h/2 from P only. Control volumes
    [x - h/2, x + h/2] x [y - h/2, y + h/2] can therefore tile the whole cavity, including the
    cells next to the walls, whose outer face then lies on the wall. The scheme is second
    order both in the bulk and in cells cut by an interface (tests/test_fv_schemes.py).
        - Convection, pressure gradient and buoyancy: as in
          compute_ffv_continuous_pde_residuals.
        - Diffusion: face flux c_f (phi_f - phi_P) / (h/2), where c_f is the
          distance-weighted harmonic mean over the half cell. This is exact for
          piecewise-linear profiles with continuous flux across the interface
          (Gao et al. 2017, Eqs. (8)-(9)).
        - Drag: cell average equal to 1/h times the integral over the porous part of each
          half cell, computed with a one-point Gauss rule on that part
          (get_porous_subsegment). This costs two extra network evaluations.

    physics.momentum_form selects how the one-domain momentum equation is written. Inside
    each region both forms describe the same physics. They differ only in the interface
    conditions they imply when integrated over a cut control volume (Hirata et al. 2009):
        "gao" (default): Gao et al. (2017), Eq. (2) with Eq. (4), nu_e = nu and p = eps p_Gao:
            (1/eps) div(u u) + grad p - nu lap u + eps drag - eps (T - T_ref) e_y = 0.
            nu is constant across the interface, so the scheme imposes continuity of the
            normal derivative of the tangential velocity, and of p = eps p_Gao.
        "hirata": the same equation divided by eps, with the discontinuous coefficient
            inside the divergence (Hirata et al. 2009, Eq. (2), mu_eff/mu = 1/eps):
            (1/eps) div(u u / eps) + grad P - div((nu/eps) grad u) + drag - (T - T_ref) e_y = 0.
            The scheme then imposes the shear condition of the two-domain Darcy-Brinkman
            model, nu du_t/dn (fluid) = (nu/eps) du_t/dn (porous) (Hirata et al. 2009,
            Eq. (15)). The intrinsic pressure P = p_Gao is continuous up to the jump in
            viscous normal stress (cf. their Eq. (14), which uses the stress form).
            The viscosity on the e and w faces is the harmonic mean of nu/eps. On the n and s
            faces, which are parallel to the interface, it is the volume-fraction average.
    Only this scheme implements "hirata"; PorousNaturalConvection2D.r_net rejects the
    other combinations.

    Args:
        neural_net_fn: callable (params, x, y) -> (u, v, p, T).
        params: network parameters.
        x, y: scalar centre of the control volume; h/2 <= x, y <= L - h/2.
        h: control-volume width.
        physics_cfg: config.physics; momentum_form is "gao" or "hirata".
        return_stencil: also return the network values on the 5-point stencil, so that
            the caller can form the control-volume average of PorousNaturalConvection2D.sol_net
            without evaluating the network a second time.

    Returns:
        Dict of residuals {"u": x-momentum, "v": y-momentum, "p": continuity, "T": energy}.
        With return_stencil, the tuple (residuals, (u_s, v_s, p_s, T_s)), where each
        stencil array has shape (5,) and holds the values at [P, e, w, n, s].

    Raises:
        ValueError: if physics.momentum_form is neither "gao" nor "hirata".
    """
    dx = dy = h
    half_h = 0.5 * h
    momentum_form = str(physics_cfg.get("momentum_form", "gao")).lower()
    if momentum_form not in ("gao", "hirata"):
        raise ValueError(f"physics.momentum_form='{momentum_form}' phải là 'gao' hoặc 'hirata'.")

    # 1. Stencil 5 điểm [P, e, w, n, s] + 2 điểm Gauss của lực cản: trung điểm phần xốp
    #    của hai nửa ô [x - h/2, x] và [x, x + h/2]
    len_l, xg_l = get_porous_subsegment(x - half_h, x, physics_cfg)
    len_r, xg_r = get_porous_subsegment(x, x + half_h, physics_cfg)
    x_all = jnp.stack([x, x + half_h, x - half_h, x, x, xg_l, xg_r])
    y_all = jnp.stack([y, y, y, y + half_h, y - half_h, y, y])

    eval_fn = jax.vmap(neural_net_fn, in_axes=(None, 0, 0))
    u_all, v_all, p_all, T_all = eval_fn(params, x_all, y_all)

    u_P, u_e, u_w, u_n, u_s, u_gl, u_gr = u_all
    v_P, v_e, v_w, v_n, v_s, v_gl, v_gr = v_all
    p_P, p_e, p_w, p_n, p_s = p_all[:5]
    T_P, T_e, T_w, T_n, T_s = T_all[:5]

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

    # 8. Lực cản Darcy-Forchheimer: trung bình ô = (1/h) * tích phân trên RIÊNG phần xốp,
    #    Gauss 1 điểm trên mỗi đoạn con xốp (bậc 2, kể cả ô cắt mặt phân cách); theo phương y
    #    dùng trung điểm y_P. Lực nổi: tích phân theo phần thể tích như ffv_continuous
    #    (T liên tục qua mặt phân cách nên quy tắc này vẫn bậc 2).
    alpha_left = get_porous_segment_fraction(x - half_h, x, physics_cfg)
    alpha_right = get_porous_segment_fraction(x, x + half_h, physics_cfg)

    fu_gl, fv_gl = compute_pure_porous_drag(u_gl, v_gl, physics_cfg)
    fu_gr, fv_gr = compute_pure_porous_drag(u_gr, v_gr, physics_cfg)
    f_u_drag = (len_l * fu_gl + len_r * fu_gr) / dx
    f_v_drag = (len_l * fv_gl + len_r * fv_gr) / dx

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

    res = {"u": ru, "v": rv, "p": rc, "T": rT}
    if return_stencil:
        return res, (u_all[:5], v_all[:5], p_all[:5], T_all[:5])
    return res


# =============================================================================
# 5. SƠ ĐỒ CAN-FFV
# Phiên bản thử nghiệm từ FFV-PINN. Tân dụng cơ chế tính đạo hàm AD tại điểm bất kỳ 
# của PINN dể có thể tính scheme bậc cao trong finite volume truyền thống.
# =============================================================================
def compute_can_ffv_pde_residuals(
    neural_net_fn, params, x, y, h, physics_cfg
):
    """"can_ffv" residuals: 5 cell centres [P, E, W, N, S] with forward-mode AD gradients.

    Experimental variant; it is not used for the paper results. Face values are
    reconstructed from cell-centre values and gradients by cubic Hermite interpolation,
        phi_e = (phi_P + phi_E)/2 - (h/8) (dphi/dx|_E - dphi/dx|_P),
    and the face derivatives of the diffusion term by the derivative of the same polynomial,
        dphi/dx|_e = 1.5 (phi_E - phi_P)/h - 0.25 (dphi/dx|_P + dphi/dx|_E).
    The resulting diagonal coefficient a_P = 6 nu/h^2 is positive, as pseudo-time stepping
    requires. Where [P, E] crosses an interface, the heat flux falls back to the
    harmonic-mean difference, because the Hermite derivative assumes a smooth T. Drag uses
    the half-cell trapezoidal rule (first order in cut cells); buoyancy is evaluated at P with
    the cell-averaged porosity. The inline comments cite Qiu & Shu (2003) and Chiu et al.
    for the Hermite formulas; these papers are not in references/.

    Args:
        neural_net_fn: callable (params, x, y) -> (u, v, p, T).
        params: network parameters.
        x, y: scalar centre of the control volume.
        h: spacing between cell centres.
        physics_cfg: config.physics.

    Returns:
        Dict of residuals {"u": x-momentum, "v": y-momentum, "p": continuity, "T": energy}.
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
    """Finite-volume residual of the control volume centred at (x, y) for the chosen scheme.

    Args:
        neural_net_fn: callable (params, x, y) -> (u, v, p, T).
        params: network parameters.
        x, y: scalar centre of the control volume.
        h: control-volume width (config.diff.h_stencil).
        physics_cfg: config.physics.
        scheme: "ffv", "ffv_continuous", "ffv_compact" or "can_ffv" (case-insensitive).
            "ad" is handled in PorousNaturalConvection2D.r_net instead.

    Returns:
        Dict of residuals {"u", "v", "p", "T"}.

    Raises:
        ValueError: for an unknown scheme.
    """
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