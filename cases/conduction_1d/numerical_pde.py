"""Finite-volume residuals of 1-D two-layer conduction (Gao et al. 2017, Sec. 4.1).

Both schemes write the balance of the cell [x - h/2, x + h/2] with face fluxes
k_f (T_neighbour - T_C) / h, where k_f is the conductivity at the face point. These fluxes
are exact for the piecewise-linear solution when D coincides with a cell centre, because
each segment [W, C] and [C, E] then lies in one material. This holds for D = 0.5 on the
default grid. For a segment that straddles D the correct coefficient is the
distance-weighted harmonic mean (porous_pinn.numerical_pde.get_exact_harmonic_k), so
randomly placed cells (diff.use_continuous_mesh) are not exact near D.
"""
import jax
import jax.numpy as jnp


def get_k_local(x, D, k1, k2):
    """Conductivity k1 for x <= D and k2 for x > D."""
    return jnp.where(x <= D, k1, k2)


def compute_can_pde_residuals_1d(T_net_fn, params, x, h, D, k1, k2):
    """Cell balance divided by h: (flux_w - flux_e) / h, an approximation of -d/dx(k dT/dx).

    The diagonal Jacobian entry dR/dT_C = (k_w + k_e) / h^2 is positive, the sign that
    pseudo-time stepping expects.

    Args:
        T_net_fn: callable (params, x) -> T.
        params: network parameters.
        x: scalar cell centre.
        h: cell width.
        D: interface position.
        k1, k2: conductivities on each side of D.

    Returns:
        Dict {"T": residual}.
    """
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
    """Cell balance multiplied by h: k_w (T_C - T_W) - k_e (T_E - T_C).

    Equal to h^2 times the residual of compute_can_pde_residuals_1d, so it scales like T
    rather than T / h^2.

    Args:
        T_net_fn: callable (params, x) -> T.
        params: network parameters.
        x: scalar cell centre.
        h: cell width.
        D: interface position.
        k1, k2: conductivities on each side of D.

    Returns:
        Dict {"T": residual}.
    """
    dx = h
    x_all = jnp.stack([x, x + dx, x - dx])  # [C, E, W]

    eval_fn = jax.vmap(T_net_fn, in_axes=(None, 0))
    T_C, T_E, T_W = eval_fn(params, x_all)

    k_e = get_k_local(x + 0.5 * dx, D, k1, k2)
    k_w = get_k_local(x - 0.5 * dx, D, k1, k2)

    rT = k_w * (T_C - T_W) - k_e * (T_E - T_C)
    return {"T": rT}