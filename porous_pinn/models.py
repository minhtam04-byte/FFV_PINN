"""PINN model for steady natural convection in a cavity with a vertical porous layer.

The benchmark is Gao et al. (2017), Sec. 4.3 (references/lbm_ref.pdf), built on the
experiments and simulations of Beckermann et al. (1988) (references/JFM1988.pdf). The
residuals come either from automatic differentiation or from a finite-volume scheme in
porous_pinn.numerical_pde, whose docstring states the governing equations.
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap

from jaxpi.derivatives import hessian_diag_fwd, value_and_jacfwd
from jaxpi.models import ForwardBVP
from porous_pinn.boundary import sample_boundaries_natural_convection
from porous_pinn.numerical_pde import (
    compute_pde_residuals_nd,
    compute_porous_and_buoyancy,
)


def uses_interface_features(config):
    """Whether the network also receives the interface features |x - x_s| and |x - x_e|.

    Args:
        config: run config.

    Returns:
        config.model.interface_features, or False when the key is absent (configs and
        run_info.json files written before the option existed).
    """
    return bool(config.get("model", {}).get("interface_features", False))


class PorousNaturalConvection2D(ForwardBVP):
    """Steady natural convection in a fluid/porous cavity (Gao et al. 2017, Sec. 4.3).

    The network maps (x, y) to (u, v, p, T) in the scaling of porous_pinn.numerical_pde.
    Losses:
        u_bc, v_bc       no slip on all four walls;
        T_dirichlet_bc   T = 1 on the hot wall x = 0 and T = 0 on the cold wall x = L;
        T_neumann_bc     dT/dy = 0 on the adiabatic walls y = 0 and y = H;
        u, v, p, T       PDE residuals (r_net). When config.pseudo_time.enabled, they are
                         relaxed by pseudo-time stepping (Wang et al. 2026, Secs. 2.3-2.4;
                         references/when pinn wrong.pdf), using sol_net as the solution value.

    With config.model.interface_features the input becomes (x, y, |x - x_s|, |x - x_e|),
    x_s and x_e being the interface positions. The extra features are continuous but have a
    slope jump at the interfaces. The network can then represent the kink of T across a
    high-conductivity layer without smoothing it into oscillations. This follows the
    cusp-capturing PINN of Tseng, Lin, Hu & Lai (2023), J. Comput. Phys. 491, 112359;
    that paper is not in references/, so verify the citation before using it in the paper.
    """
    variables = ("u", "v", "p", "T")

    @staticmethod
    def network_input_dim(config):
        """Width of the network input.

        Args:
            config: run config.

        Returns:
            4 when interface features are on, otherwise config.input_dim (2). Used by
            cases.experiment.build_model to initialise the first layer.
        """
        return 4 if uses_interface_features(config) else config.input_dim

    def __init__(self, config, lr, tx, arch, state):
        """Build the model and precompute the fixed boundary points.

        Args:
            config: run config; config.input_dim stays 2 (the number of coordinates) even when
                interface features widen the network input.
            lr, tx, arch, state: learning-rate schedule, optimizer, Flax module and train state
                from jaxpi.models (see cases.experiment.build_model).
        """
        # Đặc trưng mặt phân cách kiểu cusp-capturing (Tseng et al. 2023): |x - x_i| liên tục
        # nhưng có đạo hàm nhảy tại x_i, nên mạng biểu diễn được chỗ gãy gradient của T, v.
        self.interface_features = uses_interface_features(config)
        self.x_s = float(config.physics.get("porous_x_start", 0.0))
        self.x_e = float(config.physics.get("porous_x_end", 1.0))
        super().__init__(config, lr, tx, arch, state)

        # Khởi tạo sẵn điểm biên tĩnh trong bộ nhớ GPU
        num_bc_pts = config.training.get("num_bc_pts_per_side", 128)
        bc_data = sample_boundaries_natural_convection(num_bc_pts)

        self.wall_xy = bc_data["wall_xy"]
        self.left_xy = bc_data["left_xy"]
        self.left_T = bc_data["left_T"]
        self.right_xy = bc_data["right_xy"]
        self.right_T = bc_data["right_T"]
        self.adiabatic_xy = bc_data["adiabatic_xy"]

        # Vectorized helpers
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        self.T_pred_fn = vmap(self.T_net, (None, 0, 0))
        self.Ty_pred_fn = vmap(self.dT_dy_scalar, (None, 0, 0))

    def neural_net(self, params, x, y):
        """Evaluate the network at one point.

        Args:
            params: network parameters.
            x, y: scalar coordinates.

        Returns:
            Tuple (u, v, p, T) of scalars.
        """
        if self.interface_features:
            z = jnp.stack([x, y, jnp.abs(x - self.x_s), jnp.abs(x - self.x_e)])
        else:
            z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        return outputs[0], outputs[1], outputs[2], outputs[3]

    def u_net(self, params, x, y):
        """Horizontal velocity u at (x, y)."""
        return self.neural_net(params, x, y)[0]

    def v_net(self, params, x, y):
        """Vertical velocity v at (x, y)."""
        return self.neural_net(params, x, y)[1]

    def p_net(self, params, x, y):
        """Pressure variable p at (x, y)."""
        return self.neural_net(params, x, y)[2]

    def T_net(self, params, x, y):
        """Dimensionless temperature T at (x, y)."""
        return self.neural_net(params, x, y)[3]

    def dT_dy_scalar(self, params, x, y):
        """dT/dy at (x, y) by reverse-mode AD; used by the adiabatic-wall loss."""
        T_fn = lambda xs, ys: self.neural_net(params, xs, ys)[3]
        return jax.grad(T_fn, argnums=1)(x, y)

    def r_net(self, params, x, y):
        """PDE residuals at one collocation point.

        diff.scheme == "ad": pointwise residuals of the equations in porous_pinn.numerical_pde,
        with forward-mode AD derivatives and the coefficients of compute_porous_and_buoyancy.
        The discontinuous coefficients are evaluated pointwise, so no interface condition is
        imposed (cf. Hirata et al. 2009) and the heat flux need not be continuous across the
        layer. A trained "ad" run gives a mid-plane Nusselt number 4.7 times the wall value
        (ROADMAP.md).
        Any other scheme: the finite-volume residual of the control volume centred at (x, y),
        with width diff.h_stencil (set by porous_pinn.batching.create_batches).

        Args:
            params: network parameters.
            x, y: scalar coordinates.

        Returns:
            Dict {"u": x-momentum, "v": y-momentum, "p": continuity, "T": energy}.

        Raises:
            ValueError: if physics.momentum_form is not "gao" and the scheme is not
                "ffv_compact" (the only scheme that implements the "hirata" form).
        """
        scheme = self.config.diff.get("scheme", "ad").lower().strip()
        momentum_form = str(self.config.physics.get("momentum_form", "gao")).lower()
        if momentum_form != "gao" and scheme != "ffv_compact":
            raise ValueError(
                f"physics.momentum_form='{momentum_form}' chỉ được cài cho diff.scheme='ffv_compact'."
            )
        if scheme == "ad":
            (u, v, p, T), (d_x, d_y) = value_and_jacfwd(self.neural_net, (1, 2))(
                params, x, y
            )
            u_x, v_x, p_x, T_x = d_x
            u_y, v_y, p_y, T_y = d_y

            (u_xx, v_xx, _, T_xx), (u_yy, v_yy, _, T_yy) = hessian_diag_fwd(
                self.neural_net, (1, 2)
            )(params, x, y)
            laplace_u = u_xx + u_yy
            laplace_v = v_xx + v_yy
            laplace_T = T_xx + T_yy

            inv_eps, visc_coeff, diff_T_local, f_u_drag, f_v_drag, f_buoyancy = (
                compute_porous_and_buoyancy(u, v, T, x, self.config.physics)
            )

            ru = (
                inv_eps * (u * u_x + v * u_y)
                + p_x
                - visc_coeff * laplace_u
                + f_u_drag
            )
            rv = (
                inv_eps * (u * v_x + v * v_y)
                + p_y
                - visc_coeff * laplace_v
                + f_v_drag
                - f_buoyancy
            )
            rc = u_x + v_y
            rT = (u * T_x + v * T_y) - diff_T_local * laplace_T

            return {"u": ru, "v": rv, "p": rc, "T": rT}
        else:
            h_step = self.config.diff.get("h_stencil", 0.005)
            return compute_pde_residuals_nd(
                self.neural_net,
                params,
                x,
                y,
                h=h_step,
                physics_cfg=self.config.physics,
                scheme=scheme,
            )

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        """Unweighted boundary and residual losses for one batch.

        Args:
            params: network parameters.
            state: train state (pseudo-time stepping uses the parameters of the previous step).
            batch: (N, 2) array of collocation points from porous_pinn.batching.create_batches.

        Returns:
            Dict of scalar losses: u_bc, v_bc, T_dirichlet_bc, T_neumann_bc and the residual
            losses of ForwardBVP.compute_residual_losses.
        """
        u_wall = self.u_pred_fn(params, self.wall_xy[:, 0], self.wall_xy[:, 1])
        v_wall = self.v_pred_fn(params, self.wall_xy[:, 0], self.wall_xy[:, 1])
        u_bc_loss = jnp.mean(u_wall**2)
        v_bc_loss = jnp.mean(v_wall**2)

        T_left = self.T_pred_fn(params, self.left_xy[:, 0], self.left_xy[:, 1])
        T_right = self.T_pred_fn(params, self.right_xy[:, 0], self.right_xy[:, 1])
        T_dirichlet_loss = jnp.mean((T_left - 1.0) ** 2) + jnp.mean(T_right**2)

        Ty_adiabatic = self.Ty_pred_fn(
            params, self.adiabatic_xy[:, 0], self.adiabatic_xy[:, 1]
        )
        T_neumann_loss = jnp.mean(Ty_adiabatic**2)

        res_losses = self.compute_residual_losses(
            params, state, batch, pseudo_time=self.config.pseudo_time.enabled
        )

        return {
            "u_bc": u_bc_loss,
            "v_bc": v_bc_loss,
            "T_dirichlet_bc": T_dirichlet_loss,
            "T_neumann_bc": T_neumann_loss,
            **res_losses,
        }

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, x_star, y_star, u_ref, v_ref, T_ref):
        """Relative L2 errors of u, v and T against reference fields.

        Args:
            params: network parameters.
            x_star, y_star: 1-D grid coordinates.
            u_ref, v_ref, T_ref: reference fields indexed [ix, iy].

        Returns:
            Dict {"u_l2_error", "v_l2_error", "T_l2_error"}.
        """
        grid_eval = lambda fn: vmap(vmap(fn, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        u_pred = grid_eval(self.u_net)
        v_pred = grid_eval(self.v_net)
        T_pred = grid_eval(self.T_net)

        u_err = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_err = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        T_err = jnp.linalg.norm(T_pred - T_ref) / jnp.linalg.norm(T_ref)

        return {"u_l2_error": u_err, "v_l2_error": v_err, "T_l2_error": T_err}

    @partial(jit, static_argnums=(0,))
    def sol_net(self, params, x, y):
        """Control-volume average of (u, v, p, T), consistent with the residual stencil.

        Pseudo-time stepping pairs each residual with this value by variable name
        (jaxpi.models.ForwardBVP.sol_net), so for FV schemes it must represent the same
        control volume as the residual:
            "ad": the point value at (x, y);
            "ffv", "ffv_continuous", "ffv_compact": weights [1/2, 1/8, 1/8, 1/8, 1/8] on the
                centre and the four faces at distance h/2;
            "can_ffv": weights [3/4, 1/16, 1/16, 1/16, 1/16] on the centre and the four
                neighbouring centres at distance h.

        Args:
            params: network parameters.
            x, y: scalar coordinates.

        Returns:
            Dict {"u", "v", "p", "T"}.
        """
        scheme = self.config.diff.get("scheme", "ad").lower().strip()
        if scheme == "ad":
            u, v, p, T = self.neural_net(params, x, y)
            return {"u": u, "v": v, "p": p, "T": T}

        h = self.config.diff.get("h_stencil", 0.01)

        if scheme in ["can_ffv", "can", "nd"]:
        
            # Stencil của can_ffv nằm tại 5 tâm ô [P, E, W, N, S] (khoảng cách h)
            x_pts = jnp.stack([x, x + h, x - h, x, x])
            y_pts = jnp.stack([y, y, y, y + h, y - h])
            w_vol = jnp.array([0.75, 0.0625, 0.0625, 0.0625, 0.0625])
        else:
            # Stencil của ffv / ffv_continuous dùng 4 mặt [P, e, w, n, s] (khoảng cách h/2)
            half_h = 0.5 * h
            x_pts = jnp.stack([x, x + half_h, x - half_h, x, x])
            y_pts = jnp.stack([y, y, y, y + half_h, y - half_h])
            w_vol = jnp.array([0.5, 0.125, 0.125, 0.125, 0.125])

        u_s, v_s, p_s, T_s = vmap(self.neural_net, (None, 0, 0))(params, x_pts, y_pts)
        return {
            "u": jnp.dot(w_vol, u_s),
            "v": jnp.dot(w_vol, v_s),
            "p": jnp.dot(w_vol, p_s),
            "T": jnp.dot(w_vol, T_s),
        }