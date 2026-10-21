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
    """config.model.interface_features (mặc định False; config cũ không có khóa này)."""
    return bool(config.get("model", {}).get("interface_features", False))


class PorousNaturalConvection2D(ForwardBVP):
    variables = ("u", "v", "p", "T")

    @staticmethod
    def network_input_dim(config):
        """Số đầu vào của mạng: (x, y), thêm (|x - x_s|, |x - x_e|) khi bật đặc trưng mặt phân cách."""
        return 4 if uses_interface_features(config) else config.input_dim

    def __init__(self, config, lr, tx, arch, state):
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
        if self.interface_features:
            z = jnp.stack([x, y, jnp.abs(x - self.x_s), jnp.abs(x - self.x_e)])
        else:
            z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        return outputs[0], outputs[1], outputs[2], outputs[3]

    def u_net(self, params, x, y):
        return self.neural_net(params, x, y)[0]

    def v_net(self, params, x, y):
        return self.neural_net(params, x, y)[1]

    def p_net(self, params, x, y):
        return self.neural_net(params, x, y)[2]

    def T_net(self, params, x, y):
        return self.neural_net(params, x, y)[3]

    def dT_dy_scalar(self, params, x, y):
        T_fn = lambda xs, ys: self.neural_net(params, xs, ys)[3]
        return jax.grad(T_fn, argnums=1)(x, y)

    def r_net(self, params, x, y):
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
        """Nghiệm trung bình thể tích ô kiểm soát (Cell-Volume Average) đồng bộ theo từng Stencil:
        - 'ad': Đánh giá điểm tại tâm P(x, y).
        - 'ffv' / 'ffv_continuous': Stencil mặt kiểm soát cách half_h = h/2 -> trọng số [0.5, 0.125 x 4].
        - 'can_ffv' / 'can': Stencil tâm ô lân cận cách h -> trọng số [0.75, 0.0625 x 4] tại [P, E, W, N, S].
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