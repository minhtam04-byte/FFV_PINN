from functools import partial
import jax
from jax import jit, vmap
import jax.numpy as jnp

from jaxpi.models import ForwardBVP
from cases.conduction_1d.numerical_pde import (
    compute_can_pde_residuals_1d,
    compute_ffv_pde_residuals_1d,
    get_k_local,
)


class ConjugateHeatConduction1D(ForwardBVP):
    variables = ("T",)

    def __init__(self, config, lr, tx, arch, state):
        super().__init__(config, lr, tx, arch, state)

        self.L = config.physics.get("L", 1.0)
        self.D = config.physics.get("D", 0.5)
        self.k1 = config.physics.get("k1", 5.0)
        self.k2 = config.physics.get("k2", 1.0)
        self.Th = config.physics.get("Th", 1.0)
        self.Tc = config.physics.get("Tc", 0.0)

        self.bc_left_x = jnp.array([0.0])
        self.bc_right_x = jnp.array([self.L])
        self.T_pred_fn = vmap(self.T_net, (None, 0))

    def neural_net(self, params, x):
        z = jnp.reshape(x, (1,))
        outputs = self.state.apply_fn(params, z)
        return (outputs[0],)

    def T_net(self, params, x):
        return self.neural_net(params, x)[0]

    def r_net(self, params, x):
        scheme = self.config.diff.get("scheme", "ad").lower().strip()

        if scheme == "ad":
            def flux_fn(x_in):
                _, dT = jax.jvp(
                    lambda xi: self.T_net(params, xi),
                    (x_in,),
                    (jnp.ones_like(x_in),),
                )
                k = get_k_local(x_in, self.D, self.k1, self.k2)
                return k * dT

            _, div_flux = jax.jvp(flux_fn, (x,), (jnp.ones_like(x),))
            return {"T": -div_flux}

        elif scheme == "can":
            h_step = self.config.diff.get("h_stencil", 0.01)
            return compute_can_pde_residuals_1d(
                self.T_net, params, x, h_step, self.D, self.k1, self.k2
            )

        elif scheme in ["ffv", "nd"]:
            h_step = self.config.diff.get("h_stencil", 0.01)
            return compute_ffv_pde_residuals_1d(
                self.T_net, params, x, h_step, self.D, self.k1, self.k2
            )

        else:
            raise ValueError(f"Sơ đồ vi phân không hợp lệ: '{scheme}'")

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        T_left = self.T_pred_fn(params, self.bc_left_x)
        T_right = self.T_pred_fn(params, self.bc_right_x)

        T_bc_loss = jnp.mean((T_left - self.Th) ** 2) + jnp.mean(
            (T_right - self.Tc) ** 2
        )

        res_losses = self.compute_residual_losses(
            params, state, batch, pseudo_time=self.config.pseudo_time.enabled
        )

        return {
            "T_bc": T_bc_loss,
            **res_losses,
        }

    @partial(jit, static_argnums=(0,))
    def compute_analytical_solution(self, x):
        factor = (self.Th - self.Tc) / (
            self.D / self.k1 + (self.L - self.D) / self.k2
        )
        T_solid1 = self.Th - factor * (x / self.k1)
        T_solid2 = self.Tc + factor * ((self.L - x) / self.k2)
        return jnp.where(x <= self.D, T_solid1, T_solid2)

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, x_test):
        T_pred = self.T_pred_fn(params, x_test)
        T_exact = self.compute_analytical_solution(x_test)
        l2_err = jnp.linalg.norm(T_pred - T_exact) / (
            jnp.linalg.norm(T_exact) + 1e-8
        )
        return {"T_l2_error": l2_err}