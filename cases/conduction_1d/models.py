"""PINN model for steady 1-D conduction through two layers (Gao et al. 2017, Sec. 4.1).

Two materials with conductivities k1 (0 <= x <= D) and k2 (D <= x <= L) carry heat
between T(0) = Th and T(L) = Tc. The exact solution is piecewise linear with continuous
flux at x = D (Gao et al. 2017, Eq. (44); references/lbm_ref.pdf). This is the simplest
test of whether a residual imposes the interface flux condition.
"""
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
    """1-D conjugate conduction, -d/dx(k(x) dT/dx) = 0 with k piecewise constant.

    Losses: T_bc (Dirichlet conditions at both ends) and the residual loss T. Residual
    schemes (config.diff.scheme):
        "ad": d/dx(k dT/dx) by nested forward-mode AD. The pointwise derivative of the
            jump in k is zero, so the flux condition at D is never imposed. A trained run
            reaches a loss of ~1e-6 with a relative L2 error of 6e-2 (notes/2026-10-21_session.md).
        "ffv" (alias "nd"), "can": finite-volume flux balances (numerical_pde in this
            package), which impose flux continuity.
    """
    variables = ("T",)

    def __init__(self, config, lr, tx, arch, state):
        """Build the model and the two boundary points x = 0 and x = L.

        Args:
            config: run config; physics holds L, D, k1, k2, Th and Tc.
            lr, tx, arch, state: from jaxpi.models (see cases.experiment.build_model).
        """
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
        """Network output at scalar x, as a 1-tuple (T,)."""
        z = jnp.reshape(x, (1,))
        outputs = self.state.apply_fn(params, z)
        return (outputs[0],)

    def T_net(self, params, x):
        """Temperature T at scalar x."""
        return self.neural_net(params, x)[0]

    def r_net(self, params, x):
        """Residual of the energy equation at one point x.

        Args:
            params: network parameters.
            x: scalar coordinate.

        Returns:
            Dict {"T": residual}.

        Raises:
            ValueError: for an unknown diff.scheme.
        """
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
        """Unweighted losses for one batch: T_bc plus the residual loss from
        ForwardBVP.compute_residual_losses.
        """
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
        """Exact temperature (Gao et al. 2017, Eq. (44)).

        A constant heat flux q = (Th - Tc) / (D/k1 + (L - D)/k2) crosses both layers, so T is
        linear on each side of D, with slopes -q/k1 and -q/k2.
        """
        factor = (self.Th - self.Tc) / (
            self.D / self.k1 + (self.L - self.D) / self.k2
        )
        T_solid1 = self.Th - factor * (x / self.k1)
        T_solid2 = self.Tc + factor * ((self.L - x) / self.k2)
        return jnp.where(x <= self.D, T_solid1, T_solid2)

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, x_test):
        """Relative L2 error of T against compute_analytical_solution at x_test."""
        T_pred = self.T_pred_fn(params, x_test)
        T_exact = self.compute_analytical_solution(x_test)
        l2_err = jnp.linalg.norm(T_pred - T_exact) / (
            jnp.linalg.norm(T_exact) + 1e-8
        )
        return {"T_l2_error": l2_err}