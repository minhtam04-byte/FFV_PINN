"""Verification of the finite-volume residual schemes, in float64.

Each scheme is fed analytic fields instead of a network, so its residual can be
compared with the exact pointwise operator (the `ad` branch of r_net):
  1. consistency and order: the error vs the exact operator falls as O(h^2)
     inside the fluid and inside the porous region;
  2. interface conservation: for a piecewise-linear temperature with continuous
     heat flux k dT/dx across the fluid/porous interface, the FV energy
     residual vanishes to round-off (harmonic face conductivity);
  3. a temperature whose flux jumps at the interface gives a zero pointwise
     residual (k * T'' = 0 on both sides) but a non-zero FV residual: only the
     FV form sees the interface condition.

Run:  pytest tests/test_fv_schemes.py -q
"""
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import pytest

from cases.conduction_1d.numerical_pde import (compute_can_pde_residuals_1d,
                                               compute_ffv_pde_residuals_1d)
from porous_pinn.numerical_pde import (compute_pde_residuals_nd,
                                       compute_porous_and_buoyancy)

PHYS = ml_collections.ConfigDict({
    "Pr": 1.0, "Ra": 1.0e3, "Da": 1.0e-2, "epsilon": 0.4, "F_eps": 0.5,
    "T_ref": 0.5, "lambda_k": 3.0, "porous_x_start": 0.4, "porous_x_end": 0.9,
})
FV_SCHEMES = ["ffv", "ffv_continuous", "ffv_compact", "can_ffv"]
RNG = np.random.default_rng(0)
Y_PTS = RNG.uniform(0.2, 0.8, 16)
# stencils (reach <= h <= 0.02) stay inside one region
X_PTS = {
    "fluid": RNG.uniform(0.1, 0.3, 16),
    "porous": RNG.uniform(0.5, 0.8, 16),
}


@pytest.fixture(autouse=True)
def float64():
  """float64 for these tests only; a global flag would leak into the other test files."""
  with jax.enable_x64(True):
    yield


# ---------------------------------------------------------------------------
# Manufactured smooth fields: divergence-free velocity from a stream function
# ---------------------------------------------------------------------------
def stream(x, y):
  return 0.3 * jnp.sin(2.0 * x + 0.3) * jnp.cos(1.5 * y)


def smooth_fields(params, x, y):
  del params
  u = jax.grad(stream, 1)(x, y)
  v = -jax.grad(stream, 0)(x, y)
  p = 0.5 * jnp.cos(x) * jnp.sin(y)
  T = 0.5 + 0.3 * jnp.sin(1.3 * x) * jnp.cos(0.7 * y)
  return u, v, p, T


def exact_residuals(fields, x, y, phys):
  """Pointwise strong form, identical to the `ad` branch of PorousNaturalConvection2D.r_net."""
  f = lambda xx, yy: jnp.stack(fields(None, xx, yy))
  q = f(x, y)
  q_x, q_y = jax.jacfwd(f, 0)(x, y), jax.jacfwd(f, 1)(x, y)
  lap = jax.jacfwd(jax.jacfwd(f, 0), 0)(x, y) + jax.jacfwd(jax.jacfwd(f, 1), 1)(x, y)
  u, v, _, T = q
  inv_eps, visc, diff_T, f_u, f_v, f_b = compute_porous_and_buoyancy(u, v, T, x, phys)
  return {
      "u": inv_eps * (u * q_x[0] + v * q_y[0]) + q_x[2] - visc * lap[0] + f_u,
      "v": inv_eps * (u * q_x[1] + v * q_y[1]) + q_y[2] - visc * lap[1] + f_v - f_b,
      "p": q_x[0] + q_y[1],
      "T": u * q_x[3] + v * q_y[3] - diff_T * lap[3],
  }


def fv_residuals(fields, x, y, h, scheme, phys=PHYS):
  return jax.vmap(
      lambda xx, yy: compute_pde_residuals_nd(fields, None, xx, yy, h, phys, scheme)
  )(x, y)


class TestConsistencyOrder:
  @pytest.mark.parametrize("region", ["fluid", "porous"])
  @pytest.mark.parametrize("scheme", FV_SCHEMES)
  def test_second_order(self, scheme, region):
    x, y = jnp.asarray(X_PTS[region]), jnp.asarray(Y_PTS)
    exact = jax.vmap(lambda xx, yy: exact_residuals(smooth_fields, xx, yy, PHYS))(x, y)
    errors = []
    for h in (0.02, 0.01):
      res = fv_residuals(smooth_fields, x, y, h, scheme)
      errors.append({k: float(jnp.max(jnp.abs(res[k] - exact[k]))) for k in exact})
    for k in exact:
      order = np.log2(errors[0][k] / errors[1][k])
      assert order > 1.8, f"{scheme}/{region}/{k}: observed order {order:.2f}, errors {errors[0][k]:.2e} -> {errors[1][k]:.2e}"


# ---------------------------------------------------------------------------
# Piecewise-linear temperature across the fluid/porous interface, u = v = p = 0
# ---------------------------------------------------------------------------
X_I = PHYS.porous_x_start  # fluid (k_f) for x < X_I, porous (lambda_k * k_f) beyond
SLOPE_FLUID = -1.0


def piecewise_T(slope_porous):
  def fields(params, x, y):
    del params
    T = jnp.where(x < X_I, 1.0 + SLOPE_FLUID * (x - X_I), 1.0 + slope_porous * (x - X_I))
    zero = 0.0 * x
    return zero, zero, zero, T + 0.0 * y
  return fields


FLUX_CONTINUOUS = piecewise_T(SLOPE_FLUID / PHYS.lambda_k)  # k_f T'_f = k_p T'_p
FLUX_JUMP = piecewise_T(SLOPE_FLUID)                        # same slope: flux jumps by lambda_k
H = 0.02
FLUX_SCALE = abs(SLOPE_FLUID) / np.sqrt(PHYS.Pr * PHYS.Ra) / H  # size of one face-flux term


class TestInterfaceConservation:
  @pytest.mark.parametrize("scheme,reach", [("ffv_continuous", 1.0), ("ffv_compact", 0.5)])
  def test_exact_at_any_interface_offset(self, scheme, reach):
    # stencil centres whose [x - reach*h, x + reach*h] straddles the interface at arbitrary offsets
    x = jnp.asarray(RNG.uniform(X_I - 0.95 * reach * H, X_I + 0.95 * reach * H, 32))
    y = jnp.full_like(x, 0.5)
    r_T = fv_residuals(FLUX_CONTINUOUS, x, y, H, scheme)["T"]
    assert float(jnp.max(jnp.abs(r_T))) < 1e-10 * FLUX_SCALE

  def test_ffv_exact_when_interface_on_a_face(self):
    # ffv uses the harmonic mean of the two centre values, exact only when the
    # interface sits midway between P and a neighbour (i.e. on a cell face)
    x = jnp.asarray([X_I - 0.5 * H, X_I + 0.5 * H])
    y = jnp.full_like(x, 0.5)
    r_T = fv_residuals(FLUX_CONTINUOUS, x, y, H, "ffv")["T"]
    assert float(jnp.max(jnp.abs(r_T))) < 1e-10 * FLUX_SCALE

  def test_flux_jump_seen_by_fv_but_not_by_pointwise_form(self):
    x = jnp.asarray([X_I - 0.3 * H, X_I + 0.3 * H])
    y = jnp.full_like(x, 0.5)
    r_fv = fv_residuals(FLUX_JUMP, x, y, H, "ffv_continuous")["T"]
    r_pointwise = jax.vmap(lambda xx, yy: exact_residuals(FLUX_JUMP, xx, yy, PHYS)["T"])(x, y)
    assert float(jnp.max(jnp.abs(r_pointwise))) < 1e-12
    assert float(jnp.min(jnp.abs(r_fv))) > 0.1 * FLUX_SCALE


class TestHirataMomentumForm:
  """physics.momentum_form = "hirata": the same bulk physics divided by eps, with the
  effective viscosity nu/eps inside the divergence (Hirata et al. 2009)."""

  PHYS_H = ml_collections.ConfigDict({**PHYS.to_dict(), "momentum_form": "hirata"})

  @staticmethod
  def exact_hirata(fields, x, y, phys):
    """Pointwise Hirata form = Gao's pointwise form divided by eps, with grad(P) not grad(p)/eps."""
    f = lambda xx, yy: jnp.stack(fields(None, xx, yy))
    q = f(x, y)
    q_x, q_y = jax.jacfwd(f, 0)(x, y), jax.jacfwd(f, 1)(x, y)
    lap = jax.jacfwd(jax.jacfwd(f, 0), 0)(x, y) + jax.jacfwd(jax.jacfwd(f, 1), 1)(x, y)
    u, v, _, T = q
    inv_eps, visc, diff_T, f_u, f_v, f_b = compute_porous_and_buoyancy(u, v, T, x, phys)
    return {
        "u": inv_eps**2 * (u * q_x[0] + v * q_y[0]) + q_x[2] - inv_eps * visc * lap[0] + inv_eps * f_u,
        "v": inv_eps**2 * (u * q_x[1] + v * q_y[1]) + q_y[2] - inv_eps * visc * lap[1] + inv_eps * (f_v - f_b),
        "p": q_x[0] + q_y[1],
        "T": u * q_x[3] + v * q_y[3] - diff_T * lap[3],
    }

  @pytest.mark.parametrize("region", ["fluid", "porous"])
  def test_second_order(self, region):
    x, y = jnp.asarray(X_PTS[region]), jnp.asarray(Y_PTS)
    exact = jax.vmap(lambda xx, yy: self.exact_hirata(smooth_fields, xx, yy, self.PHYS_H))(x, y)
    errors = []
    for h in (0.02, 0.01):
      res = fv_residuals(smooth_fields, x, y, h, "ffv_compact", phys=self.PHYS_H)
      errors.append({k: float(jnp.max(jnp.abs(res[k] - exact[k]))) for k in exact})
    for k in exact:
      order = np.log2(errors[0][k] / errors[1][k])
      assert order > 1.8, f"hirata/{region}/{k}: observed order {order:.2f}"

  def test_shear_stress_jump_at_interface(self):
    # Tangential velocity v(x), piecewise linear across the vertical interface, no flow
    # through it (u = 0), no drag/buoyancy. Shear-stress continuity with nu_eff = nu/eps
    # (Hirata Eq. 15) requires v'_porous = eps * v'_fluid.
    phys = ml_collections.ConfigDict({**PHYS.to_dict(), "Da": 1e14, "F_eps": 0.0})
    slope_f = 1.0

    def shear_fields(slope_m):
      def fields(params, x, y):
        del params
        v = jnp.where(x < X_I, slope_f * (x - X_I), slope_m * (x - X_I))
        zero = 0.0 * x
        return zero, v + 0.0 * y, zero, phys.T_ref + zero
      return fields

    x = jnp.asarray(RNG.uniform(X_I - 0.45 * H, X_I + 0.45 * H, 32))
    y = jnp.full_like(x, 0.5)
    scale = np.sqrt(phys.Pr / phys.Ra) / H  # size of one viscous face flux
    hirata = ml_collections.ConfigDict({**phys.to_dict(), "momentum_form": "hirata"})
    stress_jump = shear_fields(phys.epsilon * slope_f)    # Hirata interface condition
    gradient_cont = shear_fields(slope_f)                  # Gao-form interface condition
    r = lambda fields, cfg: fv_residuals(fields, x, y, H, "ffv_compact", phys=cfg)["v"]
    assert float(jnp.max(jnp.abs(r(stress_jump, hirata)))) < 1e-10 * scale
    assert float(jnp.max(jnp.abs(r(gradient_cont, phys)))) < 1e-10 * scale
    # each form rejects the other's interface condition
    assert float(jnp.min(jnp.abs(r(gradient_cont, hirata)))) > 0.01 * scale
    assert float(jnp.min(jnp.abs(r(stress_jump, phys)))) > 0.01 * scale


class TestDragCellAverage:
  """Darcy-Forchheimer drag averaged over a control volume cut by the interface.

  The drag the scheme applies is isolated as rv(Da) - rv(Da -> inf, F = 0) for a
  tangential velocity v(x) (u = p = 0, T = T_ref): every other term is unchanged.
  The exact value is the cell average of X_p * c * (nu/Da + F/sqrt(Da)|v|) v over
  the porous part of [x - h/2, x + h/2] (c = eps for the Gao form, 1 for Hirata).
  """

  PHYS_D = ml_collections.ConfigDict({**PHYS.to_dict(), "Da": 1e-3, "F_eps": 0.0})

  @staticmethod
  def velocity_fields(v_of_x, phys):
    def fields(params, x, y):
      del params
      zero = 0.0 * x
      return zero, v_of_x(x) + 0.0 * y, zero, phys.T_ref + zero
    return fields

  @staticmethod
  def scheme_drag(v_of_x, x, h, scheme, phys):
    fields = TestDragCellAverage.velocity_fields(v_of_x, phys)
    no_drag = ml_collections.ConfigDict({**phys.to_dict(), "Da": 1e14, "F_eps": 0.0})
    y = jnp.full_like(x, 0.5)
    return (fv_residuals(fields, x, y, h, scheme, phys=phys)["v"]
            - fv_residuals(fields, x, y, h, scheme, phys=no_drag)["v"])

  @staticmethod
  def exact_drag(v_of_x, x, h, phys, n_gauss=24):
    """(1/h) * integral over the porous part of the cell (Gauss-Legendre, exact for these profiles)."""
    c = phys.epsilon if phys.get("momentum_form", "gao") == "gao" else 1.0
    nu = np.sqrt(phys.Pr / phys.Ra)
    g, w = np.polynomial.legendre.leggauss(n_gauss)
    out = []
    for xc in np.asarray(x):
      a, b = max(xc - h / 2, phys.porous_x_start), min(xc + h / 2, phys.porous_x_end)
      if b <= a:
        out.append(0.0)
        continue
      s = 0.5 * (b - a) * g + 0.5 * (a + b)
      v = np.asarray(v_of_x(jnp.asarray(s)))
      speed = np.sqrt(v**2 + 1e-6)  # same regularisation as compute_pure_porous_drag
      dens = c * (nu / phys.Da + phys.F_eps / np.sqrt(phys.Da) * speed) * v
      out.append(0.5 * (b - a) * np.dot(w, dens) / h)
    return np.array(out)

  @staticmethod
  def linear_v(x):
    return 1.0 + 20.0 * (x - X_I)

  @pytest.mark.parametrize("form", ["gao", "hirata"])
  @pytest.mark.parametrize("scheme", ["ffv_continuous", "ffv_compact"])
  def test_exact_inside_porous_layer(self, scheme, form):
    phys = ml_collections.ConfigDict({**self.PHYS_D.to_dict(), "momentum_form": form})
    if form == "hirata" and scheme != "ffv_compact":
      pytest.skip("Hirata form is implemented for ffv_compact only")
    h = 0.02
    x = jnp.asarray(X_I + h * RNG.uniform(1.0, 3.0, 16))  # cells fully inside the porous layer
    got = np.asarray(self.scheme_drag(self.linear_v, x, h, scheme, phys))
    ref = self.exact_drag(self.linear_v, x, h, phys)
    assert np.max(np.abs(got - ref)) < 1e-9 * np.max(np.abs(ref))

  @pytest.mark.xfail(strict=True, reason=(
      "Known limitation: in a cell cut by the interface the drag is (porous fraction) x "
      "(mean of the end-point drags), and one end point lies in the fluid, so the cell "
      "average is only first order. Fix: Gauss quadrature on the porous sub-segment."))
  @pytest.mark.parametrize("scheme", ["ffv_continuous", "ffv_compact"])
  def test_second_order_in_cut_cells(self, scheme):
    phys = self.PHYS_D
    offsets = np.linspace(-0.45, 0.45, 19)  # interface position inside the cell, in units of h
    errors = []
    for h in (0.02, 0.01, 0.005):
      x = jnp.asarray(X_I - offsets * h)
      got = np.asarray(self.scheme_drag(self.linear_v, x, h, scheme, phys))
      errors.append(np.max(np.abs(got - self.exact_drag(self.linear_v, x, h, phys))))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.8), f"observed orders {orders}, errors {errors}"


class TestConduction1D:
  """1D vertex-centred schemes: the analytical two-layer solution is a discrete solution."""

  L, D, K1, K2 = 1.0, 0.5, 5.0, 1.0

  def analytic(self, x):
    flux = 1.0 / (self.D / self.K1 + (self.L - self.D) / self.K2)  # Th = 1, Tc = 0
    return jnp.where(x <= self.D, 1.0 - flux * x / self.K1, flux * (self.L - x) / self.K2)

  @pytest.mark.parametrize("scheme_fn", [compute_ffv_pde_residuals_1d, compute_can_pde_residuals_1d])
  def test_analytic_solution_has_zero_residual(self, scheme_fn):
    n_x = 101
    dx = self.L / (n_x - 1)
    x = jnp.linspace(dx, self.L - dx, n_x - 2)  # nodes, one of them on the interface
    T_fn = lambda params, xx: self.analytic(xx)
    r = jax.vmap(lambda xx: scheme_fn(T_fn, None, xx, dx, self.D, self.K1, self.K2)["T"])(x)
    assert float(jnp.max(jnp.abs(r))) < 1e-9
