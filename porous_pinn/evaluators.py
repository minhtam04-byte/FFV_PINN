"""Evaluators: metrics logged by jaxpi.training.train_loop during training.

Keys follow jaxpi's "section/name" convention. The console table prints only
"loss/" and "error/" keys; everything is kept in `evaluator.history`, which
cases.experiment writes to history.pkl next to the checkpoints.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from jaxpi.evaluator import BaseEvaluator
from porous_pinn.numerical_pde import is_in_porous_zone


class HistoryEvaluator(BaseEvaluator):
  """BaseEvaluator that also keeps the full training history.

  Adds loss/total (the weighted sum of losses) and, when config.logging.log_errors is set,
  the problem-specific metrics() to every logged dict. Every logged value is appended to
  self.history[key], together with the training step in self.history["step"].
  """

  def __init__(self, config):
    """Start with an empty history.

    Args:
        config: run config (reads config.logging).
    """
    super().__init__(config)
    self.history = {}

  def metrics(self, model, params):
    """Problem-specific metrics; subclasses override this.

    Args:
        model: the jaxpi model.
        params: network parameters.

    Returns:
        Dict mapping "section/name" keys to scalars.
    """
    return {}

  def __call__(self, model, state, loss_dict, batch, *args):
    """Build the log dict for one logging step and record it in self.history.

    Args:
        model: the jaxpi model.
        state: train state after the step.
        loss_dict: unweighted losses of the step.
        batch: collocation batch of the step.
        *args: passed on to BaseEvaluator.

    Returns:
        The log dict passed to the logger.
    """
    log_dict = super().__call__(model, state, loss_dict, batch, *args)
    log_dict["loss/total"] = sum(state.loss_weights[k] * v for k, v in loss_dict.items())
    if self.config.logging.get("log_errors", False):
      log_dict.update(self.metrics(model, state.params))

    # train_loop logs at loop index `step`, after model.step has already incremented state.step
    self.history.setdefault("step", []).append(int(state.step) - 1)
    for key, value in log_dict.items():
      self.history.setdefault(key, []).append(float(np.asarray(value)))
    self.log_dict = log_dict
    return log_dict


@partial(jit, static_argnums=(0, 2))
def convection_fields(model, params, n):
  """Global diagnostics of a natural-convection solution on an n x n grid.

  Stream function, from the bottom wall upwards:
      psi(x, y) = sqrt(Ra Pr) int_0^y u dy'   (units alpha_f, as psi_max in Gao et al.
                                               2017, Sec. 4.3)
  Mass conservation requires psi(x, H) = 0, so max|psi(x, H)| / psi_max measures the
  mass imbalance.

  Total heat flux through the vertical section at x:
      Q(x) = (1/H) int_0^H (sqrt(Ra Pr) u T - (k/k_f) dT/dx) dy.
  Q is independent of x for an exact steady solution. At the walls (u = 0) it equals the
  average Nusselt number of Gao et al. (2017), Eq. (46). Nu_mid = Q(L/2) does not depend
  on the network's gradient at the wall, which FV residuals constrain only indirectly.

  Args:
      model: PorousNaturalConvection2D (static argument).
      params: network parameters.
      n: number of grid points per direction (static argument).

  Returns:
      Dict with psi_max, mass_imbalance, Nu_hot = Q(0), Nu_cold = Q(L) and
      Nu_mid = Q(L/2).
  """
  phys = model.config.physics
  scale = jnp.sqrt(phys.Ra * phys.Pr)
  x = jnp.linspace(0.0, phys.L, n)
  y = jnp.linspace(0.0, phys.H, n)
  X, Y = jnp.meshgrid(x, y)  # [iy, ix]
  u = vmap(lambda xx, yy: model.neural_net(params, xx, yy)[0])(X.ravel(), Y.ravel()).reshape(n, n)
  dy = y[1] - y[0]
  psi = jnp.concatenate([jnp.zeros((1, n)), jnp.cumsum(0.5 * (u[1:] + u[:-1]) * dy, axis=0)]) * scale
  psi_max = jnp.max(jnp.abs(psi))

  dT_dx = jax.grad(lambda xx, yy: model.neural_net(params, xx, yy)[3], argnums=0)

  def total_heat_flux(x_line):
    """Q(x_line): convective plus conductive heat flux through x = x_line, trapezoidal rule in y."""
    k_ratio = jnp.where(is_in_porous_zone(x_line, phys), phys.get("lambda_k", 1.0), 1.0)
    def u_times_T(yy):
      out = model.neural_net(params, x_line, yy)
      return out[0] * out[3]

    u_T = vmap(u_times_T)(y)
    q = scale * u_T - k_ratio * vmap(dT_dx, (None, 0))(x_line, y)
    return jnp.trapezoid(q, y) / phys.H

  return {
      "psi_max": psi_max,
      "mass_imbalance": jnp.max(jnp.abs(psi[-1])) / psi_max,
      "Nu_hot": total_heat_flux(0.0),
      "Nu_cold": total_heat_flux(phys.L),
      "Nu_mid": total_heat_flux(0.5 * phys.L),
  }


class ConvectionEvaluator(HistoryEvaluator):
  """Natural convection in a heated cavity (hot wall x = 0, cold wall x = L)."""

  def metrics(self, model, params):
    """Global diagnostics from convection_fields on a logging.eval_grid grid (default 101).

    Args:
        model: PorousNaturalConvection2D.
        params: network parameters.

    Returns:
        Dict with metric/psi_max, metric/Nu_hot, metric/Nu_cold, metric/Nu_mid and two
        conservation errors that vanish for an exact solution: error/mass_imbalance and
        error/Nu_imbalance = |Nu_hot - Nu_cold| / mean. When config.reference provides
        psi_max or Nu, it also returns the relative deviations error/psi_max_vs_ref and
        error/Nu_vs_ref.
    """
    f = convection_fields(model, params, self.config.logging.get("eval_grid", 101))
    nu_mean = 0.5 * (f["Nu_hot"] + f["Nu_cold"])
    out = {
        "metric/psi_max": f["psi_max"],
        "metric/Nu_hot": f["Nu_hot"],
        "metric/Nu_cold": f["Nu_cold"],
        "metric/Nu_mid": f["Nu_mid"],
        # Global conservation: both must be 0 for an exact solution
        "error/mass_imbalance": f["mass_imbalance"],
        "error/Nu_imbalance": jnp.abs(f["Nu_hot"] - f["Nu_cold"]) / nu_mean,
    }
    ref = self.config.get("reference", {})
    if ref.get("psi_max") is not None:
      out["error/psi_max_vs_ref"] = jnp.abs(f["psi_max"] - ref.psi_max) / ref.psi_max
    if ref.get("Nu") is not None:
      out["error/Nu_vs_ref"] = jnp.abs(nu_mean - ref.Nu) / ref.Nu
    return out
