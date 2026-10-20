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
  """BaseEvaluator that also records every logged dict, plus the total weighted loss."""

  def __init__(self, config):
    super().__init__(config)
    self.history = {}

  def metrics(self, model, params):
    """Problem-specific metrics; override in subclasses."""
    return {}

  def __call__(self, model, state, loss_dict, batch, *args):
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
  """Stream function, wall Nusselt numbers and mass imbalance on an n x n grid.

  psi(x, y) = sqrt(Ra Pr) * int_0^y u dy' (velocity in units of alpha_f / L);
  psi(x, 1) must vanish, so max|psi(x, 1)| / psi_max measures mass imbalance.
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

  def nusselt(x_wall):
    k_ratio = jnp.where(is_in_porous_zone(x_wall, phys), phys.get("lambda_k", 1.0), 1.0)
    q = -k_ratio * vmap(dT_dx, (None, 0))(x_wall, y)
    return jnp.trapezoid(q, y) / phys.H

  return {
      "psi_max": psi_max,
      "mass_imbalance": jnp.max(jnp.abs(psi[-1])) / psi_max,
      "Nu_hot": nusselt(0.0),
      "Nu_cold": nusselt(phys.L),
  }


class ConvectionEvaluator(HistoryEvaluator):
  """Natural convection in a heated cavity (hot wall x = 0, cold wall x = L)."""

  def metrics(self, model, params):
    f = convection_fields(model, params, self.config.logging.get("eval_grid", 101))
    nu_mean = 0.5 * (f["Nu_hot"] + f["Nu_cold"])
    out = {
        "metric/psi_max": f["psi_max"],
        "metric/Nu_hot": f["Nu_hot"],
        "metric/Nu_cold": f["Nu_cold"],
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
