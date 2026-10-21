"""Metrics for the 1D conjugate-conduction case (analytical reference)."""
import jax.numpy as jnp

from porous_pinn.evaluators import HistoryEvaluator


class Conduction1DEvaluator(HistoryEvaluator):
  """Logs error/l2, the relative L2 error against the exact solution (Gao et al. 2017,
  Eq. (44)) on 200 uniform points.
  """

  def metrics(self, model, params):
    """Returns:
        {"error/l2": relative L2 error of T}.
    """
    x_test = jnp.linspace(0.0, self.config.physics.L, 200)
    return {"error/l2": model.compute_l2_error(params, x_test)["T_l2_error"]}
