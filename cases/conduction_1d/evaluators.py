"""Metrics for the 1D conjugate-conduction case (analytical reference)."""
import jax.numpy as jnp

from porous_pinn.evaluators import HistoryEvaluator


class Conduction1DEvaluator(HistoryEvaluator):

  def metrics(self, model, params):
    x_test = jnp.linspace(0.0, self.config.physics.L, 200)
    return {"error/l2": model.compute_l2_error(params, x_test)["T_l2_error"]}
