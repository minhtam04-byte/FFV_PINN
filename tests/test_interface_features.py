"""Interface (cusp-capturing) input features: config.model.interface_features.

The network sees (x, y, |x - x_s|, |x - x_e|). Values stay continuous across the
interfaces while x-derivatives may jump, so a kink such as the temperature profile
across a high-conductivity porous layer can be represented without Gibbs wiggles.
Default (off) behaviour is covered by the golden regression test.
"""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from cases import experiment
from cases.fluid_porous_layer.config import aluminum


def _config(on):
  cfg = copy.deepcopy(aluminum.get_config())
  cfg.arch.hidden_dim = 32
  cfg.arch.fourier_emb.embed_dim = 32
  cfg.diff.scheme = "ffv_compact"
  cfg.model.interface_features = on
  return cfg


def _T(model, params):
  return lambda x, y: model.neural_net(params, x, y)[3]


def test_network_takes_four_inputs_and_predictors_work():
  model = experiment.build_model("fluid_porous_layer", _config(True))
  kernel = model.state.params["params"]["FourierEmbs_0"]["kernel"]
  assert kernel.shape[0] == 4
  x = jnp.linspace(0.1, 0.9, 8); y = jnp.full_like(x, 0.5)
  r = model.r_pred_fn(model.state.params, x, y)
  assert set(r) == {"u", "v", "p", "T"} and all(np.all(np.isfinite(r[k])) for k in r)


def test_derivative_may_jump_only_with_features():
  eta = 1e-4
  jumps = {}
  for on in (False, True):
    model = experiment.build_model("fluid_porous_layer", _config(on))
    dT = jax.grad(_T(model, model.state.params), argnums=0)
    x_s = model.config.physics.porous_x_start
    jumps[on] = abs(float(dT(x_s + eta, 0.5) - dT(x_s - eta, 0.5)))
  assert jumps[True] > 100 * jumps[False], jumps


def _fit_error(on, steps=1500):
  """Fit T(x, 0.5) to a flux-continuous piecewise-linear profile across x_s; max error near x_s."""
  model = experiment.build_model("fluid_porous_layer", _config(on))
  phys = model.config.physics
  x_s, lam = phys.porous_x_start, phys.lambda_k
  xs = jnp.linspace(x_s - 0.1, x_s + 0.1, 401)
  target = jnp.where(xs < x_s, 0.5 - 2.0 * (xs - x_s), 0.5 - (2.0 / lam) * (xs - x_s))
  T = jax.vmap(lambda p, x: model.neural_net(p, x, 0.5)[3], (None, 0))
  loss = lambda p: jnp.mean((T(p, xs) - target) ** 2)
  opt = optax.adam(1e-3)
  params, state = model.state.params, opt.init(model.state.params)

  @jax.jit
  def step(params, state):
    g = jax.grad(loss)(params)
    updates, state = opt.update(g, state, params)
    return optax.apply_updates(params, updates), state

  for _ in range(steps):
    params, state = step(params, state)
  near = jnp.abs(xs - x_s) < 0.02
  return float(jnp.max(jnp.abs(T(params, xs) - target)[near]))


def test_features_fit_the_kink_better():
  assert _fit_error(True) < 0.5 * _fit_error(False)
