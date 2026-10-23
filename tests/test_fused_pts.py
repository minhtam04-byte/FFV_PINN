"""Shared stencil evaluation for pseudo-time stepping: config.diff.fuse_pts_stencil.

With the option, the "ffv_compact" residual and the sol_net cell average come from one
network evaluation on the 5-point stencil. The losses and gradients must equal the default
path up to floating-point rounding. Default (off) behaviour is covered by the golden
regression test.
"""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cases import experiment
from cases.fluid_porous_layer.config import aluminum


def _model(fuse, scheme="ffv_compact"):
  cfg = copy.deepcopy(aluminum.get_config())
  cfg.arch.hidden_dim = 32
  cfg.arch.fourier_emb.embed_dim = 32
  cfg.diff.scheme = scheme
  cfg.diff.h_stencil = 1.0 / cfg.diff.n_x
  cfg.diff.fuse_pts_stencil = fuse
  return experiment.build_model("fluid_porous_layer", cfg)


def _state_and_batch(model):
  # các điểm phủ cả ô cắt mặt phân cách; prev_params lệch khỏi params, pts_weights khác 1
  x = jnp.linspace(0.05, 0.95, 64); y = jnp.linspace(0.07, 0.93, 64)
  batch = jnp.stack([x, y], axis=1)
  state = model.state
  prev = jax.tree_util.tree_map(lambda p: p * 1.01, state.params)
  state = state.replace(prev_params=prev, pts_weights={k: 3.7 for k in state.pts_weights})
  return state, batch


def test_losses_and_gradients_match_default_path():
  m0, m1 = _model(False), _model(True)
  state, batch = _state_and_batch(m0)
  (l0, d0), g0 = jax.value_and_grad(m0.loss, has_aux=True)(state.params, state, batch)
  (l1, d1), g1 = jax.value_and_grad(m1.loss, has_aux=True)(state.params, state, batch)
  assert set(d0) == set(d1)
  for k in d0:
    np.testing.assert_allclose(d1[k], d0[k], rtol=1e-6, err_msg=k)
  # cùng phép toán, khác thứ tự cộng dồn float32: so theo chuẩn của cả vector gradient
  ga = jnp.concatenate([g.ravel() for g in jax.tree_util.tree_leaves(g0)])
  gb = jnp.concatenate([g.ravel() for g in jax.tree_util.tree_leaves(g1)])
  assert float(jnp.linalg.norm(gb - ga) / jnp.linalg.norm(ga)) < 1e-5


def test_rejects_other_schemes():
  model = _model(True, scheme="ffv_continuous")
  state, batch = _state_and_batch(model)
  with pytest.raises(ValueError, match="fuse_pts_stencil"):
    model.compute_residual_losses(state.params, state, batch, pseudo_time=True)
