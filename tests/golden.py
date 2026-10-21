"""Golden values guarding against unintended numerical changes.

compute_golden() evaluates, on CPU with fixed inputs:
  A/...  residuals, sol_net and losses of every case config x diff scheme at
         initialization (full-size networks);
  B/...  parameters, pseudo-time weights and loss weights after a few steps
         through the shared trainer (small networks, every update path firing).

Regenerate tests/golden/golden.npz with `python tests/make_golden.py` ONLY after
an intended numerical change, and say so in the commit message.
"""
import copy
import tempfile

import jax
import numpy as np

from cases import experiment
from cases.conduction_1d.batching import create_batches_1d
from cases.conduction_1d.config import base as conduction_base
from cases.fluid_porous_layer.config import aluminum, glass
from cases.porous_cavity.config import base as cavity_base
from porous_pinn.batching import create_batches

SCHEMES_2D = ["ad", "ffv", "ffv_continuous", "ffv_compact", "can_ffv"]
SCHEMES_1D = ["ad", "ffv", "can"]
CONFIGS_2D = [
    ("porous_cavity", "porous_cavity/base", cavity_base),
    ("fluid_porous_layer", "fluid_porous_layer/glass", glass),
    ("fluid_porous_layer", "fluid_porous_layer/aluminum", aluminum),
]
TRAIN_RUNS = [  # (case, name, config module, scheme, continuous mesh)
    ("conduction_1d", "conduction_1d/base", conduction_base, "ffv", False),
    ("porous_cavity", "porous_cavity/base", cavity_base, "ffv_continuous", True),
    ("fluid_porous_layer", "fluid_porous_layer/aluminum", aluminum, "ffv_continuous", True),
    ("fluid_porous_layer", "fluid_porous_layer/aluminum", aluminum, "ad", False),
    ("fluid_porous_layer", "fluid_porous_layer/glass", glass, "ffv", False),
    ("fluid_porous_layer", "fluid_porous_layer/aluminum", aluminum, "ffv_compact", True),
]


def _flatten(prefix, tree, out):
  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    out[prefix + jax.tree_util.keystr(path)] = np.asarray(leaf)


def _small(config):
  """A run that trains in seconds on CPU while every update path fires."""
  config.arch.hidden_dim = 32
  config.arch.fourier_emb.embed_dim = 32
  config.training.batch_size = 64
  config.training.max_steps = 6
  config.loss_weighting.update_schedule.start = 1
  config.loss_weighting.update_schedule.every = 2
  config.pseudo_time.update_schedule.start = 2
  config.pseudo_time.update_schedule.every = 2
  config.saving.save_every_steps = 0
  return config


def compute_golden(include_training=True):
  out = {}
  b2 = jax.random.uniform(jax.random.PRNGKey(0), (64, 2), minval=0.05, maxval=0.95)
  b1 = jax.random.uniform(jax.random.PRNGKey(0), (64, 1), minval=0.05, maxval=0.95)

  for case, name, mod in CONFIGS_2D:
    for scheme in SCHEMES_2D:
      cfg = copy.deepcopy(mod.get_config())
      cfg.diff.scheme = scheme
      create_batches(cfg)  # sets h_stencil
      m = experiment.build_model(case, cfg)
      p = m.state.params
      _flatten(f"A/{name}/{scheme}/res", m.r_pred_fn(p, b2[:, 0], b2[:, 1]), out)
      _flatten(f"A/{name}/{scheme}/sol", m.sol_pred_fn(p, b2[:, 0], b2[:, 1]), out)
      _flatten(f"A/{name}/{scheme}/loss", m.losses(p, m.state, b2), out)
  # ffv_compact with the Hirata momentum form (interface-consistent one-domain formulation)
  cfg = copy.deepcopy(aluminum.get_config())
  cfg.diff.scheme = "ffv_compact"
  cfg.physics.momentum_form = "hirata"
  create_batches(cfg)
  m = experiment.build_model("fluid_porous_layer", cfg)
  _flatten("A/fluid_porous_layer/aluminum/ffv_compact_hirata/res", m.r_pred_fn(m.state.params, b2[:, 0], b2[:, 1]), out)
  _flatten("A/fluid_porous_layer/aluminum/ffv_compact_hirata/loss", m.losses(m.state.params, m.state, b2), out)

  for scheme in SCHEMES_1D:
    cfg = copy.deepcopy(conduction_base.get_config())
    cfg.diff.scheme = scheme
    create_batches_1d(cfg)
    m = experiment.build_model("conduction_1d", cfg)
    p = m.state.params
    _flatten(f"A/conduction_1d/base/{scheme}/res", m.r_pred_fn(p, b1[:, 0]), out)
    _flatten(f"A/conduction_1d/base/{scheme}/loss", m.losses(p, m.state, b1), out)

  if include_training:
    tmp = tempfile.mkdtemp()
    for i, (case, name, mod, scheme, cont) in enumerate(TRAIN_RUNS):
      cfg = _small(copy.deepcopy(mod.get_config()))
      cfg.diff.scheme = scheme
      cfg.diff.use_continuous_mesh = cont
      cfg.wandb.name = f"golden_{i}"
      cfg.saving.ckpt_path = tmp
      model, _ = experiment.train_case(case, cfg)
      _flatten(f"B/{name}/{scheme}/params", model.state.params, out)
      _flatten(f"B/{name}/{scheme}/pts", model.state.pts_weights, out)
      _flatten(f"B/{name}/{scheme}/lw", model.state.loss_weights, out)
  return out
