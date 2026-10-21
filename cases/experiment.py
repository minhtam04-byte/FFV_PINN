"""Train and reload runs of every case with the shared jaxpi trainer.

Run layout: outputs/<case>/<config.wandb.name>/{ckpt/, history.pkl, run_info.json}.
W&B logging is disabled unless WANDB_MODE is set (e.g. WANDB_MODE=offline).
"""
import copy
import importlib
import json
import math
import os
import pickle
import subprocess
import sys

os.environ.setdefault("WANDB_MODE", "disabled")

import jax
import wandb

from jaxpi.checkpointing import (create_checkpoint_manager, get_ckpt_path,
                                 has_checkpoint_steps, restore_checkpoint)
from jaxpi.models import (create_arch, create_lr_schedule, create_model, create_optimizer,
                          create_train_state)
from jaxpi.training import train

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_ROOT = os.path.join(REPO_ROOT, "outputs")

# case -> (model class, batch builder, evaluator class), as "module:attribute"
CASES = {
    "conduction_1d": (
        "cases.conduction_1d.models:ConjugateHeatConduction1D",
        "cases.conduction_1d.batching:create_batches_1d",
        "cases.conduction_1d.evaluators:Conduction1DEvaluator",
    ),
    "porous_cavity": (
        "porous_pinn.models:PorousNaturalConvection2D",
        "porous_pinn.batching:create_batches",
        "porous_pinn.evaluators:ConvectionEvaluator",
    ),
    "fluid_porous_layer": (
        "porous_pinn.models:PorousNaturalConvection2D",
        "porous_pinn.batching:create_batches",
        "porous_pinn.evaluators:ConvectionEvaluator",
    ),
}


def _load(spec):
  module, attr = spec.split(":")
  return getattr(importlib.import_module(module), attr)


def case_outputs_dir(case):
  return os.path.join(OUTPUTS_ROOT, case)


def run_dir(config):
  return os.path.dirname(get_ckpt_path(config))


def apply_runtime(config):
  """Process-wide JAX settings recorded in the config."""
  precision = config.get("jax_matmul_precision", None)
  if precision is not None:
    jax.config.update("jax_default_matmul_precision", precision)


def build_model(case, config, params=None):
  apply_runtime(config)
  model_cls = _load(CASES[case][0])
  width = (model_cls.network_input_dim(config) if hasattr(model_cls, "network_input_dim")
           else config.input_dim)
  if width == config.input_dim:  # default path, unchanged
    return create_model(config, model_cls, params=params)
  # The network takes extra input features: initialise it with the wider input while
  # config.input_dim stays the number of coordinates (used by the vmapped predictors).
  lr = create_lr_schedule(config.optim)
  tx = create_optimizer(config.optim, lr)
  arch = create_arch(config.arch)
  init_config = copy.deepcopy(config)
  init_config.input_dim = width
  state = create_train_state(init_config, tx, arch, params=params)
  return model_cls(config, lr, tx, arch, state)


def stop_on_nonfinite(step, log_dict):
  """Early stop when any logged loss is NaN/inf."""
  return any(not math.isfinite(float(v)) for k, v in log_dict.items() if k.startswith("loss/"))


def _git(*args):
  try:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def _save_run_info(config, case):
  path = run_dir(config)
  os.makedirs(path, exist_ok=True)
  info = {
      "case": case,
      "argv": sys.argv,
      "git_commit": _git("rev-parse", "HEAD"),
      "git_dirty": bool(_git("status", "--porcelain")),
      "config": json.loads(config.to_json_best_effort()),
  }
  with open(os.path.join(path, "run_info.json"), "w") as f:
    json.dump(info, f, indent=2)


def _pretrained_params(case, config):
  """Warm start from training.pretrained_ckpt_dir (fresh optimizer state)."""
  ckpt_dir = config.training.get("pretrained_ckpt_dir", None)
  if not ckpt_dir:
    return None
  donor = build_model(case, config)
  state = restore_checkpoint(create_checkpoint_manager(config.saving, ckpt_dir), donor.state)
  return state.params


def train_case(case, config, stop_fn=stop_on_nonfinite):
  """Train one run; returns (model, history).

  config.saving.ckpt_path defaults to outputs/<case>. With
  config.training.resume the run continues from its latest checkpoint and its
  history.pkl is extended.
  """
  if config.saving.get("ckpt_path", None) is None:
    config.saving.ckpt_path = case_outputs_dir(case)
  apply_runtime(config)

  batches = _load(CASES[case][1])(config)  # sets config.diff.h_stencil for FV schemes
  model = build_model(case, config, params=_pretrained_params(case, config))
  evaluator = _load(CASES[case][2])(config)
  _save_run_info(config, case)

  train(config, model, batches, evaluator=evaluator, stop_fn=stop_fn)
  wandb.finish()

  history = evaluator.history
  history_path = os.path.join(run_dir(config), "history.pkl")
  if config.training.get("resume", False) and os.path.exists(history_path):
    with open(history_path, "rb") as f:
      previous = pickle.load(f)
    history = {k: previous.get(k, []) + history.get(k, []) for k in previous.keys() | history.keys()}
  with open(history_path, "wb") as f:
    pickle.dump(history, f)
  return model, history


def run_experiment(case, exp_name, org_config, modify_config_fn=None):
  """Notebook helper: org_config.get_config(), suffix the run name, apply modify_config_fn, train."""
  config = copy.deepcopy(org_config.get_config())
  config.wandb.name = f"{config.wandb.name}_{exp_name}"
  if modify_config_fn is not None:
    config = modify_config_fn(config)
  return train_case(case, config)


def load_run(case, exp_name, org_config, step=None, modify_config_fn=None):
  """Rebuild the model of a run and restore its checkpoint; returns (model, history, config).

  The architecture must match the run, so pass the same modify_config_fn if
  it changed the network.
  """
  config = org_config.get_config()
  config.wandb.name = f"{config.wandb.name}_{exp_name}"
  if modify_config_fn is not None:
    config = modify_config_fn(config)
  if config.saving.get("ckpt_path", None) is None:
    config.saving.ckpt_path = case_outputs_dir(case)

  ckpt_dir = get_ckpt_path(config)
  # check before creating the manager: Orbax would create an empty run directory
  if not has_checkpoint_steps(ckpt_dir):
    raise FileNotFoundError(f"Không tìm thấy checkpoint nào trong thư mục: {ckpt_dir}")
  model = build_model(case, config)
  ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_dir)
  model.state = restore_checkpoint(ckpt_mngr, model.state, step=step)
  print(f"-> Đã nạp checkpoint step {int(model.state.step)} từ {ckpt_dir}")

  history = {}
  history_path = os.path.join(run_dir(config), "history.pkl")
  if os.path.exists(history_path):
    with open(history_path, "rb") as f:
      history = pickle.load(f)
  return model, history, config
