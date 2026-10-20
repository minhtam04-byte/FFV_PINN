"""Train one experiment from the command line.

The case is taken from the config path (cases/<case>/config/...). Any config
field can be overridden on the command line:

  python scripts/train.py \
      --config=cases/fluid_porous_layer/config/base_line_case_2.py \
      --exp_name=ffv_continuous_pts_seed1 \
      --config.diff.scheme=ffv_continuous \
      --config.diff.use_continuous_mesh=True \
      --config.seed=1

The run is saved to outputs/<case>/<config.wandb.name>_<exp_name>/, the layout
that porous_pinn.experiment.load_saved_model() reads, together with
run_info.json (resolved config, git commit, command line).
"""
import importlib
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from absl import app, flags
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "Path to a case config file.", lock_config=False
)
_EXP_NAME = flags.DEFINE_string(
    "exp_name", None, "Suffix appended to config.wandb.name to name the run."
)
_OUTPUTS_DIR = flags.DEFINE_string(
    "outputs_dir", None, "Where to save the run. Default: outputs/<case>."
)
flags.mark_flags_as_required(["config", "exp_name"])


def _case_from_config_path(path):
  parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
  if "cases" not in parts:
    raise ValueError(f"Config must live under cases/<case>/config/: {path}")
  return parts[parts.index("cases") + 1]


def _git(*args):
  try:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def _save_run_info(run_dir, config, case):
  os.makedirs(run_dir, exist_ok=True)
  info = {
      "case": case,
      "argv": sys.argv,
      "git_commit": _git("rev-parse", "HEAD"),
      "git_dirty": bool(_git("status", "--porcelain")),
      "config": json.loads(config.to_json_best_effort()),
  }
  with open(os.path.join(run_dir, "run_info.json"), "w") as f:
    json.dump(info, f, indent=2)


def main(argv):
  del argv
  config = _CONFIG.value
  config_path = config_flags.get_config_filename(flags.FLAGS["config"])
  case = _case_from_config_path(config_path)

  config.wandb.name = f"{config.wandb.name}_{_EXP_NAME.value}"
  outputs_dir = _OUTPUTS_DIR.value or os.path.join(REPO_ROOT, "outputs", case)
  run_dir = os.path.join(outputs_dir, config.wandb.name)
  if os.path.exists(run_dir):
    raise SystemExit(f"Run directory already exists, refusing to overwrite: {run_dir}")

  # create_batches* set config.diff.h_stencil, so build batches before saving the config
  if case == "conduction_1d":
    from cases.conduction_1d.experiment import create_batches_1d
    from cases.conduction_1d.train import train_1d

    batches = create_batches_1d(config)
    _save_run_info(run_dir, config, case)
    train_1d(config=config, sampler=batches, outputs_dir=outputs_dir)
  else:
    from porous_pinn.batching import create_batches

    trainer = importlib.import_module(f"cases.{case}.train")
    batches = create_batches(config)
    _save_run_info(run_dir, config, case)
    trainer.train_porous(config=config, samplers=batches, outputs_dir=outputs_dir)


if __name__ == "__main__":
  app.run(main)
