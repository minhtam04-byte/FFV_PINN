"""Train one experiment from the command line.

The case is taken from the config path (cases/<case>/config/...). Any config
field can be overridden on the command line:

  python scripts/train.py \
      --config=cases/fluid_porous_layer/config/aluminum.py \
      --exp_name=ffv_continuous_seed1 \
      --config.diff.scheme=ffv_continuous \
      --config.diff.use_continuous_mesh=True \
      --config.seed=1

The run is saved to outputs/<case>/<config.wandb.name>_<exp_name>/ (checkpoints,
history.pkl, run_info.json). Continue an interrupted run with
--config.training.resume=True. W&B is off unless WANDB_MODE is set.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from absl import app, flags
from ml_collections import config_flags

from cases import experiment

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
  case = parts[parts.index("cases") + 1]
  if case not in experiment.CASES:
    raise ValueError(f"Unknown case '{case}'; known: {sorted(experiment.CASES)}")
  return case


def main(argv):
  del argv
  config = _CONFIG.value
  case = _case_from_config_path(config_flags.get_config_filename(flags.FLAGS["config"]))

  config.wandb.name = f"{config.wandb.name}_{_EXP_NAME.value}"
  config.saving.ckpt_path = _OUTPUTS_DIR.value or experiment.case_outputs_dir(case)
  run_dir = experiment.run_dir(config)
  if os.path.exists(run_dir) and not config.training.get("resume", False):
    raise SystemExit(
        f"Run directory already exists: {run_dir}\n"
        "Use --config.training.resume=True to continue it, or choose another --exp_name."
    )
  experiment.train_case(case, config)


if __name__ == "__main__":
  app.run(main)
