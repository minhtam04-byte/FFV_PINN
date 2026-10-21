"""Train / save / reload round trip through cases.experiment (tiny 1D run)."""
import os

import jax
import numpy as np

from cases import experiment
from cases.conduction_1d.config import base


def _tiny(config, tmp_path):
  config.training.max_steps = 3
  config.logging.log_every_steps = 1
  config.saving.save_every_steps = 1
  config.saving.ckpt_path = str(tmp_path)
  return config


def test_train_then_load_run(tmp_path):
  modify = lambda cfg: _tiny(cfg, tmp_path)
  model, history = experiment.run_experiment("conduction_1d", "roundtrip", base, modify)

  run_dir = os.path.join(str(tmp_path), "conduction_1d_roundtrip")
  assert {"ckpt", "history.pkl", "run_info.json"} <= set(os.listdir(run_dir))
  assert history["step"] == [0, 1, 2] and "error/l2" in history and "loss/total" in history

  loaded, loaded_history, config = experiment.load_run(
      "conduction_1d", "roundtrip", base, modify_config_fn=modify)
  assert int(loaded.state.step) == 3
  for a, b in zip(jax.tree.leaves(model.state.params), jax.tree.leaves(loaded.state.params)):
    np.testing.assert_array_equal(a, b)
  assert loaded_history == history


def test_load_missing_run_raises_without_creating_it(tmp_path):
  import pytest
  modify = lambda cfg: _tiny(cfg, tmp_path)
  with pytest.raises(FileNotFoundError):
    experiment.load_run("conduction_1d", "does_not_exist", base, modify_config_fn=modify)
  assert os.listdir(tmp_path) == []
