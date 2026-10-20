"""Every case config loads, builds its model, and honours the name-keyed contracts:
loss_weights keys == keys of model.losses(), pts_weights keys == model.variables.
"""
import copy
import glob
import importlib
import os

import jax
import jax.numpy as jnp
import pytest

from cases import experiment

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(
    (path.split(os.sep)[-3], os.path.splitext(os.path.basename(path))[0])
    for path in glob.glob(os.path.join(REPO_ROOT, "cases", "*", "config", "*.py"))
    if not path.endswith("__init__.py")
)


def test_every_case_has_a_config():
  assert {case for case, _ in CONFIGS} == set(experiment.CASES)


@pytest.mark.parametrize("case,name", CONFIGS)
def test_config_contracts(case, name):
  config = copy.deepcopy(importlib.import_module(f"cases.{case}.config.{name}").get_config())
  model = experiment.build_model(case, config)
  batch = jnp.full((8, config.input_dim), 0.5)
  loss_keys = jax.eval_shape(model.losses, model.state.params, model.state, batch).keys()
  assert set(config.loss_weighting.loss_weights.keys()) == set(loss_keys)
  assert set(config.pseudo_time.pts_weights.keys()) == set(model.variables)
