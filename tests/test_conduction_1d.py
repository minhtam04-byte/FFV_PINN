"""End-to-end: train the 1D two-layer conduction case and check it against the
analytical solution (Gao et al. 2017, Sec. 4.1). Slow: run with `pytest -m slow`.
"""
import copy

import pytest

from cases import experiment
from cases.conduction_1d.config import base


@pytest.mark.slow
@pytest.mark.parametrize("scheme,continuous", [("ffv", False), ("ffv", True)])
def test_fv_converges_to_analytical_solution(tmp_path, scheme, continuous):
  config = copy.deepcopy(base.get_config())
  config.diff.scheme = scheme
  config.diff.use_continuous_mesh = continuous
  config.training.max_steps = 5000
  config.saving.save_every_steps = 0
  config.saving.ckpt_path = str(tmp_path)
  _, history = experiment.train_case("conduction_1d", config)
  assert history["error/l2"][-1] < 1e-2
