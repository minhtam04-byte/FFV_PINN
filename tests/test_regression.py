"""Compare current numerics with tests/golden/golden.npz (see tests/golden.py).

A failure means residuals, losses or training dynamics changed. If the change
is intended, regenerate the golden file with `python tests/make_golden.py`.
"""
import os

import numpy as np
import pytest

from golden import compute_golden

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "golden.npz")
RTOL, ATOL = 1e-5, 1e-6  # CPU results are deterministic; slack only for other machines/JAX builds


def _compare(current, prefix):
  ref = np.load(GOLDEN)
  keys = sorted(k for k in ref.files if k.startswith(prefix))
  assert keys, f"no golden entries with prefix {prefix}"
  assert sorted(k for k in current if k.startswith(prefix)) == keys
  bad = [k for k in keys if not np.allclose(current[k], ref[k], rtol=RTOL, atol=ATOL)]
  assert not bad, f"{len(bad)} arrays changed, e.g. {bad[:5]}"


def test_initial_residuals_and_losses():
  _compare(compute_golden(include_training=False), "A/")


@pytest.mark.slow
def test_training_steps():
  _compare(compute_golden(include_training=True), "B/")
