"""Regenerate tests/golden/golden.npz (CPU).

Only after an INTENDED numerical change; mention it in the commit message.
  python tests/make_golden.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conftest  # noqa: F401  (CPU, W&B off, repo on sys.path)

import numpy as np

from golden import compute_golden

if __name__ == "__main__":
  out = compute_golden()
  path = os.path.join(HERE, "golden", "golden.npz")
  os.makedirs(os.path.dirname(path), exist_ok=True)
  np.savez_compressed(path, **out)
  print(f"wrote {len(out)} arrays to {path}")
