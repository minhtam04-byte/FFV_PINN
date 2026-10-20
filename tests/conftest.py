"""Run every test on CPU (deterministic, independent of GPU matmul precision)."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("WANDB_MODE", "disabled")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)


def pytest_configure(config):
  config.addinivalue_line("markers", "slow: trains a model for minutes (run with -m slow)")
