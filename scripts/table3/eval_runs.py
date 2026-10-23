"""Đánh giá nhanh các run porous_cavity có cùng tiền tố trên lưới 401² (không ghi file).

Dùng:  venv/bin/python scripts/table3/eval_runs.py <tiền_tố_run> [--ref=7.808]
  Ví dụ: tab3E_da4_e04_Ra1e7 (1 run), tab3E_da4_e04 (3 run Ra), tab3E (18 run).
In: scheme, F_eps, bước cuối, Nu vách nóng/lạnh/giữa, sai số khối lượng, ψ_max, loss, độ trôi Nu trong 10k bước cuối, commit.
"""
import _common  # noqa: F401  (gốc repo, sys.path)

import glob
import os
import sys

import numpy as np

from _common import OUT, load
from porous_pinn.evaluators import convection_fields

prefix = sys.argv[1]
ref = float(next((a.split("=")[1] for a in sys.argv[2:] if a.startswith("--ref=")), "nan"))

rows = []
for d in sorted(glob.glob(os.path.join(OUT, f"porous_cavity_{prefix}*"))):
  name = os.path.basename(d)[len("porous_cavity_"):]
  model, hist, info = load(name)
  cfg = model.config
  f = {k: float(v) for k, v in convection_fields(model, model.state.params, 401).items()}
  steps = np.asarray(hist["step"])
  nu = np.asarray(hist["metric/Nu_hot"])
  last10k = nu[steps >= steps[-1] - 10000]
  drift = (last10k.max() - last10k.min()) / abs(nu[-1])
  rows.append((name, cfg.diff.scheme, cfg.physics.F_eps, int(steps[-1]), f["Nu_hot"], f["Nu_cold"], f["Nu_mid"],
               abs(f["Nu_hot"] - f["Nu_cold"]) / (0.5 * (f["Nu_hot"] + f["Nu_cold"])), f["mass_imbalance"],
               f["psi_max"], float(np.asarray(hist["loss/total"])[-1]), drift, info["git_commit"][:7]))

hdr = ("run", "scheme", "F_eps", "step", "Nu_hot", "Nu_cold", "Nu_mid", "|hot-cold|/mean", "mass_imb", "psi_max",
       "loss", "drift_10k", "commit")
print("\t".join(hdr))
for r in rows:
  print("\t".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in r))
if ref == ref:
  print(f"\nref Nu = {ref}")
  for r in rows:
    print(f"{r[0]}: Nu_hot {100 * (r[4] / ref - 1):+.2f}%  Nu_mid {100 * (r[6] / ref - 1):+.2f}%")
