"""So C / D1 / D2 cho Da = 1e-6, eps = 0.4, Ra = 1e7 và 1e9: loss, lr, mass imbalance, Nu_hot theo cửa sổ 10k bước.

Dùng:  venv/bin/python scripts/table3/compare_D.py
Kết luận (2026-10-22): Da = 1e-6 ở 50k là do train chưa đủ; D1 (resume tới 100k, lr cuối nhỏ) tốt hơn D2.
"""
import _common  # noqa: F401  (gốc repo)

import glob
import os
import pickle

import numpy as np

from _common import OUT, REF

for ra in (1e7, 1e9):
  r22, r39, dugks = REF[(1e-6, ra, 0.4)]
  tag_ra = f"1e{int(round(np.log10(ra)))}"
  print(f"\n=== Ra = {tag_ra}  (ref [22] {r22}, [39] {r39}, DUGKS {dugks})")
  print(f"{'run':6s} {'cửa sổ':>10s} {'log10 loss':>10s} {'lr':>8s} {'mass%':>6s} {'Nu_hot TB±std':>16s} {'lệch [22]':>9s} {'Nu_mid':>7s}")
  for tag in ("C", "D1", "D2"):
    f = glob.glob(os.path.join(OUT, f"porous_cavity_tab3{tag}_da6_e04_Ra{tag_ra}/history.pkl"))
    if not f:
      print(f"{tag:6s} (chưa có)")
      continue
    h = pickle.load(open(f[0], "rb")); s = np.asarray(h["step"])
    g = lambda k: np.asarray(h[k])  # noqa: E731
    for b in range(50000, int(s[-1]) + 101, 10000):
      k = (s >= b - 10000) & (s < b)
      if k.sum() == 0:
        continue
      nu = g("metric/Nu_hot")[k]
      print(f"{tag:6s} {(b - 10000) // 1000:>4d}-{b // 1000:<4d}k {np.median(np.log10(g('loss/total')[k])):10.2f} "
            f"{g('lr')[k][-1]:8.1e} {100 * g('error/mass_imbalance')[k].mean():6.1f} "
            f"{nu.mean():8.3f}±{nu.std():6.3f} {100 * (nu.mean() / r22 - 1):+8.1f}% {g('metric/Nu_mid')[k].mean():7.3f}")
