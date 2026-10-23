"""Phân tích lịch sử loss của một giai đoạn run Bảng 3 (dùng để chẩn đoán Da = 1e-6 chưa hội tụ ở 50k).

Dùng:  venv/bin/python scripts/table3/loss_analysis.py [GIAI_ĐOẠN]   (mặc định tab3C)
In theo từng run: log10(loss) trung vị mỗi đoạn 10k, độ dốc loss 30k → cuối, độ dốc và nhiễu của Nu,
sai số khối lượng; thành phần loss lớn nhất, raw residual, lr cuối và trọng số PTS trong 10k bước cuối.
"""
import _common  # noqa: F401  (gốc repo)

import glob
import os
import pickle
import sys

import numpy as np

from _common import OUT

STAGE = sys.argv[1] if len(sys.argv) > 1 else "tab3C"
TERMS = ["u_res", "v_res", "p_res", "T_res", "u_bc", "v_bc", "T_dirichlet_bc", "T_neumann_bc"]


def slope_per_10k(s, y, a, b):
  k = (s >= a) & (s < b)
  return np.polyfit(s[k] / 1e4, y[k], 1)[0]


rows = []
for d in sorted(glob.glob(os.path.join(OUT, f"porous_cavity_{STAGE}_*"))):
  h = pickle.load(open(d + "/history.pkl", "rb"))
  s = np.asarray(h["step"], float)
  end = s[-1] + 100
  L = np.log10(np.asarray(h["loss/total"]))
  nu = np.asarray(h["metric/Nu_hot"]); mi = np.asarray(h["error/mass_imbalance"])
  name = d.split(f"{STAGE}_")[1]

  def win(y, a, b):
    k = (s >= a) & (s < b)
    return y[k]

  med = [np.median(win(L, a, a + 10000)) for a in range(0, int(end), 10000)]
  sl = slope_per_10k(s, L, 30000, end)
  k = s >= 30000
  p = np.polyfit(s[k] / 1e4, nu[k], 1); resid = nu[k] - np.polyval(p, s[k] / 1e4)
  comp = {t: np.median(win(np.asarray(h["loss/" + t]), end - 10000, end)) for t in TERMS if "loss/" + t in h}
  tot = sum(comp.values()); top = sorted(comp.items(), key=lambda kv: -kv[1])[:2]
  raw = {t: np.median(win(np.asarray(h["raw_loss/" + t]), end - 10000, end)) for t in ("u_res", "v_res", "p_res", "T_res")}
  pts = {v: np.median(win(np.asarray(h["pts_weights/" + v]), end - 10000, end)) for v in "uvpT"}
  rows.append((name, med, sl, p[0], resid.std(), np.mean(win(nu, end - 10000, end)), slope_per_10k(s, mi, 30000, end),
               np.mean(win(mi, end - 10000, end)), top, tot, raw, h["lr"][-1], pts))

print("log10(loss) trung vị theo đoạn 10k  |  dốc 30k-cuối (decade/10k)  |  Nu: dốc/10k, nhiễu std, TB 10k cuối  |  mass: dốc/10k, TB")
for r in rows:
  name, med, sl, nslope, nstd, nmean, mslope, mmean = r[:8]
  print(f"{name:16s} {' '.join(f'{m:6.2f}' for m in med)} | {sl:+6.3f} | Nu {nslope:+7.3f} {nstd:6.3f} {nmean:7.3f} "
        f"| mass {100 * mslope:+6.2f}% {100 * mmean:5.1f}%")
print("\nThành phần loss (có trọng số) lớn nhất ở 10k cuối, raw residual u/v/p/T, lr cuối, PTS weights u/v/p/T")
for r in rows:
  name, top, tot, raw, lr, pts = r[0], r[8], r[9], r[10], r[11], r[12]
  print(f"{name:16s} " + ", ".join(f"{t} {100 * v / tot:4.1f}%" for t, v in top)
        + " | raw " + " ".join(f"{raw[t]:.1e}" for t in raw) + f" | lr {lr:.1e} | pts " + " ".join(f"{pts[v]:.2g}" for v in "uvpT"))
