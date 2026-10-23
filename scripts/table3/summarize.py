"""Tổng hợp Bảng 3 từ một giai đoạn run: in bảng và ghi outputs/porous_cavity/table3_<giai đoạn>.csv.

Dùng:  venv/bin/python scripts/table3/summarize.py [GIAI_ĐOẠN]
  GIAI_ĐOẠN = tiền tố run, mặc định tab3E (bản dùng cho bài). Ví dụ: tab3C (50k), tab3E (100k), tab3F_ad (AD).
  tab3E -> table3_E.csv, tab3F_ad -> table3_F_ad.csv.

Nu báo cáo = trung bình ± độ lệch chuẩn của metric/Nu_hot (ghi mỗi 100 bước) trong 10k bước cuối.
Các cột *_401 đánh giá checkpoint cuối trên lưới 401²: Nu ở vách nóng, vách lạnh, mặt cắt giữa, sai số khối lượng.
"""
import _common  # noqa: F401  (gốc repo, sys.path)

import csv
import os
import pickle
import sys

import numpy as np

from _common import OLD, OUT, REF, load, run_path
from porous_pinn.evaluators import convection_fields

STAGE = sys.argv[1] if len(sys.argv) > 1 else "tab3E"
MIN_STEP = 99900 if STAGE != "tab3C" else 0  # bỏ qua bản chép chưa resume xong

rows = []
for da_tag, da in (("da2", 1e-2), ("da4", 1e-4), ("da6", 1e-6)):
  for e_tag, eps in (("e04", 0.4), ("e06", 0.6)):
    for ra in sorted({k[1] for k in REF if k[0] == da}):
      name = f"{STAGE}_{da_tag}_{e_tag}_Ra1e{int(round(np.log10(ra)))}"
      d = run_path(name)
      if not os.path.exists(os.path.join(d, "history.pkl")):
        print(f"(chưa xong) {name}")
        continue
      h = pickle.load(open(os.path.join(d, "history.pkl"), "rb"))
      s = np.asarray(h["step"])
      if s[-1] < MIN_STEP:
        print(f"(chưa xong, bước {s[-1]}) {name}")
        continue
      nu_hist = np.asarray(h["metric/Nu_hot"])[s >= s[-1] - 10000]
      model, _, info = load(name)
      f = {k: float(v) for k, v in convection_fields(model, model.state.params, 401).items()}
      r22, r39, dugks = REF[(da, ra, eps)]
      mean = float(nu_hist.mean())
      rows.append(dict(Da=da, Ra=ra, eps=eps, ref22=r22, ref39=r39, dugks=dugks, old=OLD[(da, ra, eps)],
                       nu_mean_last10k=mean, nu_std_last10k=float(nu_hist.std()), nu_hot_401=f["Nu_hot"],
                       nu_cold_401=f["Nu_cold"], nu_mid_401=f["Nu_mid"], mass_imb=f["mass_imbalance"],
                       err_vs_22=100 * (mean / r22 - 1), old_err_vs_22=100 * (OLD[(da, ra, eps)] / r22 - 1),
                       loss=float(np.asarray(h["loss/total"])[-1]), step=int(s[-1]), commit=info["git_commit"][:7]))

out = os.path.join(OUT, f"table3_{STAGE[len('tab3'):]}.csv")
if rows:
  with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{'Da':>6} {'Ra':>6} {'eps':>4} | {'[22]':>6} {'[39]':>6} {'DUGKS':>6} | {'cũ':>6} {'mới (10k cuối)':>15} {'401²':>6} | "
      f"{'lệch cũ':>8} {'lệch mới':>8} | {'hot/cold/mid':>18} {'mass':>5}")
for r in rows:
  print(f"{r['Da']:6.0e} {r['Ra']:6.0e} {r['eps']:4.1f} | {r['ref22']:6.3f} {r['ref39']:6.3f} {r['dugks']:6.3f} | "
        f"{r['old']:6.3f} {r['nu_mean_last10k']:7.3f}±{r['nu_std_last10k']:5.3f} {r['nu_hot_401']:6.3f} | "
        f"{r['old_err_vs_22']:+7.1f}% {r['err_vs_22']:+7.1f}% | "
        f"{r['nu_hot_401']:5.2f}/{r['nu_cold_401']:5.2f}/{r['nu_mid_401']:5.2f} {100 * r['mass_imb']:4.1f}%")
print(f"\n-> {out}" if rows else "\nKhông có run nào xong; không ghi CSV.")
