"""Tổng hợp đo tốc độ (run_speed.sh) theo Bảng 2 của Wang et al.: giây / 100 bước, mean ± std.

Dùng:  venv/bin/python scripts/table3/speed_summary.py [mẫu_log ...]   (mặc định outputs/porous_cavity/logs/speed_*.log)
Đọc dòng "Iter: N  Time: t" của logger (t = thời gian 100 bước, không tính evaluator). Bỏ 1000 bước đầu (JIT)
và các khoảng ở bước 1000 và 1100 (JIT lần đầu của cập nhật GradNorm/PTS). Cột "cả run" giữ mọi khoảng từ bước 1000.
"""
import _common  # noqa: F401  (gốc repo)

import glob
import os
import re
import statistics as st
import sys

from _common import OUT

patterns = sys.argv[1:] or [os.path.join(OUT, "logs", "speed_*.log")]
files = sorted(f for p in patterns for f in glob.glob(p) if "sweep" not in os.path.basename(f))
print(f"{'run':46s} {'n':>3s} {'s/100 bước (sạch)':>18s} {'cả run':>14s}")
for f in files:
  d = [(int(m.group(1)), float(m.group(2)))
       for m in re.finditer(r"Iter: (\d+)\s+Time: ([\d.]+)", open(f).read()) if int(m.group(1)) >= 1000]
  if len(d) < 3:
    continue
  clean = [t for s, t in d if s not in (1000, 1100)]
  alls = [t for _, t in d]
  print(f"{os.path.basename(f)[:-4]:46s} {len(clean):3d} {st.mean(clean):8.3f} ± {st.stdev(clean):.3f}"
        f" {st.mean(alls):7.3f} ± {st.stdev(alls):.3f}")
