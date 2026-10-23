"""Phần dùng chung của các script Bảng 3: gốc repo, giá trị tham chiếu, dựng lại config của một run.

Import module này trước mọi import của repo. Nó thêm gốc repo vào sys.path và chuyển thư mục làm việc về đó,
nên script chạy được từ bất kỳ đâu (kể cả nút ▶ của VS Code).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

OUT = "outputs/porous_cavity"

# (Da, Ra, eps): (Ref. [22], Ref. [39], DUGKS), Table 3 của Liu, Wang, Jv & Guo (2021), CiCP 29:265
# (references/dugks_porous.pdf). Nu là Nu trung bình trên vách nóng (Eq. 4.8 của bài).
REF = {
    (1e-2, 1e3, 0.4): (1.007, 1.008, 1.008), (1e-2, 1e3, 0.6): (1.012, 1.012, 1.012),
    (1e-2, 1e4, 0.4): (1.362, 1.365, 1.360), (1e-2, 1e4, 0.6): (1.494, 1.498, 1.490),
    (1e-2, 1e5, 0.4): (3.009, 3.012, 2.990), (1e-2, 1e5, 0.6): (3.460, 3.463, 3.435),
    (1e-4, 1e5, 0.4): (1.067, 1.067, 1.064), (1e-4, 1e5, 0.6): (1.069, 1.069, 1.067),
    (1e-4, 1e6, 0.4): (2.630, 2.618, 2.600), (1e-4, 1e6, 0.6): (2.733, 2.734, 2.700),
    (1e-4, 1e7, 0.4): (7.808, 7.811, 7.751), (1e-4, 1e7, 0.6): (8.457, 8.506, 8.543),
    (1e-6, 1e7, 0.4): (1.085, 1.089, 1.077), (1e-6, 1e7, 0.6): (1.089, 1.094, 1.077),
    (1e-6, 1e8, 0.4): (2.949, 3.014, 2.980), (1e-6, 1e8, 0.6): (2.957, 3.035, 2.992),
    (1e-6, 1e9, 0.4): (11.610, 11.733, 11.591), (1e-6, 1e9, 0.6): (12.092, 12.149, 11.995),
}

# Cột "Present" cũ (notebook run_experiment_tab3.ipynb của jaxpi_porous_heat: ffv_continuous, TF32, F_eps = 0.1 cố định).
# KHÔNG dùng được cho bài báo: F_eps quá nhỏ làm Nu cao tới +9%. Chỉ giữ để đối chiếu.
OLD = {
    (1e-2, 1e3, 0.4): 1.008, (1e-2, 1e3, 0.6): 1.012, (1e-2, 1e4, 0.4): 1.399, (1e-2, 1e4, 0.6): 1.521,
    (1e-2, 1e5, 0.4): 3.136, (1e-2, 1e5, 0.6): 3.555, (1e-4, 1e5, 0.4): 1.066, (1e-4, 1e5, 0.6): 1.067,
    (1e-4, 1e6, 0.4): 2.708, (1e-4, 1e6, 0.6): 2.778, (1e-4, 1e7, 0.4): 8.511, (1e-4, 1e7, 0.6): 9.068,
    (1e-6, 1e7, 0.4): 1.073, (1e-6, 1e7, 0.6): 1.076, (1e-6, 1e8, 0.4): 2.910, (1e-6, 1e8, 0.6): 2.897,
    (1e-6, 1e9, 0.4): 11.380, (1e-6, 1e9, 0.6): 10.852,
}


def run_path(name):
  """Thư mục của run có hậu tố `name` (phần sau "porous_cavity_")."""
  return os.path.join(OUT, f"porous_cavity_{name}")


def from_run_info(saved):
  """modify_config_fn cho load_run: chép các nhánh config đã ghi đè lúc train (từ run_info.json) vào config dựng lại."""
  def fn(cfg):
    for sec in ("physics", "diff", "model"):
      for k, v in saved[sec].items():
        cfg[sec][k] = v
    cfg.jax_matmul_precision = saved["jax_matmul_precision"]
    return cfg
  return fn


def load(name):
  """Nạp checkpoint cuối của run `name` với đúng config lúc train. Trả về (model, history, run_info)."""
  from cases.experiment import load_run
  from cases.porous_cavity.config import base
  info = json.load(open(os.path.join(run_path(name), "run_info.json")))
  model, hist, _ = load_run("porous_cavity", name, base, modify_config_fn=from_run_info(info["config"]))
  return model, hist, info
