"""Nạp checkpoint run Tab3_11 cũ (jaxpi_porous_heat, cột "Present" cũ) bằng code hiện tại, đánh giá trên lưới 401².

Chỉ để truy nguồn: xác nhận cột cũ dùng ffv_continuous với F_eps = 0.1 (Nu_hot 8.519, khớp 8.511 trong bảng cũ).
Cần ổ HDD của PC có bản jaxpi_porous_heat.
Dùng:  venv/bin/python scripts/table3/eval_old_present.py [hậu_tố_tên_run]
"""
import _common  # noqa: F401  (gốc repo)

import sys

from cases.experiment import load_run
from cases.porous_cavity.config import base
from porous_pinn.evaluators import convection_fields

OLD = "/media/son/HDD-1TB/jaxpi_porous_heat/jaxpi2/nc_porous/outputs"
NAME = "PorousNC_Tab3_11_Da_1e-04_Ra_1e+07_eps_0.4" + (sys.argv[1] if len(sys.argv) > 1 else "")

for prec in ("highest", "default"):
  def fn(cfg, prec=prec):
    cfg.physics.update(Pr=1.0, Ra=1e7, Da=1e-4, epsilon=0.4, F_eps=0.1)
    cfg.diff.scheme = "ffv_continuous"
    cfg.wandb.name = NAME
    cfg.saving.ckpt_path = OLD
    cfg.jax_matmul_precision = prec
    return cfg
  model, _, _ = load_run("porous_cavity", "x", base, modify_config_fn=fn)
  f = {k: float(v) for k, v in convection_fields(model, model.state.params, 401).items()}
  print(prec, {k: round(v, 4) for k, v in f.items()},
        "hot-cold/mean=%.4f" % (abs(f["Nu_hot"] - f["Nu_cold"]) / (0.5 * (f["Nu_hot"] + f["Nu_cold"]))))
