#!/usr/bin/env bash
# Đo tốc độ theo Bảng 2 của Wang et al. ("When PINNs go wrong"): thời gian/100 bước, mean ± std, 1 GPU,
# baseline (không PTS) và có PTS. Case đại diện: Da 1e-4, Ra 1e7, eps 0.4. Mỗi run 5000 bước.
# 3 scheme × PTS {bật, tắt}, cộng ffv_compact + PTS + diff.fuse_pts_stencil. Tiền tố speed_*.
# Tổng hợp: bỏ các khoảng đo ở bước 1000 và 1100 (JIT lần đầu của cập nhật GradNorm/PTS), xem README.md.
# Dùng: bash scripts/table3/run_speed.sh [--dry_run]
cd "$(dirname "$0")/../.." || exit 1
COMMON=(--config=cases/porous_cavity/config/base.py --prefix=speed
  --set physics.Pr=1.0 --set physics.Da=1e-4 --set physics.epsilon=0.4 --set physics.F_eps=0.564809
  --set physics.Ra=1e7 --set training.max_steps=5000 --set saving.save_every_steps=0)
venv/bin/python scripts/sweep.py "${COMMON[@]}" \
  --grid diff.scheme=ad,ffv_continuous,ffv_compact --grid pseudo_time.enabled=True,False "$@"
venv/bin/python scripts/sweep.py "${COMMON[@]}" \
  --set diff.scheme=ffv_compact --set pseudo_time.enabled=True --grid diff.fuse_pts_stencil=True "$@"
