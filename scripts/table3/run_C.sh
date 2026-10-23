#!/usr/bin/env bash
# Giai đoạn C, Bảng 3 (Liu, Wang, Jv & Guo 2021, CiCP 29, Table 3): hốc xốp đồng nhất, Pr = 1,
# F_eps = 1.75/sqrt(150 eps^3) (Eq. 2.4), ffv_compact, 50k bước. 18 run, tiền tố tab3C_*.
# Dùng: bash scripts/table3/run_C.sh [--dry_run]
cd "$(dirname "$0")/../.." || exit 1
run() {  # $1 = prefix, $2 = Da, $3 = eps, $4 = F_eps, $5 = danh sách Ra
  venv/bin/python scripts/sweep.py --config=cases/porous_cavity/config/base.py --prefix="$1" \
    --set physics.Pr=1.0 --set physics.Da="$2" --set physics.epsilon="$3" --set physics.F_eps="$4" \
    --set diff.scheme=ffv_compact --grid physics.Ra="$5" "${@:6}"
}
run tab3C_da2_e04 1e-2 0.4 0.564809 1e3,1e4,1e5 "$@"
run tab3C_da2_e06 1e-2 0.6 0.307444 1e3,1e4,1e5 "$@"
run tab3C_da4_e04 1e-4 0.4 0.564809 1e5,1e6,1e7 "$@"
run tab3C_da4_e06 1e-4 0.6 0.307444 1e5,1e6,1e7 "$@"
run tab3C_da6_e04 1e-6 0.4 0.564809 1e7,1e8,1e9 "$@"
run tab3C_da6_e06 1e-6 0.6 0.307444 1e7,1e8,1e9 "$@"
