#!/usr/bin/env bash
# Giai đoạn E (Bảng 3): mọi run tab3C resume 50k -> 100k, giữ lịch lr (decay_steps = 2000), như D1.
# Bản chép tab3E_* giữ nguyên run tab3C 50k. Hai case đã làm ở D1 được chép từ tab3D1.
# Dùng: bash scripts/table3/run_E.sh (sau run_C.sh và run_D.sh). Chạy lại an toàn: run đã đủ 100k bị bỏ qua, run dở tự resume.
cd "$(dirname "$0")/../.." || exit 1
O=outputs/porous_cavity
for da in da6 da4 da2; do for e in e04 e06; do for src in $O/porous_cavity_tab3C_${da}_${e}_Ra*; do
  sfx=${src##*/porous_cavity_tab3C_}; dst=$O/porous_cavity_tab3E_$sfx
  if [ ! -d "$dst" ]; then
    if [ -d "$O/porous_cavity_tab3D1_$sfx" ]; then cp -a "$O/porous_cavity_tab3D1_$sfx" "$dst"; else cp -a "$src" "$dst"; fi
  fi
  read -r steps args < <(venv/bin/python - "$dst" <<'PY'
import json, pickle, sys
d = sys.argv[1]; p = json.load(open(d + "/run_info.json"))["config"]["physics"]
s = pickle.load(open(d + "/history.pkl", "rb"))["step"][-1]
print(s, " ".join(f"--config.physics.{k}={p[k]}" for k in ("Pr", "Da", "Ra", "epsilon", "F_eps")))
PY
)
  if [ "$steps" -ge 99900 ]; then echo "skip  $sfx (đã đủ 100k)"; continue; fi
  echo "$(date +%H:%M:%S) start $sfx (từ bước $steps)"
  venv/bin/python scripts/train.py --config=cases/porous_cavity/config/base.py --exp_name=tab3E_$sfx \
    $args --config.diff.scheme=ffv_compact --config.training.max_steps=100000 --config.training.resume=True \
    >> $O/logs/tab3E_$sfx.log 2>&1
  echo "$(date +%H:%M:%S) end   $sfx rc=$?"
done; done; done
