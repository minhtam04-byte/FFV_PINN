#!/usr/bin/env bash
# Giai đoạn D (Da = 1e-6, eps = 0.4, Ra = 1e7 và 1e9): train chưa đủ hay thiếu độ phân giải?
#   D1: chép run tab3C rồi resume 50k -> 100k, giữ lịch lr (decay_steps = 2000)
#   D2: train mới 150k bước, decay_steps = 6000 (lr cuối = lr của C tại 50k)
# Dùng: bash scripts/table3/run_D.sh. Chạy lại script này an toàn: D1 tự resume, D2 qua sweep bỏ qua run đã xong.
cd "$(dirname "$0")/../.." || exit 1
O=outputs/porous_cavity
COMMON=(--config.physics.Pr=1.0 --config.physics.Da=1e-6 --config.physics.epsilon=0.4
        --config.physics.F_eps=0.564809 --config.diff.scheme=ffv_compact)
for ra in 1e7 1e9; do
  src=$O/porous_cavity_tab3C_da6_e04_Ra$ra; dst=$O/porous_cavity_tab3D1_da6_e04_Ra$ra
  [ -d "$dst" ] || cp -a "$src" "$dst"
  steps=$(venv/bin/python -c "import pickle; print(pickle.load(open('$dst/history.pkl','rb'))['step'][-1])")
  if [ "$steps" -lt 99900 ]; then
    echo "$(date +%H:%M:%S) start D1 Ra=$ra (từ bước $steps)"
    venv/bin/python scripts/train.py --config=cases/porous_cavity/config/base.py --exp_name=tab3D1_da6_e04_Ra$ra \
      "${COMMON[@]}" --config.physics.Ra=$ra --config.training.max_steps=100000 --config.training.resume=True \
      >> $O/logs/tab3D1_da6_e04_Ra$ra.log 2>&1
    echo "$(date +%H:%M:%S) end   D1 Ra=$ra rc=$?"
  else
    echo "skip D1 Ra=$ra (đã xong)"
  fi
done
venv/bin/python scripts/sweep.py --config=cases/porous_cavity/config/base.py --prefix=tab3D2_da6_e04 \
  --set physics.Pr=1.0 --set physics.Da=1e-6 --set physics.epsilon=0.4 --set physics.F_eps=0.564809 \
  --set diff.scheme=ffv_compact --set training.max_steps=150000 --set optim.decay_steps=6000 \
  --grid physics.Ra=1e7,1e9
