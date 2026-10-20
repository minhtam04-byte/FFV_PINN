"""Run and reload experiments of the 2D porous-convection cases.

Moved from the case notebooks. The case trainer (``train_fn``) and the
outputs directory are parameters because each case has its own trainer and
its own folder under ``outputs/``.
"""
import copy
import os
import pickle

from jaxpi.checkpointing import create_checkpoint_manager, restore_checkpoint
from jaxpi.models import (
    create_arch,
    create_lr_schedule,
    create_optimizer,
    create_train_state,
)
from porous_pinn.batching import create_batches
from porous_pinn.models import PorousNaturalConvection2D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_ROOT = os.path.join(REPO_ROOT, "outputs")


def case_outputs_dir(case):
  """Thư mục outputs của một case, ví dụ outputs/fluid_porous_layer."""
  return os.path.join(OUTPUTS_ROOT, case)


def run_ablation_experiment(
    exp_name, org_config, modify_config_fn=None, *, train_fn, outputs_dir=None
):
    sub_config = copy.deepcopy(org_config.get_config())
    sub_config.wandb.name = f"{sub_config.wandb.name}_{exp_name}"

    if modify_config_fn is not None:
        sub_config = modify_config_fn(sub_config)

    batches = create_batches(sub_config)
    model, history = train_fn(
        config=sub_config, samplers=batches, outputs_dir=outputs_dir
    )
    return model, history


def load_saved_model(exp_name, org_config, step=None, outputs_dir=None):
  """Nạp mô hình và history từ thư mục outputs.

  Args:
      exp_name: Tên thí nghiệm (ví dụ: 'With_PTS_Adaptive')
      step: Bước lặp cần load (int). Nếu None, tự động load bước mới nhất.
      org_config: Cấu hình gốc
      outputs_dir: Thư mục chứa các run. Nếu None, dùng <cwd>/outputs.
  """
  if outputs_dir is None:
    outputs_dir = os.path.join(os.getcwd(), "outputs")

  # 1. Tái tạo cấu hình khớp với thí nghiệm đã chạy
  config = org_config.get_config()
  config.wandb.name = f"{config.wandb.name}_{exp_name}"

  # 2. Khởi tạo train state rỗng theo cấu trúc mạng
  lr = create_lr_schedule(config.optim)
  tx = create_optimizer(config.optim, lr)
  arch = create_arch(config.arch)
  state = create_train_state(config, tx, arch)

  # 3. Khởi tạo đối tượng model
  model = PorousNaturalConvection2D(config, lr, tx, arch, state)

  # 4. Xác định đường dẫn checkpoint và nạp trọng số
  ckpt_dir = os.path.join(outputs_dir, config.wandb.name, "ckpt")
  ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_dir)

  target_step = step if step is not None else ckpt_mngr.latest_step()
  if target_step is None:
    raise FileNotFoundError(
        f"Không tìm thấy checkpoint nào trong thư mục: {ckpt_dir}"
    )

  # Gán đè state đã phục hồi vào model
  model.state = restore_checkpoint(ckpt_mngr, model.state, step=target_step)
  print(f"directory: {ckpt_dir}")
  print(f"-> Đã nạp thành công checkpoint tại step {int(model.state.step)}")

  # 5. Đọc lại file history.pkl (nếu có)
  history_path = os.path.join(outputs_dir, config.wandb.name, "history.pkl")
  history = {}
  if os.path.exists(history_path):
    with open(history_path, "rb") as f:
      history = pickle.load(f)
    print(f"-> Đã nạp lịch sử huấn luyện: {len(history.get('loss', []))} bước log")

  return model, history, config
