import os
import pickle
import time
import jax
import numpy as np
from tqdm.auto import tqdm

from jaxpi.checkpointing import create_checkpoint_manager, save_checkpoint, restore_checkpoint
from jaxpi.models import (
    create_arch,
    create_lr_schedule,
    create_optimizer,
    create_train_state,
)
from jaxpi.utils import create_update_scheduler
from porous_pinn import models


def sample_batches(samplers):
  """Generator lấy mẫu vô hạn cho dictionary các samplers."""
  if isinstance(samplers, dict):
    iterators = {key: iter(sampler) for key, sampler in samplers.items()}
    while True:
      yield {key: next(iterator) for key, iterator in iterators.items()}
  else:
    yield from iter(samplers)


def train_porous(config, samplers, outputs_dir=None):
  # 1. Thiết lập thư mục lưu checkpoint và history
  if outputs_dir is None:
    outputs_dir = os.path.join(os.getcwd(), "outputs")
  ckpt_dir = os.path.join(outputs_dir, config.wandb.name, "ckpt")
  history_dir = os.path.join(outputs_dir, config.wandb.name)
  os.makedirs(history_dir, exist_ok=True)
  history_path = os.path.join(history_dir, "history.pkl")

  ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_dir)

  # 2. Khởi tạo kiến trúc mạng, optimizer và trạng thái huấn luyện
  lr = create_lr_schedule(config.optim)
  tx = create_optimizer(config.optim, lr)
  arch = create_arch(config.arch)

  pretrained_params = None
  if config.training.get("transfer_learning", False):
    pretrained_path = config.training.get("pretrained_ckpt_dir", None)
    if pretrained_path and os.path.isdir(pretrained_path):
      print(f"--> Đang nạp pretrained weights từ: {pretrained_path}")
      # Tạo một manager tạm để đọc checkpoint cũ
      temp_mngr = create_checkpoint_manager(config.saving, pretrained_path)
      # Khởi tạo dummy state để nhận cấu trúc dữ liệu
      dummy_state = create_train_state(config, tx, arch)
      restored_state = restore_checkpoint(temp_mngr, dummy_state)
      pretrained_params = restored_state.params
      print("--> Nạp weights thành công! Bắt đầu fine-tuning từ step 0 với optimizer mới.")
    else:
      print("--> CẢNH BÁO: Không tìm thấy pretrained checkpoint, sẽ khởi tạo ngẫu nhiên.")

  state = create_train_state(config, tx, arch, params=pretrained_params)

  # 3. Khởi tạo mô hình (kế thừa ForwardBVP chuẩn)
  model = models.PorousNaturalConvection2D(config, lr, tx, arch, state)

  # Trạng thái ban đầu làm mốc tham chiếu cho Pseudo-Time Shrink Schedule
  init_state = model.state
  history = {"loss": []}

  # 4. Sử dụng scheduler chuẩn của jaxpi2
  loss_update = create_update_scheduler(**config.loss_weighting.update_schedule)
  pseudo_time_update = create_update_scheduler(
      **config.pseudo_time.update_schedule
  )
  freq_log = config.logging.get("log_every_steps", 100)

  batches = sample_batches(samplers)
  pbar = tqdm(range(config.training.max_steps), desc="Training PINN")

  print("Bắt đầu huấn luyện...")
  start_time = time.time()

  for step in pbar:
    batch = next(batches)

    # BƯỚC 1: Cập nhật gradient một bước
    model.state, loss, loss_dict = model.step(model.state, batch)

    # BƯỚC 2: Cập nhật thích ứng trọng số thời gian giả (Pseudo-Time Stepping)
    # LƯU Ý: Chỉ truyền mảng tọa độ nội miền batch["eqn"], KHÔNG truyền cả dict batch
    if config.pseudo_time.enabled and config.pseudo_time.strategy == "dynamic":
      if pseudo_time_update(step):
        res_batch = batch["eqn"] if isinstance(batch, dict) else batch
        model.state = model.update_pts_weights(
            model.state, init_state, res_batch
        )

    # BƯỚC 3: Cập nhật trọng số cân bằng gradient (Loss Weighting)
    if config.loss_weighting.strategy == "dynamic":
      if loss_update(step):
        model.state = model.update_loss_weights(model.state, batch)

    # BƯỚC 4: Ghi nhận số liệu (Logging) - Chỉ thực hiện trên Host chính
    if jax.process_index() == 0:
      if step % freq_log == 0:
        val_loss = np.asarray(loss).item()
        history["loss"].append(val_loss)

        for k, v in loss_dict.items():
          if k not in history:
            history[k] = []
          history[k].append(np.asarray(v).item())

        if (
            hasattr(model.state, "pts_weights")
            and model.state.pts_weights is not None
        ):
          for k, v in model.state.pts_weights.items():
            hist_key = f"tau_{k}"
            if hist_key not in history:
              history[hist_key] = []
            history[hist_key].append(np.asarray(v).item())

        pbar.set_postfix({"Loss": f"{val_loss:.4e}"})

        # In thông số chi tiết theo định kỳ
        if step > 0 and step % (freq_log * 5) == 0:
          elapsed = time.time() - start_time
          details_str = " | ".join(
              [f"{k}: {np.asarray(v).item():.3e}" for k, v in loss_dict.items()]
          )
          pts_str = ""
          if (
              hasattr(model.state, "pts_weights")
              and model.state.pts_weights is not None
          ):
            pts_str = " | " + " | ".join([
                f"tau_{k}: {np.asarray(v).item():.2f}"
                for k, v in model.state.pts_weights.items()
            ])

          print(
              f"Step {step:05d} | Loss: {val_loss:.4e} | {details_str}{pts_str}"
              f" | Time/Log: {elapsed:.2f}s"
          )
          start_time = time.time()

      # Lưu checkpoint định kỳ
      if ckpt_mngr is not None and config.saving.save_every_steps > 0:
        if (
            (step + 1) % config.saving.save_every_steps == 0
            or (step + 1) == config.training.max_steps
        ):
          save_checkpoint(ckpt_mngr, model.state)

  # Lưu trữ toàn bộ lịch sử loss sau khi hoàn thành
  with open(history_path, "wb") as f:
    pickle.dump(history, f)

  if (
      ckpt_mngr is not None
      and config.saving.save_every_steps > 0
      and jax.process_index() == 0
  ):
    save_checkpoint(ckpt_mngr, model.state)
    ckpt_mngr.wait_until_finished()

  print("Hoàn thành huấn luyện.")
  return model, history