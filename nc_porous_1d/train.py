import os
import time
import pickle
import jax
import jax.numpy as jnp
from jax import grad
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from jaxpi.checkpointing import create_checkpoint_manager, save_checkpoint
from jaxpi.models import (
    create_arch,
    create_lr_schedule,
    create_optimizer,
    create_train_state,
)
from jaxpi.utils import create_update_scheduler
from nc_porous_1d import models


def extract_last_bias(params):
    """Trích xuất giá trị bias của lớp Dense cuối cùng từ PyTree params."""
    # Tìm key của lớp Dense cuối cùng
    keys = list(params.keys())
    for k in reversed(keys):
        if "bias" in params[k]:
            return params[k]["bias"]
    return None


def plot_diagnostic_history(diag_history, save_path):
    """Vẽ và lưu biểu đồ phân tích cơ chế mất ổn định / hội tụ."""
    steps = diag_history["step"]
    if len(steps) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    # 1. Đồ thị Output Network Range
    axes[0, 0].plot(steps, diag_history["mean_T"], "b-", linewidth=2, label="mean(T)")
    axes[0, 0].fill_between(
        steps,
        diag_history["min_T"],
        diag_history["max_T"],
        color="b",
        alpha=0.15,
        label="[min(T), max(T)] range",
    )
    axes[0, 0].axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Physical T_h (1.0)")
    axes[0, 0].axhline(y=0.0, color="k", linestyle=":", alpha=0.5, label="Physical T_c (0.0)")
    axes[0, 0].set_title("Network Output Field T(x) Stability", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("T value")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="best", fontsize=9)

    # 2. Đồ thị Last Layer Bias
    axes[0, 1].plot(steps, diag_history["last_bias"], "r-", linewidth=2, label=r"$b_{last}$")
    axes[0, 1].set_title("Evolution of Last Layer Bias", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Bias value")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="best", fontsize=9)

    # 3. Đồ thị Gradient Norm của Bias
    axes[1, 0].plot(steps, diag_history["grad_norm_bias"], "m-", linewidth=2, label=r"$\|\nabla_b \mathcal{L}\|$")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(r"Gradient Norm of Last Bias (log scale)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel(r"$\|\nabla_b \mathcal{L}\|$")
    axes[1, 0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="best", fontsize=9)

    # 4. Đồ thị Pseudo-Time Weight w
    axes[1, 1].plot(steps, diag_history["w_pts"], "g-", linewidth=2, label=r"PTS weight $w = 1/\tau$")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Pseudo-Time Weight w (log scale)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel(r"$w$")
    axes[1, 1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"--> Đã lưu biểu đồ phân tích chẩn đoán tại: {save_path}")


def train_1d(config, sampler):
    ckpt_dir = os.path.join(os.getcwd(), "outputs", config.wandb.name, "ckpt")
    history_dir = os.path.join(os.getcwd(), "outputs", config.wandb.name)
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "history.pkl")
    diag_plot_path = os.path.join(history_dir, "diagnostic_stability.png")

    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_dir)

    # 1. Khởi tạo mô hình
    lr = create_lr_schedule(config.optim)
    tx = create_optimizer(config.optim, lr)
    arch = create_arch(config.arch)
    state = create_train_state(config, tx, arch)

    model = models.ConjugateHeatConduction1D(config, lr, tx, arch, state)
    init_state = model.state

    history = {"loss": [], "l2_error": []}
    
    # Bộ lưu trữ chẩn đoán chi tiết (Diagnostic Tracker)
    diag_history = {
        "step": [],
        "mean_T": [],
        "min_T": [],
        "max_T": [],
        "last_bias": [],
        "grad_norm_bias": [],
        "w_pts": [],
    }

    loss_update = create_update_scheduler(**config.loss_weighting.update_schedule)
    pseudo_time_update = create_update_scheduler(**config.pseudo_time.update_schedule)
    freq_log = config.logging.get("log_every_steps", 100)
    freq_diag = 20  # Lấy mẫu chẩn đoán thường xuyên hơn để bắt kịp pha trước khi NaN

    # Lưới điểm kiểm chứng
    x_test = jnp.linspace(0.0, config.physics.L, 200)

    # Hàm tính gradient riêng của loss theo tham số (để đo gradient của bias)
    @jax.jit
    def compute_loss_grads(params, state_obj, b_data):
        return grad(lambda p: model.loss(p, state_obj, b_data)[0])(params)

    pbar = tqdm(range(config.training.max_steps), desc="Training 1D PINN")
    start_time = time.time()

    for step in pbar:
        batch = next(sampler)

        # Cập nhật gradient một bước
        model.state, loss, loss_dict = model.step(model.state, batch)

        # Cập nhật Adaptive Pseudo-Time Stepping
        if config.pseudo_time.enabled and config.pseudo_time.strategy == "dynamic":
            if pseudo_time_update(step):
                model.state = model.update_pts_weights(model.state, init_state, batch)

        # Cập nhật Dynamic Loss Weighting
        if config.loss_weighting.strategy == "dynamic":
            if loss_update(step):
                model.state = model.update_loss_weights(model.state, batch)

        # --- GHI NHẬN CHẨN ĐOÁN (DIAGNOSTIC TRACKING) ---
        if step % freq_diag == 0:
            val_loss = np.asarray(loss).item()

            # Đánh giá phân bố output T trên x_test
            T_eval = np.asarray(model.T_pred_fn(model.state.params, x_test))
            mean_T = float(np.mean(T_eval))
            min_T = float(np.min(T_eval))
            max_T = float(np.max(T_eval))

            # Lấy giá trị bias lớp cuối cùng
            bias_arr = extract_last_bias(model.state.params)
            last_bias_val = float(np.asarray(bias_arr).flatten()[0]) if bias_arr is not None else 0.0

            # Tính chuẩn gradient của bias lớp cuối
            p_grads = compute_loss_grads(model.state.params, model.state, batch)
            bias_grad = extract_last_bias(p_grads)
            norm_b_grad = float(np.linalg.norm(np.asarray(bias_grad))) if bias_grad is not None else 0.0

            w_pts_val = float(np.asarray(model.state.pts_weights.get("T", 1.0)).item())

            diag_history["step"].append(step)
            diag_history["mean_T"].append(mean_T)
            diag_history["min_T"].append(min_T)
            diag_history["max_T"].append(max_T)
            diag_history["last_bias"].append(last_bias_val)
            diag_history["grad_norm_bias"].append(norm_b_grad)
            diag_history["w_pts"].append(w_pts_val)

            # KIỂM TRA PHÁT HIỆN SỚM NAN HOẶC BÙNG NỔ BIÊN ĐỘ
            if np.isnan(val_loss) or np.isinf(val_loss) or abs(mean_T) > 1e4:
                print(f"\n[CẢNH BÁO] Phát hiện mất ổn định/NaN tại step {step}!")
                print(f"Chi tiết: Loss={val_loss:.3e} | mean(T)={mean_T:.3e} | bias={last_bias_val:.3e} | ||grad_b||={norm_b_grad:.3e}")
                # Vẽ biểu đồ trạng thái trước khi thoát
                plot_diagnostic_history(diag_history, diag_plot_path)
                return model, history

        # --- LOG CHU KỲ BÌNH THƯỜNG ---
        if step % freq_log == 0:
            val_loss = np.asarray(loss).item()
            err_dict = model.compute_l2_error(model.state.params, x_test)
            l2_err = np.asarray(err_dict["T_l2_error"]).item()

            history["loss"].append(val_loss)
            history["l2_error"].append(l2_err)

            for k, v in loss_dict.items():
                if k not in history:
                    history[k] = []
                history[k].append(np.asarray(v).item())

            if hasattr(model.state, "pts_weights") and model.state.pts_weights is not None:
                for k, v in model.state.pts_weights.items():
                    hist_key = f"tau_{k}"
                    if hist_key not in history:
                        history[hist_key] = []
                    history[hist_key].append(np.asarray(v).item())

            pbar.set_postfix({"Loss": f"{val_loss:.3e}", "L2 Err": f"{l2_err:.3e}"})

    # Lưu history
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    # Xuất đồ thị chẩn đoán toàn bộ quá trình
    plot_diagnostic_history(diag_history, diag_plot_path)

    # Trực quan hóa nghiệm so với giải tích
    T_pred = model.T_pred_fn(model.state.params, x_test)
    T_exact = model.compute_analytical_solution(x_test)

    plt.figure(figsize=(7, 5))
    plt.plot(x_test, T_exact, "k-", label="Analytical")
    plt.plot(x_test, T_pred, "r--", label=f"Predicted ({config.diff.scheme.upper()})")
    plt.axvline(x=config.physics.D, color="gray", linestyle=":", label="Interface")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title(f"1D Conjugate Conduction (L2 Error: {history['l2_error'][-1]:.3e})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(history_dir, "prediction_1d.png"), dpi=200)
    plt.close()

    print(f"Hoàn thành! Sai số Relative L2 cuối cùng: {history['l2_error'][-1]:.4e}")
    return model, history