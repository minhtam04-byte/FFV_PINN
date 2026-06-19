import numpy as np 
from matplotlib import pyplot as plt
import jax 
from jax import numpy as jnp 

def visualize_samples(samplers):
    batch_f1 = np.array(next(samplers["eqn"]))
    batch_int1 = np.array(next(samplers["bc_y0"]))
    batch_int2 = np.array(next(samplers["bc_yH"]))

    plt.figure(figsize=(10, 8))
    plt.scatter(batch_f1[:, 0], batch_f1[:, 1], c='blue', s=5, alpha=0.5, label='PDE Domain')
    plt.scatter(batch_int1[:, 0], batch_int1[:, 1], c='cyan', marker='x', s=20, label='Bottom BC (y=0)')
    plt.scatter(batch_int2[:, 0], batch_int2[:, 1], c='magenta', marker='x', s=20, label='Top BC (y=H)')
    
    plt.title("Phân bố điểm lấy mẫu")
    plt.legend()
    plt.show()

def evaluate_and_plot(model, state, L=1.0, H=1.0, nx=100, ny=100):
    print("Đang tiến hành đánh giá mô hình...")
    
    # 1. TRÍCH XUẤT PARAMS
    single_params = state.params
    
    # 2. TẠO LƯỚI TỌA ĐỘ
    x = jnp.linspace(0, L, nx)
    y = jnp.linspace(0, H, ny)
    X, Y = jnp.meshgrid(x, y)
    
    X_flat = X.flatten().reshape(-1, 1)
    Y_flat = Y.flatten().reshape(-1, 1)
    
    # 3. DỰ ĐOÁN
    @jax.jit
    def predict(params, x_in, y_in):
        return jax.vmap(model.u_net, in_axes=(None, 0, 0))(params, x_in, y_in)
    
    u_pred, v_pred, p_pred, T_pred = predict(single_params, X_flat, Y_flat)
    
    # 4. NGHIỆM GIẢI TÍCH (Gọi trực tiếp từ model - Rất tối ưu!)
    u_true, v_true, T_true = model.analytical_solution(Y_flat)

    # 5. TÍNH SAI SỐ RELATIVE L2 
    def relative_l2(pred, true):
        p_flat = pred.flatten()
        t_flat = true.flatten()
        return jnp.linalg.norm(p_flat - t_flat) / (jnp.linalg.norm(t_flat) + 1e-8)
    
    err_u = relative_l2(u_pred, u_true)
    err_v = relative_l2(v_pred, v_true)
    err_T = relative_l2(T_pred, T_true)
    
    print("-" * 30)
    print("------- RELATIVE L2 ERROR -------")
    print(f"Error u: {err_u:.4e}")
    print(f"Error v: {err_v:.4e}")
    print(f"Error T: {err_T:.4e}")
    print("-" * 30)
    
    # 6. TRỰC QUAN HÓA
    plot_data = [
        ("U_x", u_true, u_pred),
        ("U_y", v_true, v_pred),
        ("Temperature", T_true, T_pred)
    ]
    
    X_np = np.array(X)
    Y_np = np.array(Y)
    
    for name, true_flat, pred_flat in plot_data:
        True_grid = np.array(true_flat).reshape(ny, nx)
        Pred_grid = np.array(pred_flat).reshape(ny, nx)
        Err_grid = np.abs(Pred_grid - True_grid)
        
        v_min = min(True_grid.min(), Pred_grid.min())
        v_max = max(True_grid.max(), Pred_grid.max())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot True
        c0 = axes[0].contourf(X_np, Y_np, True_grid, levels=50, cmap='jet', vmin=v_min, vmax=v_max)
        axes[0].set_title(f"{name} (Giải tích)")
        fig.colorbar(c0, ax=axes[0])
        
        # Plot Pred
        c1 = axes[1].contourf(X_np, Y_np, Pred_grid, levels=50, cmap='jet', vmin=v_min, vmax=v_max)
        axes[1].set_title(f"{name} (PINN Dự đoán)")
        fig.colorbar(c1, ax=axes[1])
        
        # Plot Error
        c2 = axes[2].contourf(X_np, Y_np, Err_grid, levels=50, cmap='magma')
        axes[2].set_title("Sai số tuyệt đối |Pred - True|")
        fig.colorbar(c2, ax=axes[2])
        
        plt.tight_layout()
        plt.show()

def plot_ablation_histories(results, log_freq=100):
    fig, axes = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':', '-']
    
    for i, (exp_name, hist) in enumerate(results.items()):
        color = colors[i % len(colors)]
        style = line_styles[i % len(line_styles)]
        
        steps = np.arange(len(hist['loss'])) * log_freq

        axes.plot(steps, hist['loss'], label=exp_name, 
                  color=color, linestyle=style, linewidth=2.5, alpha=0.8)

    axes.set_title("Total Loss", fontsize=14, fontweight='bold')
    axes.set_xlabel("Steps", fontsize=12)
    axes.set_ylabel("Loss", fontsize=12)
    axes.set_yscale('log')
    axes.grid(True, which="both", ls="--", alpha=0.5)
    axes.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_pts_weights_histories(results, log_freq=100):
    pts_keys = set()
    for hist in results.values():
        for k in hist.keys():
            if k.startswith('pts_weight_'):
                pts_keys.add(k)
                
    pts_keys = sorted(list(pts_keys))
    n_keys = len(pts_keys)
    
    if n_keys == 0:
        print("Không tìm thấy dữ liệu pts_weights nào trong history.")
        return

    fig, axes = plt.subplots(1, n_keys, figsize=(5 * n_keys, 5), dpi=100)
    
    if n_keys == 1:
        axes = [axes]
        
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':', '-']
    
    for i, key in enumerate(pts_keys):
        ax = axes[i]
        
        for j, (exp_name, hist) in enumerate(results.items()):
            if key in hist:
                color = colors[j % len(colors)]
                style = line_styles[j % len(line_styles)]
                
                steps = np.arange(len(hist[key])) * log_freq
                
                ax.plot(steps, hist[key], label=exp_name, 
                        color=color, linestyle=style, linewidth=2.0, alpha=0.8)

        ax.set_title(f"Evolution of {key}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Steps", fontsize=11)
        ax.set_ylabel("Weight Value", fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.show()
