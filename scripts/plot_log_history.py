import re
import matplotlib.pyplot as plt

# Dán toàn bộ chuỗi log vào biến log_data hoặc đọc từ file log.txt
log_data = """
Step 00500 | Loss: 4.7796e-02 | T_dirichlet_bc: 6.067e-05 | T_neumann_bc: 1.797e-05 | T_res: 1.717e-03 | p_res: 2.370e-03 | u_bc: 6.841e-05 | u_res: 1.314e-02 | v_bc: 1.391e-05 | v_res: 1.480e-02 | tau_T: 4.12 | tau_p: 8.79 | tau_u: 6.55 | tau_v: 7.54 | Time/Log: 76.30s
Step 01000 | Loss: 2.4066e-02 | T_dirichlet_bc: 7.280e-06 | T_neumann_bc: 7.450e-06 | T_res: 1.106e-03 | p_res: 1.385e-03 | u_bc: 2.131e-05 | u_res: 4.939e-03 | v_bc: 4.664e-06 | v_res: 8.678e-03 | tau_T: 4.12 | tau_p: 8.79 | tau_u: 6.55 | tau_v: 7.54 | Time/Log: 6.95s
...
"""  # Thay thế bằng log thực tế hoặc đọc file

# 1. Trích xuất dữ liệu bằng Regular Expression
pattern = re.compile(
    r"Step\s+(\d+)\s+\|\s+Loss:\s+([\d\.eE+-]+).*?"
    r"T_res:\s+([\d\.eE+-]+)\s+\|\s+p_res:\s+([\d\.eE+-]+).*?"
    r"u_res:\s+([\d\.eE+-]+).*?v_res:\s+([\d\.eE+-]+).*?"
    r"tau_T:\s+([\d\.]+)\s+\|\s+tau_p:\s+([\d\.]+)\s+\|\s+tau_u:\s+([\d\.]+)\s+\|\s+tau_v:\s+([\d\.]+)"
)

steps, total_loss = [], []
u_res, v_res, p_res, T_res = [], [], [], []
tau_u, tau_v, tau_p, tau_T = [], [], [], []

for line in log_data.strip().split("\n"):
  match = pattern.search(line)
  if match:
    steps.append(int(match.group(1)))
    total_loss.append(float(match.group(2)))
    T_res.append(float(match.group(3)))
    p_res.append(float(match.group(4)))
    u_res.append(float(match.group(5)))
    v_res.append(float(match.group(6)))
    tau_T.append(float(match.group(7)))
    tau_p.append(float(match.group(8)))
    tau_u.append(float(match.group(9)))
    tau_v.append(float(match.group(10)))

# 2. Vẽ đồ thị so sánh theo phong cách Wang et al. (2026)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=120)

# Subplot 1: Lịch sử hội tụ của Loss và các thành phần thặng dư PDE
axes[0].plot(
    steps, total_loss, "k-", label="Total Loss", linewidth=2.2, alpha=0.9
)
axes[0].plot(steps, u_res, label=r"Momentum $u$ ($u_{res}$)", color="#1f77b4")
axes[0].plot(steps, v_res, label=r"Momentum $v$ ($v_{res}$)", color="#ff7f0e")
axes[0].plot(
    steps, p_res, label=r"Continuity ($p_{res}$)", color="#2ca02c", linestyle="--"
)
axes[0].plot(
    steps, T_res, label=r"Energy ($T_{res}$)", color="#d62728", linestyle="-."
)

axes[0].set_yscale("log")
axes[0].set_xlabel("Training Steps", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Residual Loss (log scale)", fontsize=11, fontweight="bold")
axes[0].set_title(
    "PDE Residuals & Total Loss", fontsize=12, fontweight="bold"
)
axes[0].grid(True, which="both", ls="--", alpha=0.4)
axes[0].legend(fontsize=9, loc="upper right")

# Đánh dấu vùng đáy tối ưu
axes[0].axvspan(
    25000, 42000, color="gray", alpha=0.15, label="Optimal convergence zone"
)

# Subplot 2: Sự tiến hóa của các trọng số thời gian giả 1/tau (pts_weights)
axes[1].plot(steps, tau_u, label=r"$1/\tau_u$", color="#1f77b4", linewidth=1.8)
axes[1].plot(steps, tau_v, label=r"$1/\tau_v$", color="#ff7f0e", linewidth=1.8)
axes[1].plot(steps, tau_T, label=r"$1/\tau_T$", color="#d62728", linewidth=1.8)
axes[1].plot(
    steps, tau_p, label=r"$1/\tau_p$", color="#2ca02c", linewidth=1.8, linestyle="--"
)

# Đường tham chiếu ngưỡng cắt đề xuất 12.0
axes[1].axhline(
    y=12.0,
    color="black",
    linestyle=":",
    linewidth=1.5,
    label=r"Suggested upper clip ($12.0$)",
)

axes[1].set_xlabel("Training Steps", fontsize=11, fontweight="bold")
axes[1].set_ylabel(
    r"Pseudo-time step size $1/\tau$", fontsize=11, fontweight="bold"
)
axes[1].set_title(
    r"Evolution of Adaptive Step Sizes ($1/\tau$)",
    fontsize=12,
    fontweight="bold",
)
axes[1].grid(True, ls="--", alpha=0.4)
axes[1].legend(fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig("loss_and_tau_history.png", dpi=300, bbox_inches="tight")
plt.close()  # Giải phóng bộ nhớ RAM/GPU sau khi lưu
print("--> Đã lưu biểu đồ thành công tại: loss_and_tau_history.png")