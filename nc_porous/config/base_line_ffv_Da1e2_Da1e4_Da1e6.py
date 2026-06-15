import ml_collections
import math

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # =========================================================================
    # 1. Weights & Biases Logging
    # =========================================================================
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Porous-Heat-ScalePINN"
    wandb.name = "NCPorous_Da1e2_Da1e4_Da1e6" # Đổi tên cho đúng bài toán
    wandb.tag = None

    # =========================================================================
    # 2. Physics Parameters (Natural Convection - Table 2 Highest Ra)bs_eqn
    # =========================================================================
    config.physics = physics = ml_collections.ConfigDict()
    physics.H = 1.0
    physics.L = 1.0
    
    # Các thông số không thứ nguyên (Theo Case 1 - Table 4)

    # Các thông số không thứ nguyên (Dimensionless numbers)
    physics.Re = 1.0       # Không dùng cho pure natural convection, đặt 1.0 để tránh lỗi chia 0
    physics.Pr = 1.0      # Từ Table 2 (Không khí)
    physics.Ra = 1e6       # Từ Table 2 (Case lớn nhất)
    physics.Da = 1e6       # Từ Table 2 (Tiệm cận pure fluid)
    physics.Je = 1.0       # Từ cấu hình chung Section 4
    
    # Các thông số môi trường xốp
    physics.epsilon = 0.9999 # Từ Table 2 (Tiệm cận 1.0)
    physics.F_eps = 0.0      # Bỏ qua lực Forchheimer phi tuyến
    #physics.alpha_m_porous = 1.362

    # Các thông số nhiệt và trọng lực
    physics.g_beta = 0.1   # Từ cấu hình chung của bài báo
    physics.delta_T = 1.0  # T_h - T_c = 1.0 - 0.0
    physics.T_ref = 0.5    # Nhiệt độ trung bình

    # Thông số cho FVM Stencil (SIMPLE-PINN)
    config.fvm_setup = fvm_setup = ml_collections.ConfigDict()
    fvm_setup.h_stencil = 0.005
    fvm_setup.alpha_u = 0.7 
    fvm_setup.alpha_p = 0.3
    fvm_setup.mode = True 

    # [THAY ĐỔI] Thông số cho Immersed Porous Body (Lớp xốp ở giữa)
    config.ibm = ibm = ml_collections.ConfigDict()
    ibm.mode = False
    ibm.eta_diffuse = 120.0    # Giữ nguyên độ dốc để ranh giới đủ sắc nét
    ibm.layer_x_center = 0.5  # Đặt lớp xốp ở chính giữa (x = 0.5)
    ibm.layer_width = 1.0 / 3.0  # Bề dày lớp xốp chiếm 1/3 chiều rộng (theo S/H = 1/3)

    # =========================================================================
    # 3. Ablation Study Control Panel (Điều khiển Scale-PINN & Sifan Wang)
    # =========================================================================
    config.ablation = ablation = ml_collections.ConfigDict()
    ablation.use_sequential = True    # BẬT để dùng Scale-PINN
    ablation.use_laplacian = False     # BẬT để dùng Laplacian smoothing
    ablation.tau_mode = "adaptive"       # Scale-PINN gốc dùng fixed tau
    
    # Thiết lập khuyên dùng từ bài báo Scale-PINN cho dòng đối lưu nhiệt
    ablation.fixed_tau_sc = 0.1       # Pseudo-time step nhỏ để tránh Local Minima
    ablation.fixed_tau = 1.5          # Tau_alpha lớn để làm mượt (Smoothing) nhẹ nhàng
    
    ablation.cfl_ratio = 0.08        # Dùng nếu bật tau_mode = 'adaptive'

    # =========================================================================
    # 4. Neural Network Architecture (NSFlowNet)
    # =========================================================================
    # config.arch = arch = ml_collections.ConfigDict()
    # arch.arch_name = "cavity"
    # arch.shared_layers = 4
    # arch.branch_layers = 3
    # arch.shared_dim = 256
    # arch.branch_dim = 128
    # arch.T_layers = 4       # Chỉ dùng 3 lớp ẩn cho Nhiệt độ
    # arch.T_dim = 128         # Kích thước lớp ẩn giảm một nửa
    # arch.activation = "silu"
    # arch.out_vars = ("u", "v", "p", "T")
    # arch.fourier_emb = ml_collections.ConfigDict(
    #    {"embed_scale": 10.0, "embed_dim": 256}
    # )
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "nsflownet"
    arch.shared_layers = 4
    arch.branch_layers = 3
    arch.shared_dim = 128
    arch.branch_dim = 64
    arch.activation = "silu"
    arch.out_vars = ("u", "v", "p", "T")
    # TẮT periodicity cứng vì bài toán là hốc kín (Cavity) với biên Dirichlet/Neumann
    arch.periodicity = None
    # arch.fourier_emb = None
    # #arch.nonlinearity = 0.2
    arch.freq_factor = 4.0

    # =========================================================================
    # 5. Optimizer (Giữ nguyên siêu bộ tăng tốc SOAP)
    # =========================================================================
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "soap"
    
    # Các thông số chuẩn của SOAP theo bài báo
    optim.beta1 = 0.9      # Giữ nguyên
    optim.beta2 = 0.999    # Giữ nguyên
    optim.eps = 1e-8       # Giữ nguyên
    
    # Lịch trình học (Learning Rate Schedule)
    optim.lr_schedule = "exponential_decay"
    optim.learning_rate = 1e-3     # Đẩy lên 1e-3 vì PTS và SOAP đã gánh phần ổn định
    optim.decay_rate = 0.9
    optim.decay_steps = 2000       # Giảm xuống 2000 theo chuẩn bài báo để LR hạ nhanh hơn ở giai đoạn sau
    optim.warmup_steps = 2000      # 2000 step đầu warmup tuyến tính từ 0 lên 1e-3 là bắt buộc
    optim.staircase = True        # Giữ False để decay mượt mà
    
    # Tắt Schedule Free để không phá hỏng cấu trúc ma trận của SOAP
    optim.schedule_free = False

    # =========================================================================
    # 6. Training Hyperparameters
    # =========================================================================
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 150000   # Ra=1e6 cần nhiều bước hơn để hình thành lớp biên sắc nét
    training.bs_eq = 4096   
    training.bs_bc = 512       

    # =========================================================================
    # 7. Global Loss Weighting (GradNorm)
    # =========================================================================
    config.loss_weighting = loss_weighting = ml_collections.ConfigDict()
    loss_weighting.strategy = "dynamic" 

    loss_weighting.loss_weights = ml_collections.ConfigDict({
        "eqn_0": 1.0, "eqn_1": 1.0, "eqn_2": 1.0, "eqn_3": 1.0,
        "bc_uv": 10.0,
        "bc_T_dirichlet": 10.0,
        "bc_T_neumann": 10.0,
    })
    loss_weighting.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 100, 
    })
    loss_weighting.momentum = 0.9

    # =========================================================================
    # 8. Pseudo-time Stepping / Adaptive Tau
    # =========================================================================
    config.pseudo_time = pseudo_time = ml_collections.ConfigDict()
    pseudo_time.enabled = False 
    pseudo_time.strategy = "dynamic" 
    pseudo_time.pts_weights = ml_collections.ConfigDict(
        {"eqn_0": 1.0, "eqn_1": 1.0, "eqn_2":1.0, "eqn_3": 1.0}
    )
    pseudo_time.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 500, 
    })
    pseudo_time.momentum = 0.9
    
    pseudo_time.shrink = shrink = ml_collections.ConfigDict()
    shrink.enabled = True
    shrink.start_log_drop = 2.0
    shrink.end_log_drop = 6.0
    shrink.min_factor = 0.1

    # =========================================================================
    # 9. Logging & Saving
    # =========================================================================
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 10 
    logging.log_errors = True
    logging.log_lr = True
    logging.log_losses = True
    logging.log_raw_losses = True
    logging.log_loss_weights = True
    logging.log_pts_weights = True
    logging.log_grads = False
    logging.log_nonlinearities = False

    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 15000
    saving.num_keep_ckpts = 10

    config.arch_input_dim = 2 # one more for HarBC embedding
    config.input_dim = 2
    config.seed = 42

    return config