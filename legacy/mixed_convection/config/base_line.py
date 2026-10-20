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
    wandb.name = "Re-10_Ra-100_Da-0.001"
    wandb.tag = None

    # =========================================================================
    # 2. Physics Parameters (Môi trường Xốp & Truyền nhiệt)
    # =========================================================================
    config.physics = physics = ml_collections.ConfigDict()
    physics.H = 1.0
    physics.L = 1.0
    
    # Các thông số không thứ nguyên (Dimensionless numbers)
    physics.Re = 10.0      # Reynolds number
    physics.Pr = 1.0       # Prandtl number
    physics.Ra = 100.0     # Rayleigh number
    physics.Da = 0.01  # Darcy number
    physics.Je = 1.0       # Ergun/Forchheimer ratio parameter
    
    # Các thông số môi trường xốp
    physics.epsilon = 0.4  # Porosity
    physics.F_eps = 0.0  # Forchheimer coefficient
    
    # Các thông số nhiệt và trọng lực
    physics.g_beta = 0.1
    physics.delta_T = 1.0
    physics.T_ref = 0.5
    
    # =========================================================================
    # 3. Ablation Study Control Panel (Điều khiển Scale-PINN & Sifan Wang)
    # =========================================================================
    config.ablation = ablation = ml_collections.ConfigDict()
    ablation.use_sequential = False    # Bật/tắt số hạng (u - u_prev)
    ablation.use_laplacian = True     # Bật/tắt Laplacian smoothing
    ablation.tau_mode = "fixed"    # "adaptive" (Sifan Wang) hoặc "fixed" (Scale-PINN gốc)
    
    # Nếu tau_mode == "fixed", sẽ dùng 2 hằng số này
    ablation.fixed_tau = 0.5          
    ablation.fixed_tau_sc = 0.5       
    
    # Nếu tau_mode == "adaptive" và use_laplacian == True, dùng tỷ lệ CFL này
    ablation.cfl_ratio = 0.05          # Lực smoothing = adaptive_tau * gamma * cfl_ratio

    # =========================================================================
    # 4. Neural Network Architecture (NSFlowNet)
    # =========================================================================
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "nsflownet"
    arch.shared_layers = 4
    arch.branch_layers = 3
    arch.shared_dim = 128
    arch.branch_dim = 64
    arch.activation = "silu"
    arch.out_vars = ("u", "v", "p", "T")
    
    # Fourier Features (Để nguyên để tránh Spectral Bias)
    # arch.fourier_emb = ml_collections.ConfigDict(
    #     {"embed_scale": 10.0, "embed_dim": 256}
    # )
    arch.periodicity = ml_collections.ConfigDict({
        "period": (2.0 * math.pi / config.physics.L,),  # Tần số góc omega = 2*pi/L
        "axis": (0,),                                   # Chỉ áp dụng cho trục x (index 0)
        "trainable": (False,)                           # Cố định, không cho mạng học lại chu kỳ
    })
    arch.fourier_emb = None
    # TẮT periodicity cứng nếu bài toán của bạn là mixed convection (Dirichlet)
    #arch.periodicity = None
    # TẮT freq_factor để làm sạch kiến trúc, tránh nhiễu ablation study
    arch.freq_factor = None 

    # =========================================================================
    # 5. Optimizer (Giữ nguyên siêu bộ tăng tốc SOAP)
    # =========================================================================
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "soap"
    optim.lr_schedule = "exponential_decay"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.warmup_steps = 2000
    optim.staircase = False
    optim.schedule_free = True

    # =========================================================================
    # 6. Training Hyperparameters
    # =========================================================================
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 10000
    training.bs_eq = 1024
    training.bs_bc = 64

    # =========================================================================
    # 7. Global Loss Weighting (GradNorm của Sifan Wang)
    # =========================================================================
    config.loss_weighting = loss_weighting = ml_collections.ConfigDict()
    loss_weighting.strategy = "dynamic"  # Tự động cân bằng BC và PDE
    
    # Khởi tạo trọng số (eqn_0..3 tương ứng với u, v, p, T)
    loss_weighting.loss_weights = ml_collections.ConfigDict({
        "bc_y0_u": 1.0, "bc_y0_v": 1.0, "bc_y0_T": 1.0, 
        "bc_yH_u": 1.0, "bc_yH_v": 1.0, "bc_yH_T": 1.0,
        "eqn_0": 1.0, "eqn_1": 1.0, "eqn_2": 1.0, "eqn_3": 1.0
    })
    loss_weighting.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 100,  # Nên update trọng số mỗi 100 bước
    })
    loss_weighting.momentum = 0.9

    # =========================================================================
    # 8. Pseudo-time Stepping / Adaptive Tau
    # =========================================================================
    config.pseudo_time = pseudo_time = ml_collections.ConfigDict()
    
    # LƯU Ý QUAN TRỌNG: Ở Cấu trúc mới, pts_weights CHÍNH LÀ Adaptive Tau!
    # Ta BẬT nó lên để nuôi thuật toán Sifan Wang
    pseudo_time.enabled = True 
    pseudo_time.strategy = "dynamic" 
    
    # Khởi tạo Tau = 1.0 cho 4 phương trình (u, v, p, T)
    pseudo_time.pts_weights = ml_collections.ConfigDict(
        {"eqn_0": 1.0, "eqn_1": 1.0, "eqn_2": 1.0, "eqn_3": 1.0}
    )
    pseudo_time.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 100,  # Cập nhật Tau cùng lúc với Loss weights
    })
    pseudo_time.momentum = 0.9
    
    # Shrinkage: Giúp Tau nhỏ dần khi mô hình gần hội tụ (Cực kỳ mạnh mẽ)
    pseudo_time.shrink = shrink = ml_collections.ConfigDict()
    shrink.enabled = True
    shrink.start_log_drop = 2.0
    shrink.end_log_drop = 6.0
    shrink.min_factor = 0.1

    # =========================================================================
    # 9. Logging & Saving
    # =========================================================================
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_lr = True
    logging.log_losses = True
    logging.log_raw_losses = True
    logging.log_loss_weights = True
    logging.log_pts_weights = True
    logging.log_grads = False
    logging.log_nonlinearities = False

    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 1000
    saving.num_keep_ckpts = 10

    config.input_dim = 2
    config.seed = 42

    return config