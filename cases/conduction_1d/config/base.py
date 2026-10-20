import ml_collections
from numpy import diff


def get_config():
    config = ml_collections.ConfigDict()

    # =========================================================================
    # 1. Weights & Biases Logging
    # =========================================================================
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "1D-Conjugate-Heat-Conduction-PINN"
    wandb.name = "conduction_1d"
    wandb.tag = None

    # =========================================================================
    # 2. Physics Parameters (1D Conjugate Conduction - Gao et al. 2017)
    # =========================================================================
    config.physics = physics = ml_collections.ConfigDict()
    physics.L = 1.0
    physics.D = 0.5
    physics.k1 = 1.3 #5.0
    physics.k2 = 1.0
    physics.Th = 1.0
    physics.Tc = 0.0

    # =========================================================================
    # 3. Neural Network Architecture (1D Input -> 1D Output)
    # =========================================================================
    config.input_dim = 1  # BẮT BUỘC: 1 chiều không gian (x)
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "modifiedmlp"
    arch.num_layers = 4
    arch.hidden_dim = 64
    arch.out_dim = 1  # BẮT BUỘC: Chỉ có duy nhất 1 đầu ra (T)
    arch.activation = "silu"
    arch.periodicity = None

    # Fourier Feature Embedding (embed_dim cho bài toán 1D)
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 2.0, "embed_dim": 64}
    )

    # =========================================================================
    # 4. Optimizer (SOAP / Adam)
    # =========================================================================
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "adam"  # Khuyên dùng Adam cho bài 1D để tránh suy biến ma trận tiền điều kiện
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8

    # Lịch trình học
    optim.lr_schedule = "exponential_decay"
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.warmup_steps = 500
    optim.staircase = False
    optim.schedule_free = False

    # =========================================================================
    # 5. Training Hyperparameters & Collocation Sampling
    # =========================================================================
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 50000
    training.resume = False
    training.batch_size = 512
    training.random_sampling = True

    # =========================================================================
    # 6. Global Loss Weighting (KHÓA BẮT BUỘC PHẢI KHỚP CHÍNH XÁC VỚI 1D)
    # =========================================================================
    config.loss_weighting = loss_weighting = ml_collections.ConfigDict()
    loss_weighting.strategy = "dynamic"  # "dynamic"  hoặc "fixed"

    # CHỈ ĐỂ 2 KHÓA: "T_bc" và "T_res"
    loss_weighting.loss_weights = ml_collections.ConfigDict({
        "T_bc": 1.0,
        "T_res": 1.0,
    })
    loss_weighting.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 500,
    })
    loss_weighting.momentum = 0.9

    # =========================================================================
    # 7. Adaptive Pseudo-time Stepping (KHÓA KHỚP VỚI variables = ("T",))
    # =========================================================================
    config.pseudo_time = pseudo_time = ml_collections.ConfigDict()
    pseudo_time.enabled = True
    pseudo_time.strategy = "dynamic"  # "dynamic" hoặc "constant"

    # CHỈ ĐỂ DUY NHẤT KHÓA "T" (Trùng với self.variables = ("T",))
    pseudo_time.pts_weights = ml_collections.ConfigDict({
        "T": 1.0,
    })
    pseudo_time.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 500,
    })
    pseudo_time.momentum = 0.9

    # Lịch trình thu hẹp bước thời gian giả
    pseudo_time.shrink = shrink = ml_collections.ConfigDict()
    shrink.enabled = True
    shrink.start_log_drop = 2.0
    shrink.end_log_drop = 5.0
    shrink.min_factor = 0.1

    # =========================================================================
    # 8. Differential Scheme
    # =========================================================================
    config.diff = diff = ml_collections.ConfigDict()
    diff.scheme = "ffv"  # "ffv", "can", hoặc "ad"
    diff.h_stencil = 0.01
    diff.n_x = 101
    diff.use_continuous_mesh = False

    # =========================================================================
    # 9. Logging & Checkpoint Saving
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

    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = 3
    saving.ckpt_path = None  # gốc thư mục run; scripts/train.py đặt thành outputs/<case>

    config.seed = 42

    # Precision của phép nhân ma trận trên GPU. "highest" = FP32 đầy đủ; mặc định của
    # JAX trên GPU Ampere là TF32, làm ψ_max lệch ~1% (xem ROADMAP.md).
    config.jax_matmul_precision = "highest"

    return config