import ml_collections


def get_config():
  """Get the default hyperparameter configuration for Porous Natural Convection."""
  config = ml_collections.ConfigDict()

  # =========================================================================
  # 1. Weights & Biases Logging
  # =========================================================================
  config.wandb = wandb = ml_collections.ConfigDict()
  wandb.project = "porous_layer-natural-convection-PINN"
  wandb.name = "PorousNC_layer"
  wandb.tag = None

  # =========================================================================
  # 2. Physics Parameters (Porous Natural Convection)
  # =========================================================================
  config.physics = physics = ml_collections.ConfigDict()
  physics.H = 1.0
  physics.L = 1.0

  # Các số không thứ nguyên
  physics.Pr = 6.44
  physics.Ra = 3.7e6
  physics.Da = 1.534e-5       # Cập nhật theo Table 4
  physics.epsilon = 0.4
  physics.F_eps = 0.554
  physics.lambda_k = 37.47    # Cập nhật alpha_m = 37.47 (Table 4)
  physics.T_ref = 0.5
  physics.porous_x_start = 1.0 / 3.0
  physics.porous_x_end = 2.0 / 3.0
  

  # Cấu hình Immersed Boundary Method (IBM) cho lớp xốp cục bộ
  config.ibm = ibm = ml_collections.ConfigDict()
  ibm.mode = False  # Đặt True nếu có phân bố vùng xốp cục bộ
  ibm.eta_diffuse = 30.0
  ibm.layer_x_center = 0.5
  ibm.layer_width = 1.0 / 3.0

  # =========================================================================
  # 3. Neural Network Architecture
  # =========================================================================
  config.arch = arch = ml_collections.ConfigDict()
  # Các kiến trúc hỗ trợ chuẩn trong jaxpi2: "modifiedmlp", "piratenet", "mlp"
  arch.arch_name = "modifiedmlp"
  arch.num_layers = 4
  arch.hidden_dim = 128
  arch.out_dim = 4  # BẮT BUỘC: 4 biến đầu ra (u, v, p, T)
  arch.activation = "silu"
  arch.periodicity = None  # Không gian kín có vách, không dùng periodic boundary

  # Fourier Feature Embedding
  arch.fourier_emb = ml_collections.ConfigDict(
      {"embed_scale": 2.0, "embed_dim": 128}
  )

  # =========================================================================
  # 4. Optimizer (SOAP / Adam)
  # =========================================================================
  config.optim = optim = ml_collections.ConfigDict()
  optim.optimizer = "soap"  # "soap" hoặc "adam"
  optim.beta1 = 0.9
  optim.beta2 = 0.999
  optim.eps = 1e-8

  # Lịch trình học (Learning Rate Schedule)
  optim.lr_schedule = "exponential_decay"
  optim.learning_rate = 1e-3
  optim.decay_rate = 0.9
  optim.decay_steps = 2000
  optim.warmup_steps = 2000
  optim.staircase = False
  optim.schedule_free = False

  # =========================================================================
  # 5. Training Hyperparameters & Collocation Sampling
  # =========================================================================
  config.training = training = ml_collections.ConfigDict()
  training.max_steps = 100000
  training.resume = False
  training.batch_size = 4096  # Kích thước batch collocation nội miền
  training.random_sampling = True

  # =========================================================================
  # 6. Global Loss Weighting (GradNorm Balancing)
  # =========================================================================
  config.loss_weighting = loss_weighting = ml_collections.ConfigDict()
  loss_weighting.strategy = "dynamic"  # "dynamic" (GradNorm) hoặc "constant"

  # Khởi tạo trọng số khớp chính xác với các khóa trong hàm losses()
  loss_weighting.loss_weights = ml_collections.ConfigDict({
      "u_bc": 10.0,
      "v_bc": 10.0,
      "T_dirichlet_bc": 10.0,
      "T_neumann_bc": 1.0,
      "u_res": 1.0,
      "v_res": 1.0,
      "p_res": 1.0,
      "T_res": 1.0,
  })
  loss_weighting.update_schedule = ml_collections.ConfigDict({
      "start": 100,
      "every": 1000,
  })
  loss_weighting.momentum = 0.9

  # =========================================================================
  # 7. Adaptive Pseudo-time Stepping (Barzilai-Borwein & Shrinkage)
  # =========================================================================
  config.pseudo_time = pseudo_time = ml_collections.ConfigDict()
  pseudo_time.enabled = True
  pseudo_time.strategy = (
      "dynamic"  # "dynamic" để bật tự thích ứng Jacobian (Barzilai-Borwein)
  )

  # Khóa BẮT BUỘC phải trùng khớp hoàn toàn với self.variables = ("u", "v", "p", "T")
  pseudo_time.pts_weights = ml_collections.ConfigDict(
      {"u": 1.0, "v": 1.0, "p": 1.0, "T": 1.0}
  )
  pseudo_time.update_schedule = ml_collections.ConfigDict({
      "start": 100,
      "every": 1000,  # Tần suất cập nhật tau thích ứng
  })
  pseudo_time.momentum = 0.9

  # Lịch trình thu hẹp bước thời gian giả theo mức giảm log-loss
  pseudo_time.shrink = shrink = ml_collections.ConfigDict()
  shrink.enabled = True
  shrink.start_log_drop = 2.0
  shrink.end_log_drop = 5.0
  shrink.min_factor = 0.1

  # =========================================================================
  # 8. Logging & Checkpoint Saving
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
  saving.save_every_steps = 10000
  saving.num_keep_ckpts = 5

  # Khởi tạo kích thước đầu vào (tọa độ không gian 2D: x, y)
  config.input_dim = 2
  config.seed = 42

  # =========================================================================
  # 9. Differential Scheme
  # =========================================================================
  config.diff = diff = ml_collections.ConfigDict()
  diff.scheme = "nd" # "nd" - numerical differentiation, "ad" - automatic differentiation; "can" - coupled automatic-numerical differentiation (CAN-PDE)
  diff.h_stencil = 0.01 # only apply with "nd" option 
  diff.n_x = 101 
  diff.n_y = 101
  diff.use_continuous_mesh = False

  # =========================================================================
  # 10. VRBA (Variable Residual Balancing Algorithm) Configuration
  # =========================================================================
  config.vrba = vrba = ml_collections.ConfigDict()
  vrba.enabled = True  
  vrba.phi = 0.1

  return config