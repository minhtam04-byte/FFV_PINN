from functools import partial

import jax.numpy as jnp
from jax import random, jit, local_device_count

from torch.utils.data import Dataset


class BaseSampler(Dataset):
    def __init__(self, batch_size, rng_key=random.PRNGKey(1234)):
        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = local_device_count()

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        batch = self.data_generation(subkey)
        return batch

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")

class MeshSampler(BaseSampler):
    def __init__(self, mesh, labels=None, batch_size=1024, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.mesh = mesh
        self.labels = labels

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        """Generates data containing batch_size samples."""
        idx = random.choice(key, self.mesh.shape[0], shape=(self.batch_size,))
        batch = self.mesh[idx, :]

        if self.labels is None:
            return batch
        else:
            batch_labels = self.labels[idx]
            return batch, batch_labels


class UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, sort_axis=0, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]
        self.sort_axis = sort_axis  # sorting by the first coordinate (e.g., time) for causal training

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        if self.sort_axis is not None:
            sorted_indices = jnp.argsort(batch[:, self.sort_axis])
            batch = batch[sorted_indices]
        return batch


class TemporalMeshSampler(BaseSampler):
    def __init__(
            self, temporal_dom, mesh, batch_size, rng_key=random.PRNGKey(1234)
    ):
        super().__init__(batch_size, rng_key)
        self.temporal_dom = temporal_dom
        self.mesh = mesh

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        key1, key2 = random.split(key)

        temporal_batch = random.uniform(
            key1,
            shape=(self.batch_size, 1),
            minval=self.temporal_dom[0],
            maxval=self.temporal_dom[1],
        )
        spatial_idx = random.choice(
            key2, self.mesh.shape[0], shape=(self.batch_size,)
        )
        spatial_batch = self.mesh[spatial_idx, :]
        batch = jnp.concatenate([temporal_batch, spatial_batch], axis=1)

        return batch

# class HybridInterfaceSampler(BaseSampler):
#     def __init__(self, config, batch_size, interface_ratio=0.25, rng_key=random.PRNGKey(1234)):
#         """
#         Sampler lai tối ưu hóa: Sử dụng phân phối Gauss (Normal) cho vùng biên 
#         để điểm loang đều, mịn màng sang hai bên giao diện vật lý.
#         """
#         super().__init__(batch_size, rng_key)
        
#         self.L = config.physics.L
#         self.H = config.physics.H
        
#         ibm_config = config.get('ibm', {})
#         self.layer_x_center = ibm_config.get('layer_x_center')
#         self.layer_width = ibm_config.get('layer_width')
        
#         # Thiết lập độ lệch chuẩn (sigma) cho phân phối Gauss dựa trên eta_diffuse
#         # eta_diffuse = 20.0 -> sigma = 0.04 giúp điểm loang rộng và mịn hơn
#         eta_diffuse = ibm_config.get('eta_diffuse')
#         self.sigma = 0.8 / eta_diffuse 
        
#         # Chia tĩnh số lượng điểm cho đồ thị JAX JIT
#         self.n_interface = int(batch_size * interface_ratio)
#         self.n_global = batch_size - self.n_interface
        
#         # Tọa độ chính xác của vách trái và vách phải khối xốp
#         self.x_left_boundary = self.layer_x_center - (self.layer_width / 2.0)
#         self.x_right_boundary = self.layer_x_center + (self.layer_width / 2.0)
        
#         # Chia đều điểm giao diện cho 2 bên vách
#         self.n_left = self.n_interface // 2
#         self.n_right = self.n_interface - self.n_left

#     @partial(jit, static_argnums=(0,))
#     def data_generation(self, key):
#         """
#         Sinh dữ liệu phân phối lai (Global Uniform + Interface Gauss) sử dụng vector hóa của JAX.
#         """
#         k1, k2, k3, k4, k5, k6 = random.split(key, 6)
        
#         # ------------------------------------------------------------
#         # PHẦN 1: Sinh điểm ngẫu nhiên đều trên toàn bộ khoang (Global)
#         # ------------------------------------------------------------
#         x_global = random.uniform(k1, shape=(self.n_global, 1), minval=0.0, maxval=self.L)
#         y_global = random.uniform(k2, shape=(self.n_global, 1), minval=0.0, maxval=self.H)
#         pts_global = jnp.concatenate([x_global, y_global], axis=-1)
        
#         # ------------------------------------------------------------
#         # PHẦN 2: Sinh điểm giao diện loang mịn bằng phân phối Gauss
#         # ------------------------------------------------------------
#         # Biên trái: Tâm là x_left_boundary, loang ra bằng self.sigma
#         x_int_left = self.x_left_boundary + random.normal(k3, shape=(self.n_left, 1)) * self.sigma
#         y_int_left = random.uniform(k4, shape=(self.n_left, 1), minval=0.0, maxval=self.H)
#         pts_left = jnp.concatenate([x_int_left, y_int_left], axis=-1)
        
#         # Biên phải: Tâm là x_right_boundary, loang ra bằng self.sigma
#         x_int_right = self.x_right_boundary + random.normal(k5, shape=(self.n_right, 1)) * self.sigma
#         y_int_right = random.uniform(k6, shape=(self.n_right, 1), minval=0.0, maxval=self.H)
#         pts_right = jnp.concatenate([x_int_right, y_int_right], axis=-1)
        
#         # Gộp hai vách biên độc lập
#         pts_interface = jnp.concatenate([pts_left, pts_right], axis=0)
        
#         # Kiểm soát tọa độ không bị tràn ra ngoài biên vật lý khoang [0, L]
#         x_clipped = jnp.clip(pts_interface[..., 0:1], 0.0, self.L)
#         y_clipped = jnp.clip(pts_interface[..., 1:2], 0.0, self.H)
#         pts_interface = jnp.concatenate([x_clipped, y_clipped], axis=-1)
        
#         # ------------------------------------------------------------
#         # PHẦN 3: Hợp nhất toàn bộ dữ liệu mẫu
#         # ------------------------------------------------------------
#         batch = jnp.concatenate([pts_global, pts_interface], axis=0)
        
#         return batch
    
    

import jax.numpy as jnp

class FVMSampler:
    def __init__(self, config, nx=60, ny=60):
        """
        Sampler lưới cấu trúc cố định cho FVM, trả về mảng tọa độ P (N, 2)
        """
        self.L = config.physics.L
        self.H = config.physics.H
        h = config.fvm_setup.h_stencil
        
        # 1. Khởi tạo lưới tọa độ trung tâm
        x = jnp.linspace(h * 4, self.L - 4 * h, nx)
        y = jnp.linspace(h * 4, self.H - 4 * h, ny)
        X, Y = jnp.meshgrid(x, y)
        
        # 2. Gom thành ma trận (N, 2)
        self.pts = jnp.concatenate([X.flatten().reshape(-1, 1), Y.flatten().reshape(-1, 1)], axis=-1)

    def __iter__(self):
        return self

    def __next__(self):
        # Trả về TOẠ ĐỘ, không trả về dict
        return self.pts


class BoundaryLayerSampler(BaseSampler):
    def __init__(self, config, batch_size, boundary_ratio=0.1, rng_key=random.PRNGKey(1234)):
        """
        Sampler AD tối ưu cho miền 100% xốp: 
        Tập trung 70% điểm vào dải hẹp sát 4 bức tường (Boundary Layers), 30% rải đều toàn miền.
        """
        super().__init__(batch_size, rng_key)
        self.L = config.physics.L
        self.H = config.physics.H
        h =  config.fvm_setup.h_stencil
        # Độ dày của lớp biên muốn tập trung AD (ví dụ 5% chiều dài miền)
        self.bl_thickness_x = 4 * h * self.L 
        self.bl_thickness_y = 4 * h * self.H
        

        # Chia đều điểm cho 4 bức tường
        self.n_per_wall = int(batch_size * boundary_ratio) // 4
        self.n_boundary = self.n_per_wall * 4
        self.n_global = batch_size - self.n_boundary

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        keys = random.split(key, 10)
        
        # 1. Rải đều toàn miền (Global Interior)
        x_glob = random.uniform(keys[0], shape=(self.n_global, 1), minval=0.0, maxval=self.L)
        y_glob = random.uniform(keys[1], shape=(self.n_global, 1), minval=0.0, maxval=self.H)
        pts_global = jnp.concatenate([x_glob, y_glob], axis=-1)
        
        # 2. Tường Trái (x gần 0)
        x_left = random.uniform(keys[2], shape=(self.n_per_wall, 1), minval=0.0, maxval=self.bl_thickness_x)
        y_left = random.uniform(keys[3], shape=(self.n_per_wall, 1), minval=0.0, maxval=self.H)
        pts_left = jnp.concatenate([x_left, y_left], axis=-1)
        
        # 3. Tường Phải (x gần L)
        x_right = random.uniform(keys[4], shape=(self.n_per_wall, 1), minval=self.L - self.bl_thickness_x, maxval=self.L)
        y_right = random.uniform(keys[5], shape=(self.n_per_wall, 1), minval=0.0, maxval=self.H)
        pts_right = jnp.concatenate([x_right, y_right], axis=-1)
        
        # 4. Tường Đáy (y gần 0)
        x_bot = random.uniform(keys[6], shape=(self.n_per_wall, 1), minval=0.0, maxval=self.L)
        y_bot = random.uniform(keys[7], shape=(self.n_per_wall, 1), minval=0.0, maxval=self.bl_thickness_y)
        pts_bot = jnp.concatenate([x_bot, y_bot], axis=-1)
        
        # 5. Tường Đỉnh (y gần H) - Đã sửa lỗi PRNGKey(0) tĩnh bằng khóa keys[9] động
        x_top = random.uniform(keys[8], shape=(self.n_per_wall, 1), minval=0.0, maxval=self.L)
        y_top = random.uniform(keys[9], shape=(self.n_per_wall, 1), minval=self.H - self.bl_thickness_y, maxval=self.H)
        pts_top = jnp.concatenate([x_top, y_top], axis=-1)
        
        # Hợp nhất ma trận tĩnh (Kích thước mảng luôn đảm bảo bằng chính xác batch_size)
        batch = jnp.concatenate([pts_global, pts_left, pts_right, pts_bot, pts_top], axis=0)
        return batch
    
class FixedGridFVMSampler(BaseSampler):
    def __init__(self, config, batch_size=None, rng_key=random.PRNGKey(1234)):
        self.L = config.physics.L
        self.H = config.physics.H
        self.h = config.fvm_setup.h_stencil
        self.key = rng_key
        
        # Tạo lưới Node-centered bao phủ từ biên này sang biên kia (0 -> L và 0 -> H)
        # Số lượng điểm tính chính xác để đảm bảo delta_x = delta_y = h_stencil
        nx = int(self.L / self.h)
        ny = int(self.H / self.h) 
        
        # 1. Tạo tọa độ chuẩn xác tuyệt đối theo bước lưới h
        x = jnp.linspace(self.h/2.0, self.L - self.h/2.0, nx)
        y = jnp.linspace(self.h/2.0, self.H - self.h/2.0, ny)
        X, Y = jnp.meshgrid(x, y)
        
        # 2. Gom thành danh sách điểm full miền (Bao gồm cả nội miền lẫn điểm nằm trên biên)
        self.full_pts = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
        self.total_pts = self.full_pts.shape[0]
        
        # 3. Setup cơ chế Batch
        self.batch_size = batch_size if batch_size is not None else self.total_pts

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        if self.batch_size >= self.total_pts:
            return self.full_pts
        
        idx = random.choice(
            key, 
            jnp.arange(self.total_pts), 
            shape=(self.batch_size,), 
            replace=False
        )
        return self.full_pts[idx]

class ContinuousFVMSampler(BaseSampler):
    def __init__(self, config, batch_size=4096, rng_key=random.PRNGKey(1234)):
        self.L = config.physics.L
        self.H = config.physics.H
        self.h = config.fvm_setup.h_stencil # Vẫn lưu h nhưng KHÔNG dùng để chia lưới
        self.batch_size = batch_size
        self.key = rng_key
        eps = self.h/2
        self.dom = jnp.array([
            [eps, config.physics.L - eps],  
            [eps, config.physics.H - eps]  
        ])
        self.dim = self.dom.shape[0]

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        return batch