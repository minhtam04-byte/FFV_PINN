import jax
import jax.numpy as jnp
import numpy as np
from jaxpi.samplers import BaseSampler

class R3CollocationSampler(BaseSampler):
    """
    R3 Sampler chuẩn hóa theo Algorithm 1 (Arka Daw et al., 2023).
    Sử dụng dynamic threshold tau = mean(residuals) nhưng đảm bảo tổng batch cố định.
    """
    def __init__(self, config, batch_size=4096, rng_key=jax.random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.L = config.physics.get("L", 1.0)
        self.H = config.physics.get("H", 1.0)

        # Khởi tạo batch ban đầu phân bố đều ngẫu nhiên
        self.key, subkey = jax.random.split(self.key)
        self.current_pts = jax.random.uniform(
            subkey, 
            shape=(self.batch_size, 2), 
            minval=jnp.array([0.0, 0.0]), 
            maxval=jnp.array([self.L, self.H])
        )

    def data_generation(self, key):
        return self.current_pts

    def update_with_residuals(self, res_dict):
        """Cập nhật điểm theo đúng Threshold tau_i = Mean(F) của bài báo."""
        # 1. Tính hàm thặng dư F(x) = |ru| + |rv| + |rc| + |rT|
        res_mag = (
            jnp.abs(res_dict["u"]) + 
            jnp.abs(res_dict["v"]) + 
            jnp.abs(res_dict["p"]) + 
            jnp.abs(res_dict["T"])
        )
        
        # 2. Ngưỡng tau_i là giá trị trung bình thặng dư của batch (Eq. 4 trong bài báo)
        tau = jnp.mean(res_mag)
        
        # Chuyển nhẹ sang numpy CPU để lọc mask linh hoạt (không ảnh hưởng GPU)
        res_mag_np = np.asarray(res_mag)
        pts_np = np.asarray(self.current_pts)
        
        # 3. Retain: Giữ lại những điểm có F(x) > tau
        retained_mask = res_mag_np > tau
        retained_pts = pts_np[retained_mask]
        num_retained = retained_pts.shape[0]

        # Đảm bảo luôn giữ ít nhất 1 điểm và không giữ toàn bộ 100%
        if num_retained == 0:
            num_retained = int(self.batch_size * 0.2)
            retained_pts = pts_np[:num_retained]
        elif num_retained >= self.batch_size:
            num_retained = int(self.batch_size * 0.8)
            retained_pts = pts_np[:num_retained]

        num_resample = self.batch_size - num_retained

        # 4. Resample: Sinh số lượng điểm bù từ phân bố đều U(Omega)
        self.key, subkey = jax.random.split(self.key)
        resampled_pts = jax.random.uniform(
            subkey,
            shape=(num_resample, 2),
            minval=jnp.array([0.0, 0.0]),
            maxval=jnp.array([self.L, self.H])
        )

        # 5. Merge: Tổng số lượng điểm luôn đúng bằng batch_size (4096)
        next_pts = np.vstack([retained_pts, np.asarray(resampled_pts)])
        self.current_pts = jnp.asarray(next_pts)