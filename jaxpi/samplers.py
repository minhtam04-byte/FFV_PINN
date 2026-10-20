import itertools
from functools import partial

import jax
import jax.numpy as jnp
from jax import random, jit


class BaseSampler:
    """Infinite iterator over randomly generated batches.

    The global batch is sharded across devices by the training step, so the
    batch size must be divisible by the number of devices.
    """

    _uid_counter = itertools.count()

    def __init__(self, batch_size, rng_key=random.PRNGKey(1234)):
        # data_generation is jitted with static self; see PINN.__init__ for
        # why the default id()-based hash is unsafe across instances.
        self._uid = next(BaseSampler._uid_counter)

        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = jax.device_count()
        assert batch_size % self.num_devices == 0, (
            f"Batch size {batch_size} must be divisible by the number of "
            f"devices {self.num_devices}"
        )

    def __hash__(self):
        return self._uid

    def __eq__(self, other):
        return self is other

    def __iter__(self):
        return self

    def __next__(self):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        batch = self.data_generation(subkey)
        return batch

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")


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
    
class FixedGridFVMSampler(BaseSampler):
    def __init__(self, config, batch_size=None, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.L = config.physics.L
        self.H = config.physics.H
        self.h = config.fvm_setup.h_stencil
        self.key = rng_key
        
        # 1. Tạo tọa độ chuẩn xác tuyệt đối theo bước lưới h
        # x = jnp.arange(self.h / 2.0, self.L, self.h)
        # y = jnp.arange(self.h / 2.0, self.H, self.h)
        nx = int(jnp.round(self.L / self.h)) + 1
        ny = int(jnp.round(self.H / self.h)) + 1
        
        x = jnp.linspace(0.0, self.L, nx)
        y = jnp.linspace(0.0, self.H, ny)
        X, Y = jnp.meshgrid(x, y)
        
        # 2. Gom thành danh sách điểm full miền (Bao gồm cả nội miền lẫn điểm nằm trên biên)
        #self.full_pts = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
        X_int = X[1:-1, 1:-1].flatten()
        Y_int = Y[1:-1, 1:-1].flatten()
        self.interior_pts = jnp.stack([X_int, Y_int], axis=-1)
        
        # 3. Bóc tách các nút trên 4 vách biên (Boundary Nodes)
        left_wall   = jnp.stack([X[:, 0], Y[:, 0]], axis=-1)      # x = 0
        right_wall  = jnp.stack([X[:, -1], Y[:, -1]], axis=-1)    # x = L
        bottom_wall = jnp.stack([X[0, :], Y[0, :]], axis=-1)      # y = 0
        top_wall    = jnp.stack([X[-1, :], Y[-1, :]], axis=-1)     # y = H
        
        self.bc_pts = jnp.concatenate([left_wall, right_wall, bottom_wall, top_wall], axis=0)
        self.total_bc_pts = self.bc_pts.shape[0]
        self.total_pts = self.interior_pts.shape[0]
        self.full_pts = self.interior_pts
        # 3. Setup cơ chế Batch
        if batch_size is None or batch_size >= self.total_pts:
            self.batch_size = self.total_pts
        else:
            self.batch_size = batch_size

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
        super().__init__(batch_size, rng_key)
        self.L = config.physics.L
        self.H = config.physics.H
        self.h = config.fvm_setup.h_stencil # Vẫn lưu h nhưng KHÔNG dùng để chia lưới
        self.batch_size = batch_size
        self.key = rng_key
        
        eps = self.h*1/2
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