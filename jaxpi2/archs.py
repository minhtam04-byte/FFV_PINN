from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict

from flax import linen as nn
from flax.core.frozen_dict import freeze

import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant, uniform

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "silu": nn.silu,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]
    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied
    trainable: Tuple[
        bool
    ]  # Specifies whether the period for each axis is trainable or not

    def setup(self):
        # Initialize period parameters as trainable or constant and store them in a flax frozen dict
        period_params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                period_params[f"period_{idx}"] = self.param(
                    f"period_{idx}", constant(self.period[idx]), ()
                )
            else:
                period_params[f"period_{idx}"] = self.period[idx]

        self.period_params = freeze(period_params)

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    nonlinearity: Union[int, list] = 0.0

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity is not None:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb is not None:
            x = FourierEmbs(**self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = self.activation_fn(x)

        x = nn.Dense(features=self.out_dim)(x)

        return x


class NSFlowNet(nn.Module):
    arch_name: Optional[str] = "NSFlowNet"
    shared_layers: int = 4
    branch_layers: int = 3
    shared_dim: int = 128
    branch_dim: int = 64
    activation: str = "silu" 
    out_vars: Tuple[str, ...] = ("u", "v", "p", "T")
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    freq_factor: Optional[float] = None  # Đã sửa lỗi typo và đổi thành float

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity is not None:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb is not None:
            x = FourierEmbs(**self.fourier_emb)(x)
        
        h = x
        start_idx = 0  # Biến kiểm soát chỉ số vòng lặp để tránh trùng tên

        # Lớp đầu tiên: Frequency Annealing (Tuỳ chọn)
        if self.freq_factor is not None: 
            h = nn.Dense(features=self.shared_dim, name="shared_dense_0")(h)
            h = h * (self.freq_factor * jnp.pi)
            h = jnp.sin(h)
            start_idx = 1  # Nếu đã có lớp 0, vòng lặp dưới phải bắt đầu từ lớp 1

        # Các lớp ẩn dùng chung
        for i in range(start_idx, self.shared_layers):
            h = nn.Dense(features=self.shared_dim, name=f"shared_dense_{i}")(h)
            h = self.activation_fn(h)

        shared_features = h

        # Các nhánh riêng biệt (Multi-branch) cho từng biến vật lý
        outputs = []
        for var in self.out_vars:
            branch_h = shared_features
            for i in range(self.branch_layers):
                branch_h = nn.Dense(features=self.branch_dim, name=f"{var}_dense_{i}")(branch_h)
                branch_h = self.activation_fn(branch_h)
            
            # Linear output
            out = nn.Dense(features=1, name=f"{var}_out")(branch_h)
            outputs.append(out)
        
        y = jnp.concatenate(outputs, axis=-1)
        return y

class CavityNet(nn.Module):
    arch_name: Optional[str] = "Cavity"
    shared_layers: int = 4
    branch_layers: int = 3
    shared_dim: int = 128
    branch_dim: int = 64
    
    # Cấu hình nhánh T (Tinh gọn)
    T_layers: int = 3      
    T_dim: int = 64        
    
    activation: str = "silu" 
    out_vars: Tuple[str, ...] = ("u", "v", "p", "T")
    
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    freq_factor: Optional[float] = 4.0 

    def setup(self):
        if self.activation == "silu":
            self.activation_fn = nn.silu
        elif self.activation == "tanh":
            self.activation_fn = jnp.tanh
        else:
            self.activation_fn = nn.relu

    @nn.compact
    def __call__(self, x_in):
        if self.periodicity is not None:
            x_in = PeriodEmbs(**self.periodicity)(x_in)
        if self.fourier_emb is not None:
            x_in = FourierEmbs(**self.fourier_emb)(x_in)

        x_norm = x_in[..., 0:1]
        y_norm = x_in[..., 1:2]
        
        # Phân luồng dữ liệu
        X_uvp = x_in 
        y_embed_T = jnp.cos(jnp.pi / 2.0 * (y_norm + 1.0))
        X_T = jnp.hstack([x_norm, y_embed_T])

        # -------------------------------------------------------------
        # DÒNG CHẢY 1: UVP 
        # -------------------------------------------------------------
        h_uvp = X_uvp
        start_idx_uvp = 0
        
        # Lớp đầu tiên: Frequency Annealing (Hàm Sin) theo nguyên bản
        if self.freq_factor is not None: 
            h_uvp = nn.Dense(features=self.shared_dim, name="uvp_shared_0")(h_uvp)
            h_uvp = jnp.sin(h_uvp * (self.freq_factor * jnp.pi))
            start_idx_uvp = 1  

        for i in range(start_idx_uvp, self.shared_layers):
            h_uvp = nn.Dense(features=self.shared_dim, name=f"uvp_shared_{i}")(h_uvp)
            h_uvp = self.activation_fn(h_uvp)

        outputs_uvp = []
        for var in ("u", "v", "p"):
            b_uvp = h_uvp
            for i in range(self.branch_layers):
                b_uvp = nn.Dense(features=self.branch_dim, name=f"{var}_branch_{i}")(b_uvp)
                b_uvp = self.activation_fn(b_uvp)
            out_var = nn.Dense(features=1, name=f"{var}_out")(b_uvp)
            outputs_uvp.append(out_var)

        # -------------------------------------------------------------
        # DÒNG CHẢY 2: Nhánh T 
        # -------------------------------------------------------------
        h_T = X_T
        start_idx_T = 0
        if self.freq_factor is not None: 
            h_T = nn.Dense(features=self.T_dim, name="T_mlp_0")(h_T)
            h_T = jnp.sin(h_T * (self.freq_factor * jnp.pi))
            start_idx_T = 1

        for i in range(start_idx_T, self.T_layers):
            h_T = nn.Dense(features=self.T_dim, name=f"T_mlp_{i}")(h_T)
            h_T = self.activation_fn(h_T)  
        out_T = nn.Dense(features=1, name="T_out")(h_T)

        outputs_uvp.append(out_T)
        y_out = jnp.concatenate(outputs_uvp, axis=-1)
        
        return y_out


class AdaIPINN(nn.Module):
    arch_name: Optional[str] = "AdaIPINN_nsflow"
    shared_layers: int = 4
    branch_layers: int = 3
    shared_dim: int = 128
    branch_dim: int = 64
    freq_factor: Optional[float] = None  # Sửa lại thành Optional
    out_vars: Tuple[str, ...] = ("u", "v", "p", "T")

    activation: str = "swish"
    num_regions: int = 3     
    n_scale: float = 10.0   
    
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None

    def setup(self):
        self.default_activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x, region_idx: int = 0):
        a_arr = self.param("a_m", constant(1.0 / self.n_scale), (self.num_regions,))
        a_m = a_arr[region_idx]
        act_fn = self.default_activation_fn

        if self.periodicity is not None:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb is not None:
            x = FourierEmbs(**self.fourier_emb)(x)

        h = x
        start_idx = 0

        if self.freq_factor is not None:
            h = nn.Dense(features=self.shared_dim, name="shared_dense_0")(h)
            h = h * (self.freq_factor * jnp.pi)
            h = jnp.sin(h)
            start_idx = 1

        for i in range(start_idx, self.shared_layers):
            h = nn.Dense(features=self.shared_dim, name=f"shared_dense_{i}")(h)
            h = act_fn(self.n_scale * a_m * h)

        shared_features = h

        outputs = []
        for var in self.out_vars:
            branch_h = shared_features
            for i in range(self.branch_layers):
                branch_h = nn.Dense(features=self.branch_dim, name=f"{var}_dense_{i}")(branch_h)
                branch_h = act_fn(self.n_scale * a_m * branch_h)

            out = nn.Dense(features=1, name=f"{var}_out")(branch_h)
            outputs.append(out)

        y = jnp.concatenate(outputs, axis=-1)

        return y


class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    nonlinearity: Union[int, list] = 0.0

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity is not None:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb is not None:
            x = FourierEmbs(**self.fourier_emb)(x)

        u = nn.Dense(features=self.hidden_dim)(x)
        v = nn.Dense(features=self.hidden_dim)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        x = nn.Dense(features=self.out_dim)(x)

        return x


class PirateBlock(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x, u, v):
        identity = x

        x = nn.Dense(features=self.hidden_dim)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = nn.Dense(features=self.hidden_dim)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = nn.Dense(features=self.hidden_dim)(x)
        x = self.activation_fn(x)

        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        x = alpha * x + (1 - alpha) * identity

        return x


class PirateNet(nn.Module):
    arch_name: Optional[str] = "PirateNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: Union[int, list] = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

        if isinstance(self.nonlinearity, (int, float)):
            self.nonlinearities = [self.nonlinearity] * self.num_layers
        else:
            assert len(self.nonlinearity) == self.num_layers
            self.nonlinearities = self.nonlinearity

    @nn.compact
    def __call__(self, x):
        if self.periodicity is not None:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb is not None:
            x = FourierEmbs(**self.fourier_emb)(x)

        u = nn.Dense(features=self.hidden_dim)(x)
        u = self.activation_fn(u)

        v = nn.Dense(features=self.hidden_dim)(x)
        v = self.activation_fn(v)

        for i in range(self.num_layers):
            x = PirateBlock(hidden_dim=self.hidden_dim,
                            output_dim=x.shape[-1],
                            activation=self.activation,
                            nonlinearity=self.nonlinearities[i])(x, u, v)

        x = nn.Dense(features=self.out_dim)(x)

        return x
    
