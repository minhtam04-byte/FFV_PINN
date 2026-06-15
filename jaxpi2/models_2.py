from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Dict

from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, value_and_grad, random, jacfwd, jacrev
from jax.tree_util import tree_map, tree_reduce, tree_leaves

from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

import optax

from jaxpi2 import archs
from jaxpi2.utils import flatten_pytree

from soap_jax import soap


from typing import NamedTuple

class PhysicsState(NamedTuple):
    u: jnp.ndarray
    v: jnp.ndarray
    p: jnp.ndarray
    T: jnp.ndarray
    u_x: jnp.ndarray
    v_x: jnp.ndarray
    p_x: jnp.ndarray
    T_x: jnp.ndarray
    u_y: jnp.ndarray
    v_y: jnp.ndarray
    p_y: jnp.ndarray
    T_y: jnp.ndarray
    u_xx: jnp.ndarray  
    v_xx: jnp.ndarray
    u_yy: jnp.ndarray
    v_yy: jnp.ndarray
    u_yx: jnp.ndarray
    v_yx: jnp.ndarray
    laplacian_u: jnp.ndarray
    laplacian_v: jnp.ndarray
    laplacian_p: jnp.ndarray
    laplacian_T: jnp.ndarray
    alpha: jnp.ndarray
    gamma: jnp.ndarray
    alpha_x: jnp.ndarray
    alpha_y: jnp.ndarray

class TrainState(train_state.TrainState):
    loss_weights: Dict
    pts_weights: Dict
    momentum: float
    prev_params: Any = None

    def apply_loss_weights(self, loss_weights, **kwargs):
        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        loss_weights = tree_map(running_average, self.loss_weights, loss_weights)
        loss_weights = lax.stop_gradient(loss_weights)

        return self.replace(
            loss_weights=loss_weights,
            **kwargs,
        )

    def apply_pts_weights(self, pts_weights, **kwargs):
        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        pts_weights = tree_map(running_average, self.pts_weights, pts_weights)
        pts_weights = lax.stop_gradient(pts_weights)

        return self.replace(
            pts_weights=pts_weights,
            **kwargs,
        )


def create_arch(config):
    arch_name = config.arch_name.lower()

    if arch_name == "mlp":
        arch = archs.Mlp(**config)

    elif arch_name == "modifiedmlp":
        arch = archs.ModifiedMlp(**config)

    elif arch_name == "piratenet":
        arch = archs.PirateNet(**config)

    elif arch_name == "nsflownet":
        arch = archs.NSFlowNet(**config)
        
    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")

    return arch


def create_lr_schedule(config):
    if config.lr_schedule == "exponential_decay":
        lr = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            transition_steps=config.decay_steps,  # every decay_steps, the learning rate decays by decay_rate
            decay_rate=config.decay_rate,
            staircase=config.staircase
        )
    elif config.lr_schedule == "cosine_decay":
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,  # total number of steps for decay
            end_value=config.end_learning_rate,
        )
    return lr


def create_optimizer(config, lr):
    optimizer = config.optimizer.lower()

    if optimizer == "adam":
        tx = optax.adam(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )

    elif optimizer == "soap":
        tx = soap(
            learning_rate=lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            weight_decay=0.0,
            precondition_frequency=2,
            max_precond_dim=10000
        )

    elif optimizer == "muon":
        tx = optax.contrib.muon(
            learning_rate=lr,
            ns_coeffs=(2, -1.5, 0.5),
            ns_steps=10,
            beta=0.99,
            adam_b1=0.99
        )

    if config.schedule_free:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.contrib.schedule_free(tx, lr, b1=config.beta1)
        )

    return tx


def create_train_state(config, tx, arch, params=None, train_state_cls=TrainState):
    # Initialize network
    x = jnp.ones(config.arch_input_dim)

    if params is None:  # if not then, used for transfer learning
        params = arch.init(random.PRNGKey(config.seed), x)

    # if config.pseudo_time.enabled:
    pts_weights = dict(config.pseudo_time.pts_weights)
    # else:
    #     pts_weights = None

    loss_weights = dict(config.loss_weighting.loss_weights)

    state = train_state_cls.create(
        apply_fn=arch.apply,
        params=params,
        prev_params=params,
        tx=tx,
        loss_weights=loss_weights,
        pts_weights=pts_weights,
        momentum=config.loss_weighting.momentum,
    )

    return state


class PINN:
    def __init__(self, config, lr, tx, arch, state):
        self.config = config
        self.lr = lr
        self.tx = tx
        self.arch = arch
        self.state = state
        self.mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")

        self.step = self.create_step_fn()
        self.update_loss_weights = self.create_update_loss_weights_fn()
        self.update_pts_weights = self.create_update_pts_weights_fn()

        self.sol_pred_fn = vmap(self.neural_net, (None,) + (0,) * self.config.input_dim)
        #self.r_pred_fn = vmap(self.r_net, (None,) + (0,) * self.config.input_dim)
        self.r_pred_base_fn = vmap(self.r_net_base, (None,) + (0,) * self.config.input_dim)
        self.r_pred_total_fn = vmap(self.r_net_total, (None, None, None) + (0,) * self.config.input_dim)

    def neural_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")
    
    def r_net_total(self, params, prev_params, pts_weights, *args):
        raise NotImplementedError("Subclasses MUST implement Scale-PINN logic here!")

    def losses(self, params, state, batch):
        raise NotImplementedError("Subclasses should implement this!")

    @partial(jit, static_argnums=(0,))
    def compute_pts_weights(self, state, init_state, batch):
        # Unpack all columns regardless of batch dimensionality (t,x) or (t,x,y) etc.
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        # Stack predictions and residuals: shape (n_components, N)
        sols_pred = jnp.stack(self.sol_pred_fn(state.params, *coords))
        sols_prev = jnp.stack(self.sol_pred_fn(state.prev_params, *coords))

        res_pred = jnp.stack(self.r_pred_base_fn(state.params, *coords))
        res_prev = jnp.stack(self.r_pred_base_fn(state.prev_params, *coords))
        res0_pred = jnp.stack(self.r_pred_base_fn(init_state.params, *coords))

        if res0_pred.ndim == 1:
            res0_pred = res0_pred[None, :]
        losses0 = jnp.mean(res0_pred ** 2, axis=1)  # (n_components,)

        def cosine_decay_from_loss(
                losses,
                loss0,
                start_log_drop=3.0,  # no decay before this
                end_log_drop=5.0,  # reach min_factor here
                min_factor=0.1,
                eps=1e-8,
        ):
            log_drop = jnp.log10((loss0 + eps) / (losses + eps))
            p = jnp.clip((log_drop - start_log_drop) / (end_log_drop - start_log_drop), 0.0, 1.0)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + jnp.cos(jnp.pi * p))

        if res_pred.ndim == 1:
            sols_pred = sols_pred[None, :]  # (n_components, N)
            sols_prev = sols_prev[None, :]  # (n_components, N)
            res_pred = res_pred[None, :]
            res_prev = res_prev[None, :]

        sol_diffs = sols_pred - sols_prev
        res_diffs = res_pred - res_prev

        losses = jnp.mean(res_pred ** 2, axis=1)  # (n_components,)

        if self.config.pseudo_time.shrink.enabled:
            factors = cosine_decay_from_loss(
                losses,
                losses0,
                start_log_drop=self.config.pseudo_time.shrink.start_log_drop,
                end_log_drop=self.config.pseudo_time.shrink.end_log_drop,
                min_factor=self.config.pseudo_time.shrink.min_factor,
            )

        else:
            factors = 1.0

        weights = (
                jnp.linalg.norm(res_diffs, axis=1)
                / (jnp.linalg.norm(sol_diffs, axis=1) + 1e-8) * factors
        )
        weights = jnp.clip(weights, a_min=1e-2, a_max=100.0)
        weights = lax.stop_gradient(weights)

        keys = list(state.pts_weights.keys())
        return dict(zip(keys, weights))

    #@partial(jit, static_argnums=(0,))
    def compute_loss_weights(self, state, batch):
        """
        Balance losses based on the gradient norms of each loss.
        """
        # Compute the gradient of each loss w.r.t. the parameters
        grads = jacrev(self.losses)(state.params, state, batch)

        # Compute the grad norm of each loss
        grad_norm_dict = {}
        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

        # Compute the mean of grad norms over all losses
        mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
        # Grad Norm Weighting
        w = tree_map(lambda x: (mean_grad_norm / (x + 1e-5 * mean_grad_norm)), grad_norm_dict)
        return w

    #@partial(jit, static_argnums=(0,))
    def loss(self, params, state, batch):
        # Compute losses
        loss_dict = self.losses(params, state, batch)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, loss_dict, state.loss_weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss, loss_dict

    def create_step_fn(self):
        @jax.jit
        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(P(), P("batch")),
            out_specs=(P(), P(), P()),
            check_rep=False
        )
        def step(state, batch):
            prev_params = state.params
            (loss, loss_dict), grads = value_and_grad(self.loss, has_aux=True)(state.params, state, batch)
            # state = state.apply_gradients(grads=grads)
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            state = state.replace(
                step=state.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                prev_params=prev_params
            )
            return state, loss, loss_dict

        return step

    def create_update_loss_weights_fn(self):
        @jax.jit
        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(P(), P("batch")),
            out_specs=P(),
            check_rep=False
        )
        def update_loss_weights(state, batch):
            loss_weights = self.compute_loss_weights(state, batch)
            state = state.apply_loss_weights(loss_weights=loss_weights)
            return state

        return update_loss_weights

    def create_update_pts_weights_fn(self):
        @jax.jit
        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(P(), P(), P("batch")),
            out_specs=P(),
            check_rep=False
        )
        def update_pts_weights(state, prev_state, batch):
            pts_weights = self.compute_pts_weights(state, prev_state, batch)
            state = state.apply_pts_weights(pts_weights=pts_weights)
            return state

        return update_pts_weights


class ForwardIVP(PINN):
    def __init__(self, config, lr, tx, arch, state):
        super().__init__(config, lr, tx, arch, state)
        if config.causal.enabled:
            self.tol = config.causal.tol
            self.num_chunks = config.causal.num_chunks
            self.triu = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1)

    @partial(jit, static_argnums=(0,))
    def compute_causal_weights(self, state, batch):
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        # Stack residuals: shape (n_components, N)
        res = jnp.stack(self.r_pred_fn(state.params, *coords))

        if res.ndim == 1:
            res = res[None, :]

        if self.config.pseudo_time.enabled:
            sols_pred = jnp.stack(self.sol_pred_fn(state.params, *coords))
            sols_prev = jnp.stack(self.sol_pred_fn(state.prev_params, *coords))

            pts_weights = jnp.array(list(state.pts_weights.values()))  # (n_components,)
            res = res + pts_weights[:, None] * (sols_pred - sols_prev)

        # Chunk, loss, and causal weights — all vectorised over components
        res = res.reshape(res.shape[0], self.num_chunks, -1)  # (n_components, chunks, N)
        losses = jnp.mean(res ** 2, axis=2)  # (n_components, chunks)
        gammas = lax.stop_gradient(
            jnp.exp(-self.tol * (losses @ self.triu))
        )  # (n_components, chunks)

        return gammas.min(axis=0)

    # @partial(jit, static_argnums=(0,))
    def compute_residual_losses(self, params, state, batch, pseudo_time=False, causal=False):
        keys = list(state.pts_weights.keys())  # TODO: Seperate IC/BC and PDE keys
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        res_pred = jnp.stack(self.r_pred_fn(params, *coords))  # (n_components, N)

        if res_pred.ndim == 1:
            res_pred = res_pred[None, :]

        if pseudo_time:
            sols_pred = jnp.stack(self.sol_pred_fn(params, *coords))
            sols_prev = jnp.stack(self.sol_pred_fn(state.prev_params, *coords))
            pts_weights = jnp.array(list(state.pts_weights.values()))  # (n_components,)
            res_pred = res_pred + pts_weights[:, None] * (sols_pred - sols_prev)

        if causal:
            res_pred = res_pred.reshape(res_pred.shape[0], self.num_chunks, -1)  # (n_components, chunks, n)
            chunk_loss = jnp.mean(res_pred ** 2, axis=2)  # (n_components, chunks)
            causal_weights = lax.stop_gradient(
                jnp.exp(-self.tol * (chunk_loss @ self.triu.T))
            )  # (K, chunks)
            per_key_losses = jnp.mean(chunk_loss * causal_weights, axis=1)  # (n_components,)
        else:
            per_key_losses = jnp.mean(res_pred ** 2, axis=1)  # (n_components,)

        return dict(zip(keys, per_key_losses))


class ForwardBVP(PINN):
    def __init__(self, config, lr, tx, arch, state):
        super().__init__(config, lr, tx, arch, state)

    # @partial(jit, static_argnums=(0,))
    # def compute_residual_losses(self, params, state, batch, pseudo_time=False):
    #     keys = list(state.pts_weights.keys())  # TODO: Seperate IC/BC and PDE keys
    #     coords = tuple(batch[:, i] for i in range(batch.shape[1]))

    #     res_pred = jnp.stack(self.r_pred_fn(params, *coords))  # (n_components, N)

    #     if res_pred.ndim == 1:
    #         res_pred = res_pred[None, :]

    #     if pseudo_time:
    #         sols_pred = jnp.stack(self.sol_pred_fn(params, *coords))
    #         sols_prev = jnp.stack(self.sol_pred_fn(state.prev_params, *coords))
    #         pts_weights = jnp.array(list(state.pts_weights.values()))  # (n_components,)
    #         res_pred = res_pred + pts_weights[:, None] * (sols_pred - sols_prev)

    #     per_key_losses = jnp.mean(res_pred ** 2, axis=1)  # (n_components,)

    #     return dict(zip(keys, per_key_losses))
    def compute_residual_losses(self, params, state, batch):
        keys = list(state.pts_weights.keys())
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))
        res_total = jnp.stack(self.r_pred_total_fn(params, state.prev_params, state.pts_weights, *coords))

        if res_total.ndim == 1:
            res_total = res_total[None, :]

        per_key_losses = jnp.mean(res_total ** 2, axis=1)
        return dict(zip(keys, per_key_losses))


class BaseHeatPorousScalePINN_2(ForwardBVP):
    def __init__(self, config, lr, tx, arch, state, lb, ub):
        super().__init__(config, lr, tx, arch, state)
        
        self.lb = lb
        self.ub = ub
        
        self.tau_sc = config.physics.get('tau_sc', 0.5)
        self.tau_alpha = config.physics.get('tau_alpha', 0.5)
        self.inf_small = jnp.finfo(jnp.float32).eps
        ibm_config = config.get('ibm', {})
        self.use_ibm = ibm_config.get('mode')

        self.use_fvm_residual = config.fvm_setup.get('mode')
        if self.use_fvm_residual:
            self.h_stencil = config.fvm_setup.get('h_stencil')
            self.alpha_u = config.fvm_setup.get('alpha_u')
            self.alpha_p = config.fvm_setup.get('alpha_p')
        
        if self.use_ibm:
            self.eta_diffuse = ibm_config.get('eta_diffuse', 30.0)
            self.layer_x_center = ibm_config.get('layer_x_center', 0.5)
            self.layer_width = ibm_config.get('layer_width', 1.0 / 3.0)

    def u_net(self, params, x, y, **kwargs):
        outputs = self.arch.apply(params, jnp.concatenate([x, y], axis=-1))
        return outputs[..., 0:1], outputs[..., 1:2], outputs[..., 2:3], outputs[..., 3:4]

    def _get_geometry_alpha(self, x_in, y_in):
        """
        Trả về phân bố porosity (độ rỗng).
        - Nếu bật IBM: Chuyển tiếp mượt giữa Fluid (alpha=1) và Porous (alpha=epsilon_p)
        - Nếu tắt IBM: Giả sử toàn miền là môi trường xốp đồng nhất (alpha = epsilon_p)
        """
        if self.use_ibm:
            phi = self.sdf_fn(x_in, y_in)
            return (1.0 + self.epsilon_p) / 2.0 + (1.0 - self.epsilon_p) / 2.0 * jnp.tanh(self.eta_diffuse * phi)
        else:
            return self.epsilon_p + 0.0 * x_in

    def _get_geometry_gamma(self, x_in, y_in):
        """
        Trả về cờ kích hoạt lực cản Darcy-Forchheimer.
        - Nếu bật IBM: Chỉ bật ở vùng có phi < 0 (gamma -> 1)
        - Nếu tắt IBM: Bật trên toàn miền (gamma = 1)
        """
        if self.use_ibm:
            phi = self.sdf_fn(x_in, y_in)
            return 0.5 * (1.0 - jnp.tanh(self.eta_diffuse * phi))
        else:
            return 1.0 + 0.0 * x_in

    # --- 2. TÍNH ĐẠO HÀM LIÊN TỤC (AD) ---
    def _get_physics_components(self, params, x, y, **kwargs):
        x_s = jnp.squeeze(x)
        y_s = jnp.squeeze(y)

        def forward(x_in, y_in):
            u, v, p, T = self.u_net(params, x_in, y_in, **kwargs)
            return u, v, p, T

        u, v, p, T = forward(x_s, y_s)
        jac_f = jax.jacrev(forward, argnums=(0, 1))(x_s, y_s)
        jac_u, jac_v, jac_p, jac_T = jac_f

        u_x, u_y = jac_u
        v_x, v_y = jac_v
        p_x, p_y = jac_p
        T_x, T_y = jac_T

        hess_x = jax.jacfwd(jax.jacrev(forward, argnums=0), argnums=0)(x_s, y_s)
        hess_y = jax.jacfwd(jax.jacrev(forward, argnums=1), argnums=1)(x_s, y_s)
        hess_xy = jax.jacfwd(jax.jacrev(forward, argnums=0), argnums=1)(x_s, y_s) 

        u_xx, v_xx, p_xx, T_xx = hess_x
        u_yy, v_yy, p_yy, T_yy = hess_y
        u_yx, v_yx, p_yx, T_yx = hess_xy 

        alpha = self._get_geometry_alpha(x_s, y_s)
        gamma = self._get_geometry_gamma(x_s, y_s)

        # 5. [TỐI ƯU HIỆU NĂNG]: Cắt bỏ hoàn toàn AutoDiff khi không dùng IBM
        if self.use_ibm:
            jac_alpha = jax.jacrev(self._get_geometry_alpha, argnums=(0, 1))(x_s, y_s)
            alpha_x, alpha_y = jac_alpha
        else:
            alpha_x, alpha_y = jnp.array(0.0), jnp.array(0.0)

        # 6. Đóng gói an toàn vào PhysicsState
        return PhysicsState(
            u=u, v=v, p=p, T=T,
            u_x=u_x, v_x=v_x, p_x=p_x, T_x=T_x,
            u_y=u_y, v_y=v_y, p_y=p_y, T_y=T_y,
            u_xx=u_xx, v_xx=v_xx, 
            u_yy=u_yy, v_yy=v_yy,
            u_yx=u_yx, v_yx=v_yx,
            laplacian_u=u_xx + u_yy,
            laplacian_v=v_xx + v_yy,
            laplacian_p=p_xx + p_yy,
            laplacian_T=T_xx + T_yy,
            alpha=alpha, 
            gamma=gamma, 
            alpha_x=alpha_x, 
            alpha_y=alpha_y
        )

    def _compute_base_residuals_from_k(self, fields: PhysicsState):
        inv_K = (1.0 / self.Da_p) / (self.H**2)
        V_mag_k = jnp.sqrt(fields.u**2 + fields.v**2 + self.inf_small)
        inv_alpha = 1.0 / (fields.alpha + self.inf_small)

        # Eq 9a (Phương trình liên tục - Continuity)
        f_e = fields.u_x + fields.v_y + inv_alpha * (fields.alpha_x * fields.u + fields.alpha_y * fields.v)

        # TENSOR ỨNG SUẤT NHỚT ĐẦY ĐỦ (Không rút gọn thành Laplacian)
        # Khai triển div( grad(u) + grad(u)^T )
        visc_u = self.nu * (2.0 * fields.u_xx + fields.u_yy + fields.v_yx)
        visc_v = self.nu * (fields.u_yx + fields.v_xx + 2.0 * fields.v_yy)

        # Eq 9b - Base (Động lượng - Momentum)
        conv_u = 2.0 * fields.u * fields.u_x + fields.v * fields.u_y + fields.u * fields.v_y
        diff_u = - inv_alpha * (fields.alpha_x * fields.u**2 + fields.alpha_y * fields.u * fields.v)
        f_u_base = conv_u + fields.p_x - visc_u - diff_u 

        conv_v = fields.v * fields.u_x + fields.u * fields.v_x + 2.0 * fields.v * fields.v_y
        diff_v = - inv_alpha * (fields.alpha_x * fields.u * fields.v + fields.alpha_y * fields.v**2)
        f_v_base = conv_v + fields.p_y - visc_v - diff_v 
        
        alpha_m_local = fields.alpha * self.alpha_m_fluid + (1.0 - fields.alpha) * self.alpha_m_porous
        
        # 2. Đạo hàm của alpha_m_local theo không gian
        grad_alpha_m_x = (self.alpha_m_fluid - self.alpha_m_porous) * fields.alpha_x
        grad_alpha_m_y = (self.alpha_m_fluid - self.alpha_m_porous) * fields.alpha_y

        # 3. Số hạng khuếch tán nhiệt (Đã bao gồm sự biến thiên của alpha_m)
        diff_T = alpha_m_local * fields.laplacian_T + (grad_alpha_m_x * fields.T_x + grad_alpha_m_y * fields.T_y)

        # 4. Phương trình Nhiệt độ (Bảo toàn số hạng conv_T đã thêm T * div(u))
        conv_T = fields.u * fields.T_x + fields.v * fields.T_y #+ fields.T * (fields.u_x + fields.v_y)
        f_T_base = conv_T - diff_T
    
        #f_T_base = (fields.u * fields.T_x + fields.v * fields.T_y) - self.alpha_m_porous * fields.laplacian_T
        # Eq 10 - Drag (RHS)
        f_u_drag = -fields.gamma * ((self.nu * self.epsilon_p * inv_K) * fields.u + (self.epsilon_p**2 * self.F_eps * jnp.sqrt(inv_K)) * V_mag_k * fields.u)
        
        # (Buoyancy) hướng lên (+)
        f_v_drag = -fields.gamma * ((self.nu * self.epsilon_p * inv_K) * fields.v + (self.epsilon_p**2 * self.F_eps * jnp.sqrt(inv_K)) * V_mag_k * fields.v) \
                 + self.g_beta * (fields.T - self.T_ref) 

        # Trả về Residual = LHS - RHS_1 - RHS_2 = f_base - f_drag
        return f_u_base - f_u_drag, f_v_base - f_v_drag, f_e, f_T_base

    def r_net_base(self, params, x, y, **kwargs):
        fields = self._get_physics_components(params, x, y, **kwargs)
        return self._compute_base_residuals_from_k(fields)

    # --- 4. MODULE SIMPLE-PINN (VIRTUAL STENCIL) ---
    def _compute_simple_correction(self, params, prev_params, fields: PhysicsState, old: PhysicsState, 
                                   f_u_base, f_v_base, f_e_base, x, y, **kwargs):
        h = self.h_stencil
        inf_small = getattr(self, 'inf_small', 1e-8)
        
        # 1. Dự đoán Mạng Nơ-ron tại 4 điểm lân cận ảo (Virtual Stencil)
        u_E, v_E, p_E, _ = self.u_net(params, x + h, y, **kwargs)
        u_W, v_W, p_W, _ = self.u_net(params, x - h, y, **kwargs)
        u_N, v_N, p_N, _ = self.u_net(params, x, y + h, **kwargs)
        u_S, v_S, p_S, _ = self.u_net(params, x, y - h, **kwargs)

        old_u_E, old_v_E, old_p_E, _ = self.u_net(prev_params, x + h, y, **kwargs)
        old_u_W, old_v_W, old_p_W, _ = self.u_net(prev_params, x - h, y, **kwargs)
        old_u_N, old_v_N, old_p_N, _ = self.u_net(prev_params, x, y + h, **kwargs)
        old_u_S, old_v_S, old_p_S, _ = self.u_net(prev_params, x, y - h, **kwargs)

        # Định nghĩa mảng biến thiên (Correction) cục bộ giữa bước n và n-1
        delta_p_c = fields.p - old.p
        delta_u_c = fields.u - old.u
        delta_v_c = fields.v - old.v

        u_prime_E, u_prime_W = (u_E - old_u_E), (u_W - old_u_W)
        u_prime_N, u_prime_S = (u_N - old_u_N), (u_S - old_u_S)
        
        v_prime_E, v_prime_W = (v_E - old_v_E), (v_W - old_v_W)
        v_prime_N, v_prime_S = (v_N - old_v_N), (v_S - old_v_S)

        # 2. Xấp xỉ toán tử Laplacian Áp suất hiệu chỉnh (Nhân sẵn h^2)
        lap_p_h2 = (p_E - old_p_E) + (p_W - old_p_W) + (p_N - old_p_N) + (p_S - old_p_S) - 4.0 * delta_p_c

        # Phần đóng góp của nút lân cận (Neighbor Part) kế thừa từ TENSOR ỨNG SUẤT NHỚT ĐẦY ĐỦ của bạn
        # Phương trình U: nu * (2.0 * u_xx + u_yy) -> Phần nút lân cận = nu * (2.0 * (E + W) + (N + S))
        visc_u_nb = self.nu * (2.0 * (u_prime_E + u_prime_W) + (u_prime_N + u_prime_S))
        # Phương trình V: nu * (v_xx + 2.0 * v_yy) -> Phần nút lân cận = nu * ((E + W) + 2.0 * (N + S))
        visc_v_nb = self.nu * ((v_prime_E + v_prime_W) + 2.0 * (v_prime_N + v_prime_S))

        # Gradient áp suất hiệu chỉnh bằng sai phân trung tâm
        delta_px_fd = ((p_E - old_p_E) - (p_W - old_p_W)) / (2.0 * h)
        delta_py_fd = ((p_N - old_p_N) - (p_S - old_p_S)) / (2.0 * h)

        # 3. Đồng bộ hóa toán lý môi trường xốp (Darcy-Forchheimer)
        inv_K = (1.0 / self.Da_p) / (self.H**2)
        V_mag_k = jnp.sqrt(fields.u**2 + fields.v**2 + inf_small)

        # Tính toán Drag Lực cản toàn phần đúng theo quy ước _compute_base_residuals_from_k
        f_u_drag = -fields.gamma * ((self.nu * self.epsilon_p * inv_K) * fields.u + (self.epsilon_p**2 * self.F_eps * jnp.sqrt(inv_K)) * V_mag_k * fields.u)
        f_v_drag = -fields.gamma * ((self.nu * self.epsilon_p * inv_K) * fields.v + (self.epsilon_p**2 * self.F_eps * jnp.sqrt(inv_K)) * V_mag_k * fields.v) \
                   + self.g_beta * (fields.T - self.T_ref)

        # Khôi phục sai số dư TOÀN PHẦN (Total Strong Residual = LHS - RHS) làm số hạng nguồn
        f_u_total = f_u_base - f_u_drag
        f_v_total = f_v_base - f_v_drag

        # 4. Tính toán hệ số trung tâm a_P động (Dynamic Main Diagonal Coefficient)
        # - Thành phần nhớt anisotropy đóng góp: 6.0 * nu
        # - Thành phần lực cản Darcy + Forchheimer được tuyến tính hóa cục bộ theo dải mờ gamma
        drag_linearized = fields.gamma * ((self.nu * self.epsilon_p * inv_K) + (self.epsilon_p**2 * self.F_eps * jnp.sqrt(inv_K)) * V_mag_k)
        a_P = 6.0 * self.nu + drag_linearized * (h**2)

        # 5. Các số hạng R của bộ giải sửa sai liên kết SIMPLE (Đã nhân đồng bộ thể tích h^2)
        R_u = (visc_u_nb - (delta_px_fd * h**2) - (f_u_total * h**2)) / a_P
        R_v = (visc_v_nb - (delta_py_fd * h**2) - (f_v_total * h**2)) / a_P
        R_p = 0.25 * lap_p_h2 - (a_P * f_e_base) / 4.0

        # 6. Trả về toán tử Loss hiệu chỉnh (Residual Correction Terms)
        rc_u_term = delta_u_c - self.alpha_u * R_u
        rc_v_term = delta_v_c - self.alpha_u * R_v
        rc_p_term = delta_p_c - self.alpha_p * R_p

        return rc_u_term, rc_v_term, rc_p_term

    # --- 5: SCALE-PINN + GỌI SIMPLE-PINN ---
    def r_net_total(self, params, prev_params, pts_weights, x, y, **kwargs):
        fields = self._get_physics_components(params, x, y, **kwargs)

        old_fields = jax.lax.stop_gradient(self._get_physics_components(prev_params, x, y, **kwargs))
        f_u_base, f_v_base, f_e_base, f_T_base = self._compute_base_residuals_from_k(fields)

        # Scale-PINN
        S_u, S_v, S_p, S_T = 0.0, 0.0, 0.0, 0.0
        
        use_seq = getattr(self.config.ablation, 'use_sequential', True)
        use_lap = getattr(self.config.ablation, 'use_laplacian', True)
        tau_mode = getattr(self.config.ablation, 'tau_mode', 'adaptive')

        if use_seq:
            if tau_mode == 'adaptive':
                # Lấy trực tiếp từ pts_weights (Mặc định 1.0 nếu chưa khởi tạo)
                # Đảm bảo thứ tự này khớp với thứ tự return của r_net_base
                inv_tau_u = pts_weights.get('eqn_0', 1.0) 
                inv_tau_v = pts_weights.get('eqn_1', 1.0)
                inv_tau_p = pts_weights.get('eqn_2', 1.0)
                inv_tau_T = pts_weights.get('eqn_3', 1.0)
            else:
                # Scale-PINN gốc thường dùng chung 1 Tau_sc cho toàn hệ, 
                # hoặc bạn có thể tách ra nếu có tau_sc_u, tau_sc_v...
                inv_tau_u = inv_tau_v = inv_tau_T = (1.0 / self.tau_sc) 
                inv_tau_p = (1.0 / self.tau_sc) 

            # Hiệu chỉnh bậc 1 với Tau độc lập
            S_u = inv_tau_u * (fields.u - old_fields.u)
            S_v = inv_tau_v * (fields.v - old_fields.v)
            S_p = inv_tau_p * (fields.p - old_fields.p)
            S_T = inv_tau_T * (fields.T - old_fields.T)

            if use_lap:
                gamma_u = self.nu / self.epsilon_p  
                gamma_v = self.nu / self.epsilon_p
                gamma_T = self.alpha_m_porous

                if tau_mode == 'adaptive':
            
                    cfl = getattr(self.config.ablation, 'cfl_ratio', 1.0)
                    zeta_u = inv_tau_u * gamma_u * cfl
                    zeta_v = inv_tau_v * gamma_v * cfl
                    zeta_T = inv_tau_T * gamma_T * cfl
                else:
                    zeta_u = gamma_u / self.tau_alpha
                    zeta_v = gamma_v / self.tau_alpha
                    zeta_T = gamma_T / self.tau_alpha

                S_u -= zeta_u * (fields.laplacian_u - old_fields.laplacian_u)
                S_v -= zeta_v * (fields.laplacian_v - old_fields.laplacian_v)
                S_T -= zeta_T * (fields.laplacian_T - old_fields.laplacian_T)

        f_u_total = f_u_base + S_u
        f_v_total = f_v_base + S_v
        f_e_total = f_e_base + S_p
        f_T_total = f_T_base + S_T
        
        # SIMPLE-PINN
        if self.use_fvm_residual:
            rc_u, rc_v, rc_p = self._compute_simple_correction(
                params, prev_params, fields, old_fields, 
                f_u_base, f_v_base, f_e_base, x, y, **kwargs
            )
        else:
            rc_u, rc_v, rc_p = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

        return f_u_total, f_v_total, f_e_total, f_T_total, rc_u, rc_v, rc_p

    # --- 6. HÀM LOSSES BẮT BUỘC PHẢI GHI ĐÈ ĐỂ XỬ LÝ 7 OUTPUTS ---
    def losses(self, params, state, batch):
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))
        
        outputs = self.r_pred_total_fn(params, state.prev_params, state.pts_weights, *coords)
        f_u, f_v, f_e, f_T, rc_u, rc_v, rc_p = outputs
        
        loss_dict = {}
        # L2 Norm (MSE) cho Scale-PINN PDE
        loss_dict['eqn_0'] = jnp.mean(f_u ** 2) 
        loss_dict['eqn_1'] = jnp.mean(f_v ** 2) 
        loss_dict['eqn_2'] = jnp.mean(f_e ** 2) 
        loss_dict['eqn_3'] = jnp.mean(f_T ** 2) 
        
        # L1 Norm (MAE) cho SIMPLE-PINN Correction (Như thiết kế của bài báo)
        loss_dict['rc_u'] = jnp.mean(jnp.abs(rc_u))
        loss_dict['rc_v'] = jnp.mean(jnp.abs(rc_v))
        loss_dict['rc_p'] = jnp.mean(jnp.abs(rc_p))
        
        return loss_dict