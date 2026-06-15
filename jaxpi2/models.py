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
    
    elif arch_name == "cavity": 
        arch = archs.CavityNet(**config)
        
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
        if isinstance(batch, dict):
            coords = tuple(batch['eqn_fvm'][:, i] for i in range(batch['eqn_fvm'].shape[1]))
        else:
            coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        # CHẠY SONG SONG TRÊN LƯỚI: Mỗi hàm chỉ evaluate đúng 1 lần duy nhất!
        sols_curr, res_curr = self.r_sol_pred_nd_fn(state.params, *coords)
        sols_prev, res_prev = self.r_sol_pred_nd_fn(state.prev_params, *coords)
        _, res_init         = self.r_sol_pred_nd_fn(init_state.params, *coords)

        sol_diffs = sols_curr - sols_prev  # Shape: (Batch_size, 4)
        res_diffs = res_curr - res_prev    # Shape: (Batch_size, 4)

        # =========================================================================
        # ĐỒNG BỘ HÓA PHÂN TÁN (SHARD_MAP) - Rút gọn theo trục Batch (axis=0)
        # =========================================================================
        axis_name = "batch"

        global_loss0 = jax.lax.pmean(jnp.mean(res_init ** 2, axis=0), axis_name=axis_name)
        global_loss  = jax.lax.pmean(jnp.mean(res_curr ** 2, axis=0), axis_name=axis_name)

        global_norm_res = jnp.sqrt(jax.lax.psum(jnp.sum(res_diffs ** 2, axis=0), axis_name=axis_name))
        global_norm_sol = jnp.sqrt(jax.lax.psum(jnp.sum(sol_diffs ** 2, axis=0), axis_name=axis_name))

        # Tính toán hệ số suy giảm Cosine
        def cosine_decay_from_loss(losses, loss0, start_log_drop=3.0, end_log_drop=5.0, min_factor=0.1, eps=1e-8):
            log_drop = jnp.log10((loss0 + eps) / (losses + eps))
            p = jnp.clip((log_drop - start_log_drop) / (end_log_drop - start_log_drop), 0.0, 1.0)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + jnp.cos(jnp.pi * p))

        if self.config.pseudo_time.shrink.enabled:
            factors = cosine_decay_from_loss(
                global_loss, global_loss0,
                start_log_drop=self.config.pseudo_time.shrink.start_log_drop,
                end_log_drop=self.config.pseudo_time.shrink.end_log_drop,
                min_factor=self.config.pseudo_time.shrink.min_factor,
            )
        else:
            factors = 1.0

        # Trọng số đầu ra phẳng dạng vector (4,)
        weights = global_norm_res / (global_norm_sol + 1e-8) * factors
        #weights = jnp.clip(weights, a_min=1e-2, a_max=100.0)
        w_u = jnp.clip(weights[0], a_min=0.01, a_max=100.0)
        w_v = jnp.clip(weights[1], a_min=0.01, a_max=100.0)
        w_T = jnp.clip(weights[3], a_min=0.01, a_max=100.0)

        # 2. "CẦM CƯƠNG" ÁP SUẤT: Giới hạn trần rất gắt hoặc nhân hệ số làm mềm
        # Không bao giờ được để w_p (tức 1/tau_p) vượt quá 5.0 hoặc 10.0
        #w_p = jnp.clip(weights[2] * 0.1, a_min=0.01, a_max=10.0) 
        max_w_uv = jnp.maximum(w_u, w_v) 
        w_p = 0.5 * max_w_uv
        # Đóng gói trả về
        weights_final = jnp.array([w_u, w_v, w_p, w_T])
        weights_final = lax.stop_gradient(weights_final)
        weights = lax.stop_gradient(weights)

        keys = list(state.pts_weights.keys())
        return dict(zip(keys, weights_final))

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
        def update_pts_weights(state, init_state, batch):
            pts_weights = self.compute_pts_weights(state, init_state, batch)
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

    @partial(jit, static_argnums=(0,))
    def compute_residual_losses(self, params, state, batch):
        keys = list(state.pts_weights.keys())
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))
        res_total = jnp.stack(self.r_pred_total_fn(params, state.prev_params, state.pts_weights, *coords))

        if res_total.ndim == 1:
            res_total = res_total[None, :]

        per_key_losses = jnp.mean(res_total ** 2, axis=1)
        return dict(zip(keys, per_key_losses))


class BaseHeatPorousScalePINN(ForwardBVP):
    def __init__(self, config, lr, tx, arch, state, lb, ub):
        super().__init__(config, lr, tx, arch, state)
        
        self.lb = lb
        self.ub = ub
        
        self.tau_sc = config.physics.get('tau_sc', 0.5)
        self.tau_alpha = config.physics.get('tau_alpha', 0.5)
        self.inf_small = 1e-6#jnp.finfo(jnp.float32).eps
        ibm_config = config.get('ibm', {})
        self.use_ibm = ibm_config.get('mode')
        self.r_sol_pred_nd_fn = jax.vmap(self.r_sol_net_nd, (None, 0, 0))

        self.use_fvm_residual = config.fvm_setup.get('mode')
        if self.use_fvm_residual:
            self.h_stencil = config.fvm_setup.get('h_stencil')
            self.alpha_u = config.fvm_setup.get('alpha_u')
            self.alpha_p = config.fvm_setup.get('alpha_p')
        
        if self.use_ibm:
            self.eta_diffuse = ibm_config.get('eta_diffuse', 30.0)
            self.layer_x_center = ibm_config.get('layer_x_center', 0.5)
            self.layer_width = ibm_config.get('layer_width', 1.0 / 3.0)
        def r_net_base_nd_wrapper(params, x, y, **kwargs):
            _, residuals = self._compute_fvm_base_residuals(params, x, y, **kwargs)
            return residuals
            
        self.r_pred_base_fn = jax.vmap(r_net_base_nd_wrapper, in_axes=(None, 0, 0))
        self.r_pred_total_fn = jax.vmap(self.r_net_nd, in_axes=(None, None, None, 0, 0))

    def u_net(self, params, x, y, **kwargs):
        outputs = self.arch.apply(params, jnp.concatenate([x, y], axis=-1))
        return outputs[..., 0:1], outputs[..., 1:2], outputs[..., 2:3], outputs[..., 3:4]
    
    def r_sol_net_nd(self, params, x, y, **kwargs):
        """Wrap output để đưa vào compute_pts_weights"""
        sols_tuple, res_tuple = self._compute_fvm_base_residuals(params, x, y, **kwargs)
        
        # Trả về shape (4,) cho sols và (4,) cho residuals
        return jnp.array(sols_tuple), jnp.array(res_tuple)

    def _get_geometry_alpha(self, x_in, y_in):
        if self.use_ibm:
            phi = self.sdf_fn(x_in, y_in)
            return (1.0 + self.epsilon_p) / 2.0 + (1.0 - self.epsilon_p) / 2.0 * jnp.tanh(self.eta_diffuse * phi)
        else:
            return self.epsilon_p + 0.0 * x_in

    def _get_geometry_gamma(self, x_in, y_in):
        if self.use_ibm:
            phi = self.sdf_fn(x_in, y_in)
            return 0.5 * (1.0 - jnp.tanh(self.eta_diffuse * phi))
        else:
            return 1.0 + 0.0 * x_in


    def _compute_fvm_base_residuals(self, params, x, y, **kwargs):
        """Tính toán nghiệm tại P và thặng dư nền theo chuẩn Simplified FVM"""
        h = self.h_stencil
        inf_small = self.inf_small
        
        xs = jnp.array([x, x + h, x - h, x, x, x + h/2, x - h/2, x, x])
        ys = jnp.array([y, y, y, y + h, y - h, y, y, y + h/2, y - h/2])
        
        def get_vals(p, x_in, y_in):
            return self.u_net(p, x_in, y_in, **kwargs)
            
        u_raw, v_raw, p_raw, T_raw = jax.vmap(get_vals, in_axes=(None, 0, 0))(params, xs, ys)
        u_P, v_P, p_P, T_P = u_raw[0], v_raw[0], p_raw[0], T_raw[0]

        # --- ÁP DỤNG LSA-PINN (Giữ nguyên y hệt code cũ) ---
        is_E_ext, is_W_ext = (x + h) > self.L, (x - h) < 0.0
        is_N_ext, is_S_ext = (y + h) > self.H, (y - h) < 0.0
        is_e_ext, is_w_ext = (x + h/2) >= self.L, (x - h/2) <= 0.0
        is_n_ext, is_s_ext = (y + h/2) >= self.H, (y - h/2) <= 0.0

        ratio_W = h / jnp.maximum(x, inf_small)
        ratio_E = h / jnp.maximum(self.L - x, inf_small)
        ratio_S = h / jnp.maximum(y, inf_small)
        ratio_N = h / jnp.maximum(self.H - y, inf_small)

        # Nội suy tuyến tính động cho Vận tốc (Điều kiện No-slip: u_BC = 0, v_BC = 0)
        u_E = jnp.where(is_E_ext, u_P + (0.0 - u_P) * ratio_E, u_raw[1]) 
        v_E = jnp.where(is_E_ext, v_P + (0.0 - v_P) * ratio_E, v_raw[1])
        u_W = jnp.where(is_W_ext, u_P + (0.0 - u_P) * ratio_W, u_raw[2]) 
        v_W = jnp.where(is_W_ext, v_P + (0.0 - v_P) * ratio_W, v_raw[2])
        u_N = jnp.where(is_N_ext, u_P + (0.0 - u_P) * ratio_N, u_raw[3]) 
        v_N = jnp.where(is_N_ext, v_P + (0.0 - v_P) * ratio_N, v_raw[3])
        u_S = jnp.where(is_S_ext, u_P + (0.0 - u_P) * ratio_S, u_raw[4]) 
        v_S = jnp.where(is_S_ext, v_P + (0.0 - v_P) * ratio_S, v_raw[4])

        # Nội suy tuyến tính động cho Nhiệt độ (Vách Tây nóng T=1, Vách Đông lạnh T=0)
        T_W = jnp.where(is_W_ext, T_P + (1.0 - T_P) * ratio_W, T_raw[2])
        T_E = jnp.where(is_E_ext, T_P + (0.0 - T_P) * ratio_E, T_raw[1])
        
        # Các vách Nam/Bắc đoạn nhiệt (Neumann: dT/dy = 0 -> Ghost cell bằng chính nó, giữ nguyên)
        T_N = jnp.where(is_N_ext, T_P, T_raw[3])
        T_S = jnp.where(is_S_ext, T_P, T_raw[4])

        p_E = jnp.where(is_E_ext, p_P, p_raw[1])
        p_W = jnp.where(is_W_ext, p_P, p_raw[2])
        p_N = jnp.where(is_N_ext, p_P, p_raw[3])
        p_S = jnp.where(is_S_ext, p_P, p_raw[4])
        # --- SỬA THÀNH ĐOẠN ĐỘC LẬP CHUẨN FVM ---
        u_e = jnp.where(is_e_ext, 0.0, u_raw[5])
        u_w = jnp.where(is_w_ext, 0.0, u_raw[6])
        v_n = jnp.where(is_n_ext, 0.0, v_raw[7])
        v_s = jnp.where(is_s_ext, 0.0, v_raw[8])

        u_n_face = jnp.where(is_n_ext, 0.0, u_raw[7])
        u_s_face = jnp.where(is_s_ext, 0.0, u_raw[8])
        v_e_face = jnp.where(is_e_ext, 0.0, v_raw[5])
        v_w_face = jnp.where(is_w_ext, 0.0, v_raw[6])
        
        T_e_face = jnp.where(is_e_ext, 0.0, T_raw[5])
        T_w_face = jnp.where(is_w_ext, 1.0, T_raw[6])
        T_n_face = jnp.where(is_n_ext, T_P, T_raw[7]) 
        T_s_face = jnp.where(is_s_ext, T_P, T_raw[8])

        # --- TÍNH TOÁN FVM ---
        conv_u_fvm = (u_e * u_e - u_w * u_w) / h + (v_n * u_n_face - v_s * u_s_face) / h
        conv_v_fvm = (u_e * v_e_face - u_w * v_w_face) / h + (v_n * v_n - v_s * v_s) / h
        conv_T_fvm = (u_e * T_e_face - u_w * T_w_face) / h + (v_n * T_n_face - v_s * T_s_face) / h

        p_x_fvm = (p_E - p_W) / (2.0 * h)
        p_y_fvm = (p_N - p_S) / (2.0 * h)
        
        visc_u = jnp.sqrt(self.Pr / self.Ra) * (u_E + u_W + u_N + u_S - 4.0 * u_P) / (h**2)
        visc_v = jnp.sqrt(self.Pr / self.Ra) * (v_E + v_W + v_N + v_S - 4.0 * v_P) / (h**2)
        
        eps_local = self._get_geometry_alpha(x, y) 
        gamma_local = self._get_geometry_gamma(x, y) 
        inv_eps = 1.0 / jnp.maximum(eps_local, inf_small)
        alpha_m_fluid = 1.0 / jnp.sqrt(self.Pr * self.Ra)
        alpha_m_porous = 1.0 / jnp.sqrt(self.Pr * self.Ra)#need to modify when have both fluid and porous 
        alpha_m_local = eps_local * alpha_m_fluid + (1.0 - eps_local) * alpha_m_porous
        diff_T = alpha_m_local * (T_E + T_W + T_N + T_S - 4.0 * T_P) / (h**2)

        f_e_fvm = (u_e - u_w) / h + (v_n - v_s) / h

        inv_K = (1.0 / self.Da_p)
        V_mag_k = jnp.sqrt(u_P**2 + v_P**2 + self.inf_small)
        drag_linear = jnp.sqrt(self.Pr / self.Ra) * eps_local * inv_K
        drag_nonlin = eps_local * self.F_eps * jnp.sqrt(inv_K)
        f_u_drag = -gamma_local * (drag_linear * u_P + drag_nonlin * V_mag_k * u_P)
        f_v_drag = -gamma_local * (drag_linear * v_P + drag_nonlin * V_mag_k * v_P) + eps_local * (T_P - self.T_ref) 

        # --- TRẢ VỀ NGHIỆM VÀ THẶNG DƯ (Chưa có PTS) ---
        r_n_u = inv_eps * conv_u_fvm + eps_local * p_x_fvm - visc_u - f_u_drag
        r_n_v = inv_eps * conv_v_fvm + eps_local * p_y_fvm - visc_v - f_v_drag
        r_n_T = conv_T_fvm - diff_T

        sols = (u_P, v_P, p_P, T_P)
        residuals = (r_n_u, r_n_v, f_e_fvm, r_n_T)
        
        return sols, residuals
    
    def r_net_nd(self, params, prev_params, pts_weights, x, y, **kwargs):
        # Trích xuất trọng số thời gian ảo
        inv_tau_u = pts_weights.get('eqn_0', 1.0) 
        inv_tau_v = pts_weights.get('eqn_1', 1.0)
        inv_tau_p = pts_weights.get('eqn_2', 1.0)
        inv_tau_T = pts_weights.get('eqn_3', 1.0)

        # Lấy nghiệm và thặng dư FVM nền ở Epoch hiện tại
        sols_curr, res_base = self._compute_fvm_base_residuals(params, x, y, **kwargs)
        
        # Lấy nghiệm ở Epoch trước (dừng gradient) để tính delta u
        sols_prev, _ = jax.lax.stop_gradient(self._compute_fvm_base_residuals(prev_params, x, y, **kwargs))

        # Tính toán độ lệch nghiệm
        du_P = sols_curr[0] - sols_prev[0]
        dv_P = sols_curr[1] - sols_prev[1]
        dp_P = sols_curr[2] - sols_prev[2]
        dT_P = sols_curr[3] - sols_prev[3]

        # Phương trình PTS hoàn chỉnh: (u^k - u^{k-1})/\tau + R_FVM = 0
        fvm_eqn_0 = res_base[0] + inv_tau_u * du_P
        fvm_eqn_1 = res_base[1] + inv_tau_v * dv_P
        fvm_eqn_2 = res_base[2] + inv_tau_p * dp_P
        fvm_eqn_3 = res_base[3] + inv_tau_T * dT_P

        return fvm_eqn_0, fvm_eqn_1, fvm_eqn_2, fvm_eqn_3




