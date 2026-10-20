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

from jaxpi import archs, fvm
from jaxpi.utils import flatten_pytree

from soap_jax import soap
from typing import NamedTuple
from jaxpi import fvm


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

        sols_curr, res_curr = self.r_sol_pred_nd_fn(state.params, *coords)
        sols_prev, res_prev = self.r_sol_pred_nd_fn(state.prev_params, *coords)
        _, res_init         = self.r_sol_pred_nd_fn(init_state.params, *coords)

        sol_diffs = sols_curr - sols_prev  # Shape: (Batch_size, 4)
        res_diffs = res_curr - res_prev    # Shape: (Batch_size, 4)

        axis_name = "batch"
        global_loss0 = jax.lax.pmean(jnp.mean(res_init ** 2, axis=0), axis_name=axis_name)
        global_loss  = jax.lax.pmean(jnp.mean(res_curr ** 2, axis=0), axis_name=axis_name)

        global_norm_res = jnp.sqrt(jax.lax.psum(jnp.sum(res_diffs ** 2, axis=0), axis_name=axis_name))
        global_norm_sol = jnp.sqrt(jax.lax.psum(jnp.sum(sol_diffs ** 2, axis=0), axis_name=axis_name))

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

        weights = global_norm_res / (global_norm_sol + 1e-8) * factors

        w_u = jnp.clip(weights[0], 0.01, 100.0)
        w_v = jnp.clip(weights[1], 0.01, 100.0)
        w_T = jnp.clip(weights[3], 0.01, 100.0)
        max_w_uv = jnp.maximum(w_u, w_v) 
        w_p = 0.5 * max_w_uv
        
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
        # pde_bc_grads = {k: v for k, v in grads.items() if not k.startswith('rc_')}

        # # 3. Tính chuẩn gradient (Grad Norm) cho PDE và BC
        # grad_norm_dict = {}
        # for key, value in pde_bc_grads.items():
        #     flattened_grad = flatten_pytree(value)
        #     grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

        # # 4. Tính trọng số động cho các thặng dư PDE & BC
        # mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
        # w_pde_bc = tree_map(lambda x: (mean_grad_norm / (x + 1e-5 * mean_grad_norm)), grad_norm_dict)

        # # 5. Cập nhật trọng số động mới vào dict trọng số của state
        # new_weights = {**state.loss_weights, **w_pde_bc}
        # return new_weights

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

        self.inf_small = 1e-6
        ibm_config = config.get('ibm', {})
        self.use_ibm = ibm_config.get('mode')
        self.r_sol_pred_nd_fn = jax.vmap(self.r_sol_net_nd, (None, 0, 0))
        self.h_stencil= config.fvm_setup.get('h_stencil')
        
        
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
        """
        =================================================================================
        Simple-PINN for GOVERNING EQUATIONS FOR FLOW AND HEAT TRANSFER THROUGH POROUS MEDIA 
        https://doi.org/10.1016/S0017-9310(97)00008-2
        =================================================================================

        1. Continuity Equation:
           du/dx + dv/dy = 0

        2. x-Momentum Equation:
           (1/eps) * du/dt + (1/eps) * { u * d/dx(u/eps) + v * d/dy(u/eps) }
               = - (1/eps) * d(eps * p_f)/dx - (Pr / Da) * u
                 - (1.75 / sqrt(150)) * ((u^2 + v^2)^(1/2) / sqrt(Da)) * (u / eps^(3/2))
                 + (Pr / eps) * (d^2 u / dx^2 + d^2 u / dy^2)

        3. y-Momentum Equation:
           (1/eps) * dv/dt + (1/eps) * { u * d/dx(v/eps) + v * d/dy(v/eps) }
               = - (1/eps) * d(eps * p_f)/dy - (Pr / Da) * v
                 - (1.75 / sqrt(150)) * ((u^2 + v^2)^(1/2) / sqrt(Da)) * (v / eps^(3/2))
                 + (Pr / eps) * (d^2 v / dx^2 + d^2 v / dy^2)
                 + Ra * Pr * T

        4. Energy Equation (Eq. 15):
           sigma * dT/dt + u * dT/dx + v * dT/dy = d^2 T / dx^2 + d^2 T / dy^2
        =================================================================================
        """
        """Tính toán nghiệm, Stencil, Faces và thặng dư FVM qua module fvm.py"""
        h = self.h_stencil
        inf_small = self.inf_small
        # 1. Stencil 5 điểm tâm ô (Cell Centers)
        xs = jnp.array([x, x + h, x - h, x, x])
        ys = jnp.array([y, y, y, y + h, y - h])


        def get_vals(p, x_in, y_in):
            return self.u_net(p, x_in, y_in, **kwargs)


        # Chỉ forward mạng qua 5 điểm
        u_raw, v_raw, p_raw, T_raw = jax.vmap(get_vals, in_axes=(None, 0, 0))(
            params, xs, ys
        )

        u_P, v_P, p_P, T_P = u_raw[0], v_raw[0], p_raw[0], T_raw[0]
        u_E_raw, v_E_raw, p_E_raw, T_E_raw = u_raw[1], v_raw[1], p_raw[1], T_raw[1]
        u_W_raw, v_W_raw, p_W_raw, T_W_raw = u_raw[2], v_raw[2], p_raw[2], T_raw[2]
        u_N_raw, v_N_raw, p_N_raw, T_N_raw = u_raw[3], v_raw[3], p_raw[3], T_raw[3]
        u_S_raw, v_S_raw, p_S_raw, T_S_raw = u_raw[4], v_raw[4], p_raw[4], T_raw[4]
        # Xử lý biên LSA cho tâm ô lân cận
        is_E_ext, is_W_ext = (x + h) > self.L, (x - h) < 0.0
        is_N_ext, is_S_ext = (y + h) > self.H, (y - h) < 0.0

        ratio_W = h / jnp.maximum(x, inf_small)
        ratio_E = h / jnp.maximum(self.L - x, inf_small)
        ratio_S = h / jnp.maximum(y, inf_small)
        ratio_N = h / jnp.maximum(self.H - y, inf_small)

        u_E = jnp.where(is_E_ext, u_P + (0.0 - u_P) * ratio_E, u_E_raw)
        v_E = jnp.where(is_E_ext, v_P + (0.0 - v_P) * ratio_E, v_E_raw)
        u_W = jnp.where(is_W_ext, u_P + (0.0 - u_P) * ratio_W, u_W_raw)
        v_W = jnp.where(is_W_ext, v_P + (0.0 - v_P) * ratio_W, v_W_raw)
        u_N = jnp.where(is_N_ext, u_P + (0.0 - u_P) * ratio_N, u_N_raw)
        v_N = jnp.where(is_N_ext, v_P + (0.0 - v_P) * ratio_N, v_N_raw)
        u_S = jnp.where(is_S_ext, u_P + (0.0 - u_P) * ratio_S, u_S_raw)
        v_S = jnp.where(is_S_ext, v_P + (0.0 - v_P) * ratio_S, v_S_raw)

        T_W = jnp.where(is_W_ext, T_P + (1.0 - T_P) * ratio_W, T_W_raw)
        T_E = jnp.where(is_E_ext, T_P + (0.0 - T_P) * ratio_E, T_E_raw)
        T_N = jnp.where(is_N_ext, T_P, T_N_raw)
        T_S = jnp.where(is_S_ext, T_P, T_S_raw)

        p_E = jnp.where(is_E_ext, p_P, p_E_raw)
        p_W = jnp.where(is_W_ext, p_P, p_W_raw)
        p_N = jnp.where(is_N_ext, p_P, p_N_raw)
        p_S = jnp.where(is_S_ext, p_P, p_S_raw)

        # 3.1. Tái dựng vận tốc đối lưu tại các mặt kiểm soát (Face Fluxes)
        u_e_face = 0.5 * (u_P + u_E)
        u_w_face = 0.5 * (u_P + u_W)
        v_n_face = 0.5 * (v_P + v_N)
        v_s_face = 0.5 * (v_P + v_S)

        # 3.2. Thông lượng đối lưu qua các mặt (Convective Fluxes) cho từng biến
        # Flux = V_normal * phi
        flux_u_e = u_e_face * (0.5 * (u_P + u_E))
        flux_u_w = u_w_face * (0.5 * (u_P + u_W))
        flux_u_n = v_n_face * (0.5 * (u_P + u_N))
        flux_u_s = v_s_face * (0.5 * (u_P + u_S))

        flux_v_e = u_e_face * (0.5 * (v_P + v_E))
        flux_v_w = u_w_face * (0.5 * (v_P + v_W))
        flux_v_n = v_n_face * (0.5 * (v_P + v_N))
        flux_v_s = v_s_face * (0.5 * (v_P + v_S))

        flux_T_e = u_e_face * (0.5 * (T_P + T_E))
        flux_T_w = u_w_face * (0.5 * (T_P + T_W))
        flux_T_n = v_n_face * (0.5 * (T_P + T_N))
        flux_T_s = v_s_face * (0.5 * (T_P + T_S))

        # Đạo hàm đối lưu bậc O(1): (F_e - F_w)/h + (F_n - F_s)/h
        conv_u = (flux_u_e - flux_u_w) / h + (flux_u_n - flux_u_s) / h
        conv_v = (flux_v_e - flux_v_w) / h + (flux_v_n - flux_v_s) / h
        conv_T = (flux_T_e - flux_T_w) / h + (flux_T_n - flux_T_s) / h

        # 3.3. Gradient áp suất bậc O(1)
        p_x = (p_E - p_W) / (2.0 * h)
        p_y = (p_N - p_S) / (2.0 * h)

        # 3.4. Khuếch tán bậc O(1) (Standard 5-point Laplacian / h^2)
        visc_coeff = jnp.sqrt(self.Pr / self.Ra)
        diff_T_coeff = 1.0 / jnp.sqrt(self.Pr * self.Ra)

        laplace_u = (u_E + u_W + u_N + u_S - 4.0 * u_P) / (h**2)
        laplace_v = (v_E + v_W + v_N + v_S - 4.0 * v_P) / (h**2)
        laplace_T = (T_E + T_W + T_N + T_S - 4.0 * T_P) / (h**2)

        # 3.5. Số hạng nguồn môi trường xốp và lực nổi (Bậc O(1))
        eps_local = self._get_geometry_alpha(x, y)
        gamma_local = self._get_geometry_gamma(x, y)
        inv_eps = 1.0 / jnp.maximum(eps_local, inf_small)

        inv_K = 1.0 / self.Da_p
        V_mag_P = jnp.sqrt(u_P**2 + v_P**2 + inf_small)
        drag_linear = visc_coeff * eps_local * inv_K
        drag_nonlin = eps_local * self.F_eps * jnp.sqrt(inv_K)

        f_u_drag = gamma_local * (drag_linear + drag_nonlin * V_mag_P) * u_P
        f_v_drag = gamma_local * (drag_linear + drag_nonlin * V_mag_P) * v_P
        f_buoyancy = eps_local * (T_P - self.T_ref)

        # 3.6. Thặng dư phương trình vi phân cơ sở r (Chuẩn O(1))
        r_n_u = inv_eps * conv_u + p_x - (visc_coeff * inv_eps) * laplace_u + f_u_drag
        r_n_v = (
            inv_eps * conv_v
            + p_y
            - (visc_coeff * inv_eps) * laplace_v
            + f_v_drag
            - f_buoyancy
        )
        r_n_T = conv_T - diff_T_coeff * laplace_T
        r_e_c = (u_e_face - u_w_face) / h + (v_n_face - v_s_face) / h

        sols = (u_P, v_P, p_P, T_P)
        residuals = (r_n_u, r_n_v, r_e_c, r_n_T)
        stencils = (
            (u_E, u_W, u_N, u_S),
            (v_E, v_W, v_N, v_S),
            (p_E, p_W, p_N, p_S),
            (T_E, T_W, T_N, T_S),
        )
        temp = 0 
        return sols, residuals, stencils, temp
        
    # def _compute_fvm_base_residuals(self, params, x, y, **kwargs):
    #     """
    #     =================================================================================
    #     Simple-PINN for GOVERNING EQUATIONS FOR FLOW AND HEAT TRANSFER THROUGH POROUS MEDIA 
    #     https://doi.org/10.1016/S0017-9310(97)00008-2
    #     =================================================================================

    #     1. Continuity Equation:
    #        du/dx + dv/dy = 0

    #     2. x-Momentum Equation:
    #        (1/eps) * du/dt + (1/eps) * { u * d/dx(u/eps) + v * d/dy(u/eps) }
    #            = - (1/eps) * d(eps * p_f)/dx - (Pr / Da) * u
    #              - (1.75 / sqrt(150)) * ((u^2 + v^2)^(1/2) / sqrt(Da)) * (u / eps^(3/2))
    #              + (Pr / eps) * (d^2 u / dx^2 + d^2 u / dy^2)

    #     3. y-Momentum Equation:
    #        (1/eps) * dv/dt + (1/eps) * { u * d/dx(v/eps) + v * d/dy(v/eps) }
    #            = - (1/eps) * d(eps * p_f)/dy - (Pr / Da) * v
    #              - (1.75 / sqrt(150)) * ((u^2 + v^2)^(1/2) / sqrt(Da)) * (v / eps^(3/2))
    #              + (Pr / eps) * (d^2 v / dx^2 + d^2 v / dy^2)
    #              + Ra * Pr * T

    #     4. Energy Equation (Eq. 15):
    #        sigma * dT/dt + u * dT/dx + v * dT/dy = d^2 T / dx^2 + d^2 T / dy^2
    #     =================================================================================
    #     """
    #     """Tính toán nghiệm, Stencil, Faces và thặng dư FVM qua module fvm.py"""
    #     h = self.h_stencil
    #     inf_small = self.inf_small
        
    #     xs = jnp.array([x, x + h, x - h, x, x, x + h/2, x - h/2, x, x])
    #     ys = jnp.array([y, y, y, y + h, y - h, y, y, y + h/2, y - h/2])
        
    #     def get_vals(p, x_in, y_in):
    #         return self.u_net(p, x_in, y_in, **kwargs)
            
    #     u_raw, v_raw, p_raw, T_raw = jax.vmap(get_vals, in_axes=(None, 0, 0))(params, xs, ys)
    #     u_P, v_P, p_P, T_P = u_raw[0], v_raw[0], p_raw[0], T_raw[0]

    #     # u_P, v_P, p_P, T_P = u_raw[0], v_raw[0], p_raw[0], T_raw[0]
    #     # u_E, v_E, p_E, T_E = u_raw[1], v_raw[1], p_raw[1], T_raw[1]
    #     # u_W, v_W, p_W, T_W = u_raw[2], v_raw[2], p_raw[2], T_raw[2]
    #     # u_N, v_N, p_N, T_N = u_raw[3], v_raw[3], p_raw[3], T_raw[3]
    #     # u_S, v_S, p_S, T_S = u_raw[4], v_raw[4], p_raw[4], T_raw[4]

    #     # u_e, v_e_face, p_e, T_e_face = u_raw[5], v_raw[5], p_raw[5], T_raw[5]
    #     # u_w, v_w_face, p_w, T_w_face = u_raw[6], v_raw[6], p_raw[6], T_raw[6]
    #     # u_n_face, v_n, p_n, T_n_face = u_raw[7], v_raw[7], p_raw[7], T_raw[7]
    #     # u_s_face, v_s, p_s, T_s_face = u_raw[8], v_raw[8], p_raw[8], T_raw[8]

    #     # --- 1. LSA XỬ LÝ BIÊN ---
    #     is_E_ext, is_W_ext = (x + h) > self.L, (x - h) < 0.0
    #     is_N_ext, is_S_ext = (y + h) > self.H, (y - h) < 0.0
    #     is_e_ext, is_w_ext = (x + h/2) >= self.L, (x - h/2) <= 0.0
    #     is_n_ext, is_s_ext = (y + h/2) >= self.H, (y - h/2) <= 0.0

    #     ratio_W = h / jnp.maximum(x, inf_small)
    #     ratio_E = h / jnp.maximum(self.L - x, inf_small)
    #     ratio_S = h / jnp.maximum(y, inf_small)
    #     ratio_N = h / jnp.maximum(self.H - y, inf_small)

    #     # Tâm ô lân cận E, W, N, S
    #     u_E = jnp.where(is_E_ext, u_P + (0.0 - u_P) * ratio_E, u_raw[1]) 
    #     v_E = jnp.where(is_E_ext, v_P + (0.0 - v_P) * ratio_E, v_raw[1])
    #     u_W = jnp.where(is_W_ext, u_P + (0.0 - u_P) * ratio_W, u_raw[2]) 
    #     v_W = jnp.where(is_W_ext, v_P + (0.0 - v_P) * ratio_W, v_raw[2])
    #     u_N = jnp.where(is_N_ext, u_P + (0.0 - u_P) * ratio_N, u_raw[3]) 
    #     v_N = jnp.where(is_N_ext, v_P + (0.0 - v_P) * ratio_N, v_raw[3])
    #     u_S = jnp.where(is_S_ext, u_P + (0.0 - u_P) * ratio_S, u_raw[4]) 
    #     v_S = jnp.where(is_S_ext, v_P + (0.0 - v_P) * ratio_S, v_raw[4])

    #     T_W = jnp.where(is_W_ext, T_P + (1.0 - T_P) * ratio_W, T_raw[2])
    #     T_E = jnp.where(is_E_ext, T_P + (0.0 - T_P) * ratio_E, T_raw[1])
    #     T_N = jnp.where(is_N_ext, T_P, T_raw[3])
    #     T_S = jnp.where(is_S_ext, T_P, T_raw[4])

    #     p_E = jnp.where(is_E_ext, p_P, p_raw[1])
    #     p_W = jnp.where(is_W_ext, p_P, p_raw[2])
    #     p_N = jnp.where(is_N_ext, p_P, p_raw[3])
    #     p_S = jnp.where(is_S_ext, p_P, p_raw[4])

    #     # Mặt kiểm soát e, w, n, s
    #     u_e = jnp.where(is_e_ext, 0.0, u_raw[5])
    #     u_w = jnp.where(is_w_ext, 0.0, u_raw[6])
    #     v_n = jnp.where(is_n_ext, 0.0, v_raw[7])
    #     v_s = jnp.where(is_s_ext, 0.0, v_raw[8])

    #     u_n_face = jnp.where(is_n_ext, 0.0, u_raw[7])
    #     u_s_face = jnp.where(is_s_ext, 0.0, u_raw[8])
    #     v_e_face = jnp.where(is_e_ext, 0.0, v_raw[5])
    #     v_w_face = jnp.where(is_w_ext, 0.0, v_raw[6])
        
    #     p_e = jnp.where(is_e_ext, p_P, p_raw[5])
    #     p_w = jnp.where(is_w_ext, p_P, p_raw[6])
    #     p_n = jnp.where(is_n_ext, p_P, p_raw[7])
    #     p_s = jnp.where(is_s_ext, p_P, p_raw[8])

    #     T_e_face = jnp.where(is_e_ext, 0.0, T_raw[5])
    #     T_w_face = jnp.where(is_w_ext, 1.0, T_raw[6])
    #     T_n_face = jnp.where(is_n_ext, T_P, T_raw[7]) 
    #     T_s_face = jnp.where(is_s_ext, T_P, T_raw[8])

    #     # --- 2. SỐ HẠNG ĐỐI LƯU & KHUẾCH TÁN (GỌI fvm.py) ---
    #     # Đối lưu: div_convective_flux_2d trả về h * (...), chia cho h để ra bậc O(1)
    #     conv_u = fvm.div_convective_flux_2d(u_e, u_w, v_n, v_s, u_e, u_w, u_n_face, u_s_face, h) / h**2
    #     conv_v = fvm.div_convective_flux_2d(u_e, u_w, v_n, v_s, v_e_face, v_w_face, v_n, v_s, h) / h**2
    #     conv_T = fvm.div_convective_flux_2d(u_e, u_w, v_n, v_s, T_e_face, T_w_face, T_n_face, T_s_face, h) / h**2

    #     # Gradient áp suất vi phân bậc O(1)
    #     p_x = (p_E - p_W) / (2.0 * h)
    #     p_y = (p_N - p_S) / (2.0 * h)

    #     # Khuếch tán: laplacian_2d trả về \Delta_h, chia cho h^2 để ra bậc O(1)
    #     visc_coeff = jnp.sqrt(self.Pr / self.Ra)
    #     laplace_u = fvm.laplacian_2d(u_P, u_E, u_W, u_N, u_S) / h**2 
    #     laplace_v = fvm.laplacian_2d(v_P, v_E, v_W, v_N, v_S) / h**2

    #     diff_T_coeff = 1.0 / jnp.sqrt(self.Pr * self.Ra)
    #     laplace_T = fvm.laplacian_2d(T_P, T_E, T_W, T_N, T_S) / h**2

    #     # --- 3. MÔI TRƯỜNG XỐP & LỰC THỂ TÍCH ---
    #     eps_local = self._get_geometry_alpha(x, y)
    #     gamma_local = self._get_geometry_gamma(x, y)
    #     inv_eps = 1.0 / jnp.maximum(eps_local, inf_small)

    #     inv_K = 1.0 / self.Da_p
    #     V_mag_k = fvm.velocity_magnitude(u_P, v_P, inf_small)
    #     drag_linear = visc_coeff * eps_local * inv_K
    #     drag_nonlin = eps_local * self.F_eps * jnp.sqrt(inv_K)

    #     f_u_drag = gamma_local * (drag_linear + drag_nonlin * V_mag_k) * u_P #* h**2
    #     f_v_drag = gamma_local * (drag_linear + drag_nonlin * V_mag_k) * v_P #* h**2
    #     f_buoyancy = eps_local * (T_P - self.T_ref) # * h**2

    #     # --- 4. THẶNG DƯ PDE CƠ SỞ ---
    #     r_n_u = inv_eps * conv_u + p_x - (visc_coeff * inv_eps) * laplace_u + f_u_drag
    #     r_n_v = inv_eps * conv_v + p_y - (visc_coeff * inv_eps) * laplace_v + f_v_drag - f_buoyancy
        
    #     # Phương trình liên tục vi phân chuẩn O(1)
    #     r_e_c = fvm.div_velocity_2d(u_e, u_w, v_n, v_s) / h
        
    #     r_n_T = conv_T - diff_T_coeff * laplace_T

    #     sols = (u_P, v_P, p_P, T_P)
    #     residuals = (r_n_u, r_n_v, r_e_c, r_n_T)
    #     stencils = ((u_E, u_W, u_N, u_S), (v_E, v_W, v_N, v_S), (p_E, p_W, p_N, p_S), (T_E, T_W, T_N, T_S))
    #     faces = ((u_e, u_w, v_n, v_s), (p_e, p_w, p_n, p_s))    
        
    #     return sols, residuals, stencils, faces


    def r_net_nd(self, params, prev_params, pts_weights, x, y, **kwargs):
        h = self.h_stencil
        inf_small = self.inf_small
        
        # 1. Nghiệm và thặng dư PDE ở bước hiện tại n (dạng O(1))
        sols_curr, res_base, stencil_curr, _ = self._compute_fvm_base_residuals(params, x, y, **kwargs)
        u_P, v_P, p_P, T_P = sols_curr
        r_u, r_v, r_c, r_T = res_base
        
        (u_E, u_W, u_N, u_S), (v_E, v_W, v_N, v_S), \
        (p_E, p_W, p_N, p_S), (T_E, T_W, T_N, T_S) = stencil_curr

        # 2. Nghiệm ở bước trước n-1 (Stop Gradient)
        sols_prev, _, stencil_prev, _ = jax.lax.stop_gradient(
            self._compute_fvm_base_residuals(prev_params, x, y, **kwargs)
        )
        u_P_prev, v_P_prev, p_P_prev, T_P_prev = sols_prev
        (u_E_prev, u_W_prev, u_N_prev, u_S_prev), (v_E_prev, v_W_prev, v_N_prev, v_S_prev), \
        (p_E_prev, p_W_prev, p_N_prev, p_S_prev), (T_E_prev, T_W_prev, T_N_prev, T_S_prev) = stencil_prev
    
        # 3. Hệ số nới lỏng alpha (mặc định 0.84 theo bài báo)
        alpha_u = pts_weights.get('eqn_0', 0.84)
        alpha_v = pts_weights.get('eqn_1', 0.84)
        alpha_T = pts_weights.get('eqn_3', 0.84)

        # 4. Thiết lập hệ số FVM vi phân O(1) cho môi trường xốp
        eps_local = self._get_geometry_alpha(x, y)
        gamma_local = self._get_geometry_gamma(x, y)
        inv_eps = 1.0 / jnp.maximum(eps_local, inf_small)
        
        visc_coeff = jnp.sqrt(self.Pr / self.Ra)
        drag_linear = visc_coeff * eps_local * (1.0 / self.Da_p)

        # Hệ số khuếch tán bậc vi phân O(1) ứng với toán tử laplacian_2d / h^2
        diff_flux_coeff = visc_coeff * inv_eps / (h ** 2)
        a_P_diff = 4.0 * diff_flux_coeff + gamma_local * drag_linear #* (h ** 2)
        a_NB_diff = -1.0 * diff_flux_coeff

        # 5. Hiệu chỉnh biến thiên Gradient áp suất delta_b ở bậc vi phân O(1)
        # b_u = dp/dx ≈ (p_E - p_W) / (2h)
        delta_bP_u = ((p_E - p_W) - (p_E_prev - p_W_prev)) / (2.0 * h)
        delta_bP_v = ((p_N - p_S) - (p_N_prev - p_S_prev)) / (2.0 * h)

        # Tổng biến thiên của các nút lân cận
        sum_du_NB = a_NB_diff * fvm.sum_stencil_neighbors(
            u_E - u_E_prev, u_W - u_W_prev, u_N - u_N_prev, u_S - u_S_prev
        )
        sum_dv_NB = a_NB_diff * fvm.sum_stencil_neighbors(
            v_E - v_E_prev, v_W - v_W_prev, v_N - v_N_prev, v_S - v_S_prev
        )

        # 6. Tính số hạng hiệu chỉnh R (Eq. 27 theo chuẩn vi phân O(1))
        R_u = - (sum_du_NB + delta_bP_u + r_u) / a_P_diff
        R_v = - (sum_dv_NB + delta_bP_v + r_v) / a_P_diff

        # 7. Residual Correction Loss L1-norm (Eq. 26)
        RC_u = jnp.abs((u_P - u_P_prev) - alpha_u * R_u) * 0
        RC_v = jnp.abs((v_P - v_P_prev) - alpha_v * R_v) * 0

        diff_T_coeff = 1.0 / jnp.sqrt(self.Pr * self.Ra)
        diff_flux_coeff_T = diff_T_coeff / (h ** 2)
        a_P_T_diff = 4.0 * diff_flux_coeff_T
        a_NB_T_diff = -1.0 * diff_flux_coeff_T
        sum_dT_NB = a_NB_T_diff * fvm.sum_stencil_neighbors(
            T_E - T_E_prev, T_W - T_W_prev, T_N - T_N_prev, T_S - T_S_prev
        )
        R_T = - (sum_dT_NB + r_T) / a_P_T_diff
        RC_T = jnp.abs((T_P - T_P_prev) - alpha_T * R_T) * 0

        return (r_u, r_v, r_c, r_T), (RC_u, RC_v, RC_T)