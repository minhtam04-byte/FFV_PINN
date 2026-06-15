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

        # SỬA Ở ĐÂY: Dùng r_pred_total_fn (Đã chứa Scale-PINN bên trong)
        res_total = jnp.stack(self.r_pred_total_fn(params, state.prev_params, state.pts_weights, *coords))

        if res_total.ndim == 1:
            res_total = res_total[None, :]

        # KHÔNG CẦN CỘNG THÊM pseudo_time Ở ĐÂY NỮA
        # Vì nó đã được cộng bên trong r_net_total!

        per_key_losses = jnp.mean(res_total ** 2, axis=1)
        return dict(zip(keys, per_key_losses))



class BaseHeatPorousScalePINN(ForwardBVP):
    def __init__(self, config, lr, tx, arch, state, lb, ub):
        # Truyền các tham số hạ tầng lên cho class cha (ForwardBVP)
        super().__init__(config, lr, tx, arch, state)
        
        self.lb = lb
        self.ub = ub
        
        # Lấy các tham số vật lý
        self.tau_sc = config.physics.get('tau_sc', 0.5)
        self.tau_alpha = config.physics.get('tau_alpha', 0.5)
        self.nu = config.physics.get('nu', 0.01)
        self.alpha_f = config.physics.get('alpha_f', 0.01)
        self.epsilon_p = config.physics.get('epsilon', 0.4)
        self.Da_p = config.physics.get('Da', 1e-3)
        self.H = config.physics.get('H', 1.0)
        self.F_eps = config.physics.get('F_eps', 0.1)
        self.g_beta = config.physics.get('g_beta', 1.0)
        self.alpha_m_porous = config.physics.get('alpha_m_porous', 0.01)

    def u_net(self, params, x, y, **kwargs):
        """Wrapper để gọi neural_net của JAXPI"""
        
        outputs = self.arch.apply(params, jnp.concatenate([x, y], axis=-1))
        return outputs[..., 0:1], outputs[..., 1:2], outputs[..., 2:3], outputs[..., 3:4]

    def _get_physics_components(self, params, x, y, **kwargs):
        
        x_s = jnp.squeeze(x)
        y_s = jnp.squeeze(y)

        def forward(x_in, y_in):
            u, v, p, T = self.u_net(params, x_in, y_in, **kwargs)
            return u,v,p,T

        jac_f = jax.jacrev(forward, argnums=(0, 1))(x_s, y_s)
        jac_u, jac_v, jac_p, jac_T = jac_f
        

        u_x, u_y = jac_u
        v_x, v_y = jac_v
        p_x, p_y = jac_p
        T_x, T_y = jac_T


        hess_x = jax.jacfwd(jax.jacrev(forward, argnums=0), argnums=0)(x_s, y_s)
        hess_y = jax.jacfwd(jax.jacrev(forward, argnums=1), argnums=1)(x_s, y_s)


        u_xx, v_xx, p_xx, T_xx = hess_x
        u_yy, v_yy, p_yy, T_yy = hess_y

        u, v, p, T = forward(x_s, y_s)

        return {
            'u': u, 'v': v, 'p': p, 'T': T,
            'u_x': u_x, 'v_x': v_x, 'p_x': p_x, 'T_x': T_x,
            'u_y': u_y, 'v_y': v_y, 'p_y': p_y, 'T_y': T_y,
            'u_xx': u_xx, 'v_xx': v_xx, 'T_xx': T_xx,
            'u_yy': u_yy, 'v_yy': v_yy, 'T_yy': T_yy,
            'laplacian_u': u_xx + u_yy,
            'laplacian_v': v_xx + v_yy,
            'laplacian_T': T_xx + T_yy
        }

    # =========================================================================
    # 1. HÀM VẬT LÝ THUẦN TÚY (Dùng để nuôi thuật toán Sifan Wang)
    # =========================================================================
    def r_net_base(self, params, x, y, **kwargs):
        k = self._get_physics_components(params, x, y, **kwargs)

        eps = self.epsilon_p
        inv_K = (1.0 / self.Da_p) / (self.H**2)
        V_mag_k = jnp.sqrt(k['u']**2 + k['v']**2)

        # Phương trình liên tục (pde_e)
        f_e = k['u_x'] + k['v_y']

        # Phương trình động lượng và năng lượng
        f_u_base = (k['u']*k['u_x'] + k['v']*k['u_y'])/(eps**2) + k['p_x'] \
                 - (self.nu/eps)*k['laplacian_u'] + (self.nu*inv_K)*k['u'] \
                 + (self.F_eps*jnp.sqrt(inv_K)) * V_mag_k * k['u']

        f_v_base = (k['u']*k['v_x'] + k['v']*k['v_y'])/(eps**2) + k['p_y'] \
                 - (self.nu/eps)*k['laplacian_v'] + (self.nu*inv_K)*k['v'] \
                 + (self.F_eps*jnp.sqrt(inv_K)) * V_mag_k * k['v'] \
                 - self.g_beta * (k['T'] - 0.5)

        f_T_base = (k['u']*k['T_x'] + k['v']*k['T_y']) - self.alpha_m_porous * k['laplacian_T']

        return f_u_base, f_v_base, f_e, f_T_base

    # =========================================================================
    # 2. HÀM TỔNG HỢP ABLATION (Chứa Scale-PINN và Laplacian)
    # =========================================================================
    # =========================================================================
    # 2. HÀM TỔNG HỢP ABLATION (Đã bóc tách Tau cho từng phương trình)
    # =========================================================================
    def r_net_total(self, params, prev_params, pts_weights, x, y, **kwargs):
        # 1. Lấy Residual gốc (Thứ tự: u, v, p, T)
        f_u_base, f_v_base, f_e_base, f_T_base = self.r_net_base(params, x, y, **kwargs)
        
        S_u, S_v, S_p, S_T = 0.0, 0.0, 0.0, 0.0
        
        use_seq = getattr(self.config.ablation, 'use_sequential', True)
        use_lap = getattr(self.config.ablation, 'use_laplacian', True)
        tau_mode = getattr(self.config.ablation, 'tau_mode', 'adaptive')

        if use_seq:
            k = self._get_physics_components(params, x, y, **kwargs)
            old = jax.lax.stop_gradient(self._get_physics_components(prev_params, x, y, **kwargs))

            # --- A. LẤY HỆ SỐ TAU ĐỘC LẬP CHO TỪNG PHƯƠNG TRÌNH ---
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
            S_u = inv_tau_u * (k['u'] - old['u'])
            S_v = inv_tau_v * (k['v'] - old['v'])
            S_p = inv_tau_p * (k['p'] - old['p'])
            S_T = inv_tau_T * (k['T'] - old['T'])

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

                S_u -= zeta_u * (k['laplacian_u'] - old['laplacian_u'])
                S_v -= zeta_v * (k['laplacian_v'] - old['laplacian_v'])
                S_T -= zeta_T * (k['laplacian_T'] - old['laplacian_T'])

        return f_u_base + S_u, f_v_base + S_v, f_e_base + S_p, f_T_base + S_T