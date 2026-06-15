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
        if isinstance(batch, dict):
            pts_list = []
            if 'eqn_ad' in batch: pts_list.append(batch['eqn_ad'])
            if 'eqn_fvm' in batch: pts_list.append(batch['eqn_fvm'])
            all_pts = jnp.concatenate(pts_list, axis=0) if pts_list else jnp.zeros((1, self.config.input_dim))
            coords = tuple(all_pts[:, i] for i in range(all_pts.shape[1]))
        else:
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
        """
        Phiên bản AD cho Môi trường Xốp ĐỒNG NHẤT (Uniform Porous Media)
        Đã hiệu chỉnh chuẩn theo Generalized Navier-Stokes (Liu et al. 2021)
        """
        inv_K = (1.0 / self.Da_p) / (self.H**2)
        V_mag_k = jnp.sqrt(fields.u**2 + fields.v**2 + self.inf_small)
        
        # Độ rỗng là hằng số trong môi trường đồng nhất
        eps = self.epsilon_p
        inv_eps = 1.0 / eps

        # 1. Phương trình liên tục (Continuity): div(U) = 0 cho vận tốc biểu kiến (Darcy velocity)
        f_e = fields.u_x + fields.v_y 

        # 2. Tensor ứng suất (Viscous terms) - Lưu ý u nhận v_yx, v nhận u_yx
        visc_u = self.nu * fields.laplacian_u#(2.0 * fields.u_xx + fields.u_yy + fields.v_yx)
        visc_v = self.nu * fields.laplacian_v#(fields.u_yx + fields.v_xx + 2.0 * fields.v_yy)

        # 3. Phương trình Động lượng cơ sở (ĐÃ THÊM 1/epsilon VÀO ĐỐI LƯU)
        conv_u = fields.u * fields.u_x + fields.v * fields.u_y#2.0 * fields.u * fields.u_x + fields.v * fields.u_y + fields.u * fields.v_y
        f_u_base = inv_eps * conv_u + eps * fields.p_x - visc_u 

        conv_v = fields.u * fields.v_x + fields.v * fields.v_y #fields.v * fields.u_x + fields.u * fields.v_x + 2.0 * fields.v * fields.v_y
        f_v_base = inv_eps * conv_v + eps * fields.p_y - visc_v 
        
        # 4. Phương trình Năng lượng
        alpha_m_local = eps * self.alpha_m_fluid + (1.0 - eps) * self.alpha_m_porous
        diff_T = alpha_m_local * fields.laplacian_T 
        conv_T = fields.u * fields.T_x + fields.v * fields.T_y
        f_T_total = conv_T - diff_T
        
        # 5. Lực cản Darcy-Forchheimer & Lực nổi (ĐÃ THÊM epsilon VÀO LỰC NỔI)
        f_u_drag = -fields.gamma * ((self.nu * eps * inv_K) * fields.u + (eps * self.F_eps * jnp.sqrt(inv_K)) * V_mag_k * fields.u)
        
        f_v_drag = -fields.gamma * ((self.nu * eps * inv_K) * fields.v + (eps * self.F_eps * jnp.sqrt(inv_K)) * V_mag_k * fields.v) \
                   + eps * self.g_beta * (fields.T - self.T_ref) 

        return f_u_base - f_u_drag, f_v_base - f_v_drag, f_e, f_T_total

    def r_net_nd(self, params, prev_params, pts_weights, x, y, **kwargs):
        h = self.h_stencil
        inf_small = self.inf_small
        inv_tau_u = pts_weights.get('eqn_0', 1.0) 
        inv_tau_v = pts_weights.get('eqn_1', 1.0)
        inv_tau_p = pts_weights.get('eqn_2', 1.0)
        inv_tau_T = pts_weights.get('eqn_3', 1.0)
        # =========================================================================
        # STEP 1: TRÍCH XUẤT GIÁ TRỊ TẠI 9 ĐIỂM (Tâm P, 4 Lân cận E,W,N,S và 4 Mặt e,w,n,s)
        # BÀI BÁO: Mục 2.3.1 (Simplified FVM), Công thức (15) và (30)
        # LÝ DO: FFV-PINN yêu cầu mạng Neural Network dự đoán TRỰC TIẾP giá trị tại các 
        # mặt kiểm soát (cách tâm h/2) thay vì nội suy trung bình cộng.
        # =========================================================================
        xs = jnp.array([x, x + h, x - h, x, x, x + h/2, x - h/2, x, x])
        ys = jnp.array([y, y, y, y + h, y - h, y, y, y + h/2, y - h/2])
        
        def get_vals(p, x_in, y_in):
            return self.u_net(p, x_in, y_in, **kwargs)
            
        # Mạng hiện tại (Vòng lặp k / Epoch n)
        u_all, v_all, p_all, T_all = jax.vmap(get_vals, in_axes=(None, 0, 0))(params, xs, ys)
        u_P, u_E, u_W, u_N, u_S, u_e, u_w, u_n, u_s = u_all
        v_P, v_E, v_W, v_N, v_S, v_e, v_w, v_n, v_s = v_all
        p_P, p_E, p_W, p_N, p_S, p_e, p_w, p_n, p_s = p_all
        T_P, T_E, T_W, T_N, T_S, T_e, T_w, T_n, T_s = T_all

        # Mạng bước lặp trước (Vòng lặp k-1 / Epoch n-1) - Dừng đồ thị Gradient
        old_u_all, old_v_all, old_p_all, old_T_all = jax.lax.stop_gradient(
            jax.vmap(get_vals, in_axes=(None, 0, 0))(prev_params, xs, ys)
        )
        old_u_P, old_u_E, old_u_W, old_u_N, old_u_S, old_u_e, old_u_w, old_u_n, old_u_s = old_u_all
        old_v_P, old_v_E, old_v_W, old_v_N, old_v_S, old_v_e, old_v_w, old_v_n, old_v_s = old_v_all
        old_p_P, old_p_E, old_p_W, old_p_N, old_p_S, old_p_e, old_p_w, old_p_n, old_p_s = old_p_all
        old_T_P, old_T_E, old_T_W, old_T_N, old_T_S, old_T_e, old_T_w, old_T_n, old_T_s = old_T_all

        # Tính toán độ biến thiên (delta phi) giữa 2 epoch cho Residual Correction
        du_P = u_P - old_u_P
        sum_du_NB = (u_E - old_u_E) + (u_W - old_u_W) + (u_N - old_u_N) + (u_S - old_u_S)
        
        dv_P = v_P - old_v_P
        sum_dv_NB = (v_E - old_v_E) + (v_W - old_v_W) + (v_N - old_v_N) + (v_S - old_v_S)
        
        dp_P = p_P - old_p_P
        
        dT_P = T_P - old_T_P
        sum_dT_NB = (T_E - old_T_E) + (T_W - old_T_W) + (T_N - old_T_N) + (T_S - old_T_S)

        # =========================================================================
        # STEP 2: ĐỒNG BỘ THUỘC TÍNH VẬT LIỆU CỤC BỘ
        # =========================================================================
        eps_local = self._get_geometry_alpha(x, y) 
        gamma_local = self._get_geometry_gamma(x, y) 
        inv_eps = 1.0 / (eps_local + inf_small)

        # =========================================================================
        # STEP 3: TÍNH TOÁN THÔNG LƯỢNG (Simplified FVM)
        # BÀI BÁO: Công thức (6) cho khuếch tán, (7) cho Gradient P, (15) cho đối lưu
        # =========================================================================
        # 3.1 Phương trình liên tục (Công thức 30)
        f_e_fvm = (u_e - u_w) / h + (v_n - v_s) / h

        # 3.2 Đối lưu Động lượng (Dùng trực tiếp u_e, v_e... tại các mặt)
        conv_u_fvm = (u_e * u_e - u_w * u_w) / h + (v_n * u_n - v_s * u_s) / h
        conv_v_fvm = (u_e * v_e - u_w * v_w) / h + (v_n * v_n - v_s * v_s) / h
        
        # Đối lưu Động lượng của epoch trước (Phục vụ tính số hạng b)
        old_conv_u_fvm = (old_u_e * old_u_e - old_u_w * old_u_w) / h + (old_v_n * old_u_n - old_v_s * old_u_s) / h
        old_conv_v_fvm = (old_u_e * old_v_e - old_u_w * old_v_w) / h + (old_v_n * old_v_n - old_v_s * old_v_s) / h

        # 3.3 Đối lưu nhiệt
        conv_T_fvm = (u_e * T_e - u_w * T_w) / h + (v_n * T_n - v_s * T_s) / h
        old_conv_T_fvm = (old_u_e * old_T_e - old_u_w * old_T_w) / h + (old_v_n * old_T_n - old_v_s * old_T_s) / h

        # 3.4 Gradient Áp suất (Central Difference từ lân cận - Công thức 7a, 7b chia cho h^2)
        p_x_fvm = (p_E - p_W) / (2.0 * h)
        p_y_fvm = (p_N - p_S) / (2.0 * h)
        
        old_p_x_fvm = (old_p_E - old_p_W) / (2.0 * h)
        old_p_y_fvm = (old_p_N - old_p_S) / (2.0 * h)

        # 3.5 Khuếch tán Laplacian số học (Công thức 6)
        u_xx_yy = (u_E + u_W + u_N + u_S - 4.0 * u_P) / (h**2)
        v_xx_yy = (v_E + v_W + v_N + v_S - 4.0 * v_P) / (h**2)
        T_xx_yy = (T_E + T_W + T_N + T_S - 4.0 * T_P) / (h**2)

        visc_u = self.nu * u_xx_yy
        visc_v = self.nu * v_xx_yy
        
        alpha_m_local = eps_local * self.alpha_m_fluid + (1.0 - eps_local) * self.alpha_m_porous
        diff_T = alpha_m_local * T_xx_yy

        # 3.6 Số hạng nguồn: Môi trường Xốp & Boussinesq
        inv_K = (1.0 / self.Da_p) / (self.H**2)
        V_mag_k = jnp.sqrt(u_P**2 + v_P**2 + inf_small)
        
        drag_linear = self.nu * eps_local * inv_K
        drag_nonlin = eps_local * self.F_eps * jnp.sqrt(inv_K)
        
        f_u_drag = -gamma_local * (drag_linear * u_P + drag_nonlin * V_mag_k * u_P)
        f_v_drag = -gamma_local * (drag_linear * v_P + drag_nonlin * V_mag_k * v_P) \
                   + eps_local * self.g_beta * (T_P - self.T_ref) 

        # =========================================================================
        # STEP 4: HỢP NHẤT PHẦN DƯ PHƯƠNG TRÌNH NỀN (BASE FVM RESIDUALS: r^n)
        # =========================================================================
        f_u_total = inv_eps * conv_u_fvm + eps_local * p_x_fvm - visc_u - f_u_drag
        f_v_total = inv_eps * conv_v_fvm + eps_local * p_y_fvm - visc_v - f_v_drag
        f_T_base = conv_T_fvm - diff_T

        # # =========================================================================
        # # STEP 5: TÍNH CÁC HỆ SỐ a_P, a_NB VÀ delta_b CHO RESIDUAL CORRECTION
        # # BÀI BÁO: Mục 2.3.2, hệ số $a_P$ và $a_{NB}$ từ phương trình tổng quát (16)
        # # LÝ DO: Xác định đạo hàm của r^n đối với điểm trung tâm P và lân cận NB.
        # # =========================================================================
        # a_NB_uv = -self.nu / (h**2)
        # a_NB_T  = -alpha_m_local / (h**2)
        
        # # a_P là hệ số đi kèm phi_P trong phương trình dư tổng (r^n)
        # a_P_u = 4.0 * self.nu / (h**2) + gamma_local * (drag_linear + drag_nonlin * V_mag_k)
        # a_P_v = 4.0 * self.nu / (h**2) + gamma_local * (drag_linear + drag_nonlin * V_mag_k)
        # a_P_T = 4.0 * alpha_m_local / (h**2)
        
        # # delta_b = (b^n - b^{n-1}) là độ lệch các số hạng không nhân với u_P hay u_NB (đối lưu, áp suất, lực nổi)
        # delta_b_u = inv_eps * (conv_u_fvm - old_conv_u_fvm) + eps_local * (p_x_fvm - old_p_x_fvm)
        # delta_b_v = inv_eps * (conv_v_fvm - old_conv_v_fvm) + eps_local * (p_y_fvm - old_p_y_fvm) - eps_local * self.g_beta * dT_P
        # delta_b_T = conv_T_fvm - old_conv_T_fvm

        # # =========================================================================
        # # STEP 6: TÍNH TOÁN RC LOSS THEO BÀI BÁO (RESIDUAL CORRECTION)
        # # BÀI BÁO: Công thức (26) và (27)
        # # =========================================================================
        # # Số hạng hiệu chỉnh dư R (Công thức 27)
        # R_u = (-f_u_total - a_NB_uv * sum_du_NB - delta_b_u) / (a_P_u + inf_small)
        # R_v = (-f_v_total - a_NB_uv * sum_dv_NB - delta_b_v) / (a_P_v + inf_small)
        # R_T = (-f_T_base  - a_NB_T  * sum_dT_NB - delta_b_T) / (a_P_T + inf_small)
        
        # # L_rc Loss Term (Công thức 26): | phi^n - phi^{n-1} - alpha * R |
        # # Trong hệ thống mất mát tự động của JAX/PINN, ta trả về hàm lõi bên trong trị tuyệt đối
        # rc_u = du_P - self.alpha_u * R_u
        # rc_v = dv_P - self.alpha_u * R_v
        # rc_T = dT_P - self.alpha_T * R_T
        
        # # LƯU Ý CHO ÁP SUẤT: 
        # # Lưới FVM đồng vị không có a_P trực tiếp cho áp suất trong phương trình liên tục (a_P = 0).
        # # Do vậy, RC cho áp suất áp dụng cập nhật truyền thống Pseudo-Transient (Artificial Compressibility).
        # rc_p = dp_P + self.alpha_p * f_e_fvm
        fvm_eqn_0 = f_u_total + inv_tau_u * du_P
        fvm_eqn_1 = f_v_total + inv_tau_v * dv_P
        fvm_eqn_3 = f_T_base + inv_tau_T * dT_P
        
        # Artificial Compressibility cho áp suất
        fvm_eqn_2 = f_e_fvm + inv_tau_p * dp_P


        return fvm_eqn_0, fvm_eqn_1, fvm_eqn_2, fvm_eqn_3#, rc_u, rc_v, rc_p, rc_T

    # === CÁC HÀM CẦN THIẾT ĐỂ TƯƠNG THÍCH VỚI LỚP PINN ===
    def r_net_base(self, params, x, y, **kwargs):
        """Tính ma trận thời gian ảo cho Scale-PINN"""
        fields = self._get_physics_components(params, x, y, **kwargs)
        f_u_total, f_v_total, f_e, f_T_total = self._compute_base_residuals_from_k(fields)
        #f_u_fvm, f_v_fvm, f_e_fvm, f_T_fvm = self.r_net_nd(params, )
        return f_u_total, f_v_total, f_e, f_T_total

    def r_net_total(self, params, prev_params, pts_weights, x, y, **kwargs):
        """Wrap lại logic cho ForwardBVP"""
        return self.r_net_scale_pinn_only(params, prev_params, pts_weights, x, y, **kwargs)

    def r_net_scale_pinn_only(self, params, prev_params, pts_weights, x, y, **kwargs):
        fields = self._get_physics_components(params, x, y, **kwargs)
        old_fields = jax.lax.stop_gradient(self._get_physics_components(prev_params, x, y, **kwargs))
        f_u_base, f_v_base, f_e_base, f_T_base = self._compute_base_residuals_from_k(fields)

        inv_tau_u = pts_weights.get('eqn_0', 1.0) 
        inv_tau_v = pts_weights.get('eqn_1', 1.0)
        inv_tau_p = pts_weights.get('eqn_2', 1.0)
        inv_tau_T = pts_weights.get('eqn_3', 1.0)

        S_u = inv_tau_u * (fields.u - old_fields.u)
        S_v = inv_tau_v * (fields.v - old_fields.v)
        S_p = inv_tau_p * (fields.p - old_fields.p)
        S_T = inv_tau_T * (fields.T - old_fields.T)

        if getattr(self.config.ablation, 'use_laplacian', True):
            gamma_u = gamma_v = self.nu / self.epsilon_p
            gamma_T = self.alpha_m_porous
            cfl = getattr(self.config.ablation, 'cfl_ratio', 1.0)
            
            S_u -= inv_tau_u * gamma_u * cfl * (fields.laplacian_u - old_fields.laplacian_u)
            S_v -= inv_tau_v * gamma_v * cfl * (fields.laplacian_v - old_fields.laplacian_v)
            S_T -= inv_tau_T * gamma_T * cfl * (fields.laplacian_T - old_fields.laplacian_T)

        return f_u_base + S_u, f_v_base + S_v, f_e_base + S_p, f_T_base + S_T


