import jax 
import jax.numpy as jnp 

from jaxpi2.models import BaseHeatPorousScalePINN

class LidCavityPorous_fvm(BaseHeatPorousScalePINN):
    def __init__(self, config, lr, tx, arch, state, lb, ub):
        super().__init__(config, lr, tx, arch, state, lb, ub)
        
        self.H = config.physics.get('H')
        self.L = config.physics.get('L')
        self.T_h = 1.0
        self.T_c = 0.0
        self.delta_T = config.physics.get('delta_T', self.T_h - self.T_c)
        self.T_ref = config.physics.get('T_ref')

        self.Pr = config.physics.get('Pr')
        self.Ra = config.physics.get('Ra')
        self.Da_p = config.physics.get('Da')
        self.Je = config.physics.get('Je')
        self.epsilon_p = config.physics.get('epsilon')
        self.F_eps = config.physics.get('F_eps')
        self.alpha_T = 1.0
        self.g_beta = config.physics.get('g_beta') 
        self.nu = jnp.sqrt(self.g_beta * self.delta_T * (self.H**3) * self.Pr / self.Ra)
        
        self.alpha_m_fluid = self.nu / self.Pr 
        self.alpha_m_porous = self.alpha_m_fluid #config.physics.get('alpha_m_porous')
        self.Re = config.physics.get('Re')
        self.v0 = self.Re * self.nu / self.H 
        
        self.dT_dy_pred_fn = jax.vmap(self.dT_dy_scalar, in_axes=(None, 0, 0))

    def neural_net(self, params, x, y, **kwargs):
        x_non_dim = x 
        y_non_dim = y 
        X_in = jnp.hstack([x_non_dim, y_non_dim])
        outputs = self.state.apply_fn(params, X_in)
        return outputs[..., 0],outputs[..., 1],outputs[..., 2],outputs[..., 3]
    
    def u_net(self, params, x, y, **kwargs):
        return self.neural_net(params, x, y, **kwargs)

    def get_T_scalar(self, params, x, y):
        _, _, _, T = self.u_net(params, x, y)
        return T
    
    def dT_dy_scalar(self, params, x, y):
        return jax.grad(self.get_T_scalar, argnums=2)(params, x, y)
    
    def sdf_fn(self, x_val, y_val):
        x_c = self.config.ibm.get('layer_x_center') 
        W = self.config.ibm.get('layer_width')      
        return jnp.abs(x_val - x_c) - (W / 2.0)
    
    #     return loss_dict
    def losses(self, params, state, batch):
        loss_dict = {}
        
        # =================================================================
        # 1. TÍNH TOÁN PDE (NỘI MIỀN FVM)
        # =================================================================
        if 'eqn_fvm' in batch:
            coords_fvm = tuple(batch['eqn_fvm'][:, i] for i in range(batch['eqn_fvm'].shape[1]))
            
            f_u_fvm, f_v_fvm, f_e_fvm, f_T_fvm = jax.vmap(
                self.r_net_nd, 
                in_axes=(None, None, None, 0, 0)
            )(params, state.prev_params, state.pts_weights, *coords_fvm)
            
            loss_dict['eqn_0'] = jnp.mean(f_u_fvm ** 2)
            loss_dict['eqn_1'] = jnp.mean(f_v_fvm ** 2)
            loss_dict['eqn_3'] = jnp.mean(f_T_fvm ** 2)
            loss_dict['eqn_2'] = jnp.mean(f_e_fvm ** 2)


        predict_fn = jax.vmap(self.u_net, in_axes=(None, 0, 0))
        
        def get_T_scalar(p, x_val, y_val):
            out = self.u_net(p, x_val, y_val)
            return jnp.squeeze(out[3]) # out[3]   
        #  dT/dy (Neumann)
        dT_dy_vmap = jax.vmap(jax.grad(get_T_scalar, argnums=2), in_axes=(None, 0, 0))

        # =================================================================
        # 3. BC LOSS (SOFT CONSTRAINTS)
        # =================================================================
        
        # 3.1. Vách Đáy (y = 0): u = 0, v = 0, dT/dy = 0 (Neumann)
        if 'bc_uv_zero' in batch:
            bx, by = batch['bc_uv_zero'][:, 0], batch['bc_uv_zero'][:, 1]
            u_pred, v_pred, _, _ = predict_fn(params, bx, by)
            loss_dict['bc_uv'] = jnp.mean(u_pred**2) + jnp.mean(v_pred**2)

        # 3. DIRICHLET NHIỆT ĐỘ (Vách Tây và Đông)
        if 'bc_T_fixed' in batch:
            bx, by = batch['bc_T_fixed'][:, 0], batch['bc_T_fixed'][:, 1]
            T_target = batch['bc_T_fixed'][:, 2] # Lấy giá trị đích (1.0 hoặc 0.0)
            _, _, _, T_pred = predict_fn(params, bx, by)
            loss_dict['bc_T_dirichlet'] = jnp.mean((T_pred - T_target)**2)

        # 4. NEUMANN NHIỆT ĐỘ (Vách Bắc và Nam)
        if 'bc_T_neumann' in batch:
            bx, by = batch['bc_T_neumann'][:, 0], batch['bc_T_neumann'][:, 1]
            dT_dy_pred = dT_dy_vmap(params, bx, by)
            loss_dict['bc_T_neumann'] = jnp.mean(dT_dy_pred**2)

        return loss_dict