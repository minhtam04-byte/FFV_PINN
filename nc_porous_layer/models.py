import jax 
import jax.numpy as jnp 

from jaxpi2.models_2 import BaseHeatPorousScalePINN_2

class LidCavityPorous_fvm(BaseHeatPorousScalePINN_2):
    def __init__(self, config, lr, tx, arch, state, lb, ub):
        super().__init__(config, lr, tx, arch, state, lb, ub)
        
        self.H = config.physics.get('H')
        self.L = config.physics.get('L')
        self.T_h = 1.0
        self.T_c = 0.0
        self.delta_T = config.physics.get('delta_T', self.T_h - self.T_c)
        self.T_ref = config.physics.get('T_ref')

        # 2. Non-dimensional numbers
        self.Pr = config.physics.get('Pr')
        self.Ra = config.physics.get('Ra')
        self.Da_p = config.physics.get('Da')
        self.Je = config.physics.get('Je')
        self.epsilon_p = config.physics.get('epsilon')
        self.F_eps = config.physics.get('F_eps')
        
        self.g_beta = config.physics.get('g_beta') 
        self.nu = jnp.sqrt(self.g_beta * self.delta_T * (self.H**3) * self.Pr / self.Ra)
        
        # Khuếch tán nhiệt trong môi trường xốp
        self.alpha_m_fluid = self.nu / self.Pr 
        self.alpha_m_porous = config.physics.get('alpha_m_porous')
        self.Re = config.physics.get('Re')
        self.v0 = self.Re * self.nu / self.H 
        

        self.dT_dy_pred_fn = jax.vmap(self.dT_dy_scalar, in_axes=(None, 0, 0))

    def neural_net(self, params, x, y, **kwargs):
        x_norm = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        y_norm = 2.0 * (y - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0
        y_embed_T = jnp.cos(jnp.pi * y / self.L)
        X_in = jnp.hstack([x_norm, y_norm, y_embed_T])
        outputs = self.state.apply_fn(params, X_in)
        u_raw = outputs[..., 0:1]
        v_raw = outputs[..., 1:2]
        p_final = outputs[..., 2:3]
        T_raw = outputs[..., 3:4]
        
        # Hard constraint for Dirichlet (u, v = 0 for 4 boundaries)
        bubble_uv = (x / self.L) * (1.0 - x / self.L) * (y / self.H) * (1.0 - y / self.H)
        u_final = bubble_uv * u_raw
        v_final = bubble_uv * v_raw

        # Hard constraint for Dirichlet  temperature(T=1 left, T=0 right)
        bubble_x = (x / self.L) * (1.0 - x / self.L)
        T_final = (1.0 - x / self.L) + bubble_x * T_raw
        
        return u_final, v_final, p_final, T_final
    
    def u_net(self, params, x, y, **kwargs):
        return self.neural_net(params, x, y, **kwargs)

    def get_T_scalar(self, params, x, y):
        _, _, _, T = self.u_net(params, x, y)
        return jnp.squeeze(T)
    
    def dT_dy_scalar(self, params, x, y):
        return jax.grad(self.get_T_scalar, argnums=2)(params, x, y)
    
    def sdf_fn(self, x_val, y_val):
        """
        Hàm SDF cho một lớp xốp thẳng đứng nằm ở giữa miền tính toán.
        phi < 0: Bão hòa xốp (Porous)
        phi > 0: Chất lưu thuần túy (Fluid)
        """
        # Lấy thông số từ config (hoặc hardcode)
        x_c = self.config.ibm.get('layer_x_center') # Tâm của lớp xốp
        W = self.config.ibm.get('layer_width')      # Bề dày lớp xốp

        return jnp.abs(x_val - x_c) - (W / 2.0)
    
    
    def losses(self, params, state, batch):
        loss_dict = {}
        
        # A. (Scale-PINN + SIMPLE-PINN)
        if 'eqn' in batch:
            coords = tuple(batch['eqn'][:, i] for i in range(batch['eqn'].shape[1]))
            
            # outputs
            outputs = self.r_pred_total_fn(params, state.prev_params, state.pts_weights, *coords)
            f_u, f_v, f_e, f_T, rc_u, rc_v, rc_p = outputs
            
            # MSE for PDE 
            loss_dict['eqn_0'] = jnp.mean(f_u ** 2)
            loss_dict['eqn_1'] = jnp.mean(f_v ** 2)
            loss_dict['eqn_2'] = jnp.mean(f_e ** 2)
            loss_dict['eqn_3'] = jnp.mean(f_T ** 2)
            
            # MAE for SIMPLE-PINN Correction
            loss_dict['rc_u'] = jnp.mean(jnp.abs(rc_u))
            loss_dict['rc_v'] = jnp.mean(jnp.abs(rc_v))
            loss_dict['rc_p'] = jnp.mean(jnp.abs(rc_p))

        # Neumann for condition 
        for side in ['bc_y0', 'bc_yH']:
            if side in batch:
                coords = batch[side] # shape: (batch_size, 2)
                bx, by = coords[:, 0], coords[:, 1]
                dT_dy_pred = self.dT_dy_pred_fn(params, bx, by)
                loss_dict[f'{side}_T'] = jnp.mean(dT_dy_pred**2)

        return loss_dict