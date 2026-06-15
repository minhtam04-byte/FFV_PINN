import jax 
import jax.numpy as jnp 

from Tam_project_2.jaxpi2.models_1 import BaseHeatPorousScalePINN 

class MixedConvectionPorous(BaseHeatPorousScalePINN):
    def __init__(self, config, lr, tx, arch, state, lb, ub):
        super().__init__(config, lr, tx, arch, state, lb, ub)
        
        self.H = 1.0
        self.L = 1.0
        
        self.Da_p = config.physics.get('Da')
        self.epsilon_p = config.physics.get('epsilon')
                 
        self.Pr = config.physics.get('Pr')
        self.Ra = config.physics.get('Ra')
        self.Je = config.physics.get('Je')
        self.u0 = 0.1
        self.T_h = 1.0
        self.T_c = 0.0
        self.delta_T = self.T_h - self.T_c
        self.T_ref = (self.T_h + self.T_c) / 2.0
        
        self.g_beta = 0.1
        self.nu = jnp.sqrt(self.g_beta * self.delta_T * self.H**3 * self.Pr / self.Ra)
        self.v0 = config.physics.get('Re', 10.0) * self.nu / self.H
        
        self.alpha_m_porous = self.nu / self.Pr 
        
        self.Re = config.physics.get('Re')
        self.r1 = self.Re / (2.0 * self.epsilon_p)
        term_sq = self.Re**2 + (4.0 * (self.epsilon_p**3) * self.Je) / self.Da_p
        self.r2 = (1.0 / (2.0 * self.epsilon_p * self.Je)) * jnp.sqrt(term_sq)
        self.F_eps = config.physics.get('F_eps')

    def neural_net(self, params, x, y):
        y_norm = 2.0 * (y - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0
        X_input = jnp.hstack([x, y_norm]) # x already put to periodic layer so it not need to norm
        #jax.debug.print("--- X_input Shape: {s} ---", s=X_input.shape)
        outputs = self.arch.apply(params, X_input)
        return outputs[..., 0], outputs[..., 1], outputs[..., 2], outputs[..., 3]
    
    def u_net(self, params, x, y):
        return self.neural_net(params, x, y)

    def analytical_solution(self, y):
        exp_term = jnp.exp(self.r1 * (y / self.H - 1.0))
        sinh_term = jnp.sinh(self.r2 * y / self.H) / jnp.sinh(self.r2)
        u_star = self.u0 * exp_term * sinh_term
        v_star = jnp.full_like(y, self.v0)
        PrRe = self.Pr * self.Re
        temp_ratio = (jnp.exp(PrRe * y / self.H) - 1.0) / (jnp.exp(PrRe) - 1.0)
        T_star = self.T_h - self.delta_T * temp_ratio
        return u_star, v_star, T_star
    

    def r_net_base(self, params, x, y, **kwargs):
        k = self._get_physics_components(params, x, y, **kwargs)

        eps = self.epsilon_p
        inv_K = (1.0 / self.Da_p) / (self.H**2)
        V_mag_k = jnp.sqrt(k['u']**2 + k['v']**2)

        # Phương trình liên tục (pde_e)
        f_e = k['u_x'] + k['v_y']
        PrRe = self.Pr * self.Re
        temp_ratio = (jnp.exp(PrRe * y / self.H) - 1.0) / (jnp.exp(PrRe) - 1.0)
        a_y = (self.nu * inv_K) * self.v0 - self.g_beta * self.delta_T * (0.5 - temp_ratio)
        # Phương trình động lượng và năng lượng
        f_u_base = (k['u']*k['u_x'] + k['v']*k['u_y'])/(eps**2) + k['p_x'] \
                 - (self.nu/eps)*k['laplacian_u'] + (self.nu*inv_K)*k['u'] \
                 + (self.F_eps*jnp.sqrt(inv_K)) * V_mag_k * k['u']

        f_v_base = (k['u']*k['v_x'] + k['v']*k['v_y'])/(eps**2) + k['p_y'] \
                 - (self.nu/eps)*k['laplacian_v'] + (self.nu*inv_K)*k['v'] \
                 + (self.F_eps*jnp.sqrt(inv_K)) * V_mag_k * k['v'] \
                 - self.g_beta * (k['T'] - self.T_ref) - a_y

        f_T_base = (k['u']*k['T_x'] + k['v']*k['T_y']) - self.alpha_m_porous * k['laplacian_T']

        return f_u_base, f_v_base, f_e, f_T_base
    
    # 3. SỬA LẠI HÀM LOSSES CHO CHUẨN JAXPI
    def losses(self, params, state, batch):
        loss_dict = {}
        
        if 'eqn' in batch:
            pde_losses = self.compute_residual_losses(params, state, batch['eqn'])
            loss_dict.update(pde_losses)

        # B. Boundary Conditions Loss
        for side in ['bc_y0', 'bc_yH']:
            if side in batch:
                coords = batch[side] # shape: (batch_size, 2)
                bx, by = coords[:, 0], coords[:, 1]
        
                preds = jnp.stack(self.sol_pred_fn(params, bx, by))
                u_pred, v_pred, p_pred, T_pred = preds[0], preds[1], preds[2], preds[3]
                
                u_true, v_true, T_true = self.analytical_solution(by)
                
                # Tính Mean Squared Error
                loss_dict[f'{side}_u'] = jnp.mean((u_pred - u_true)**2)
                loss_dict[f'{side}_v'] = jnp.mean((v_pred - v_true)**2)
                loss_dict[f'{side}_T'] = jnp.mean((T_pred - T_true)**2)

        return loss_dict