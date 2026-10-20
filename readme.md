# Physics-Informed Neural Network for Natural Convection in Porous Media

## 1. Project Overview & Physical Formulation
This project models steady 2D natural convection in porous media using a mesh-free PINN framework built on **JAX** and **Flax**, adapted from `jaxpi` (Wang et al., 2026). 

- **Governing Equations (Dimensionless $O(1)$ Scaling):**
  - **Continuity:** $\nabla \cdot \mathbf{u} = 0$ (coupled with pseudo-pressure).
  - **Momentum (Darcy-Brinkman-Forchheimer):**
    $\frac{1}{\epsilon}(\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p - \sqrt{\frac{Pr}{Ra}}\frac{1}{\epsilon}\nabla^2 \mathbf{u} + \mathbf{F}_{\text{drag}} - \epsilon(T - T_{\text{ref}})\hat{\mathbf{j}} = 0$
    where $\mathbf{F}_{\text{drag}} = \gamma \left(\sqrt{\frac{Pr}{Ra}}\frac{\epsilon}{Da} + \epsilon \frac{F_\epsilon}{\sqrt{Da}}\vert{}\mathbf{u}\vert{}\right)\mathbf{u}$.
  - **Energy:** $(\mathbf{u} \cdot \nabla)T - \frac{1}{\sqrt{Pr \cdot Ra}}\nabla^2 T = 0$.
- **Boundary Conditions (Unit Cavity $[0, 1]^2$):**
  - Velocity: No-slip ($u = v = 0$) on all 4 walls.
  - Temperature: Dirichlet hot wall ($T = 1$) at $x=0$, cold wall ($T = 0$) at $x=1$; Neumann adiabatic ($\partial T/\partial y = 0$) at $y=0$ and $y=1$.
- **Core Numerical Innovations:**
  - **Forward-Mode AD:** Spatial derivatives via `jax.linearize` / JVP (`value_and_jacfwd`) and pure diagonal Laplacians (`hessian_diag_fwd`) avoiding full Hessians.
  - **Adaptive Pseudo-Time Stepping (PTS):** Dynamic $\tau$ scheduling via Barzilai-Borwein and residual shrinkage.
  - **Optimizer:** SOAP (Second-order Clipping / Preconditioner) with warmup cosine decay.

---

## 2. Directory Structure

```text
├── nc_porous/
│   ├── __init__.py
│   ├── config.py              # MLCollections ConfigDict (Physics, Arch, Optim, PTS)
│   ├── models.py              # PorousNaturalConvection2D (subclass of ForwardBVP)
│   ├── train.py               # train_porous function (training loop with PTS updates)
│   └── utils.py               # Boundary sampler, compute_nusselt_balance, export_to_vtk
├── jaxpi2/                    # Base framework core (models, derivatives, optimizers)
│   ├── derivatives.py         # value_and_jacfwd, hessian_diag_fwd
│   ├── models.py              # ForwardBVP, TrainState, PTS / loss weighting logic
│   └── samplers.py            # UniformSampler
├── outputs/                   # Auto-generated checkpoints & history
│   ├── experiments_manifest.json # Registry of all sweep runs
│   └── <wandb_name>/          # Model checkpoints (.orbax) and history.pkl
├── run_sweep.py               # Headless parametric sweep runner (OOM-safe)
├── analysis.ipynb             # Post-processing (Loss curves, Nusselt balance, VTK export)
└── README.md                  # Project context & architecture spec