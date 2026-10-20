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
├── jaxpi/                  # PINN framework adapted from jaxpi2 (Wang et al., 2026)
├── porous_pinn/            # Shared code for the 2D porous-convection cases
│   ├── numerical_pde.py    # Porous coefficients + FV residual schemes (ffv, ffv_continuous, can_ffv)
│   ├── models.py           # PorousNaturalConvection2D (subclass of ForwardBVP)
│   ├── boundary.py         # Wall boundary-point sampling
│   ├── batching.py         # create_batches: collocation points per diff scheme
│   ├── experiment.py       # run_ablation_experiment, load_saved_model, case_outputs_dir
│   ├── samplers.py         # R3CollocationSampler
│   ├── vrba.py             # vRBA weights
│   └── plotting.py         # Plotting and Nusselt helpers
├── cases/                  # One sub-package per benchmark (trainer + configs)
│   ├── conduction_1d/      # Gao et al. 2017, Sec. 4.1 (analytical solution)
│   ├── porous_cavity/      # Homogeneous porous cavity (Gao et al. 2017, Table 1)
│   └── fluid_porous_layer/ # Gao et al. 2017, Sec. 4.3 (glass / aluminum beads)
├── scripts/
│   ├── train.py            # CLI training entry point (config overrides, run_info.json)
│   ├── plot_log_history.py # Plot loss / tau from a pasted training log
│   └── pack_project.py     # Concatenate sources into core_codebase.txt
├── notebooks/              # Experiments and analysis per case (legacy/ = old, not runnable)
├── outputs/<case>/         # Checkpoints, history.pkl, figures (git-ignored)
├── references/             # Reference papers (git-ignored)
└── legacy/                 # Old code kept for reference (not runnable)
```

## 3. Usage

```bash
source venv/bin/activate.fish

# Train one run; results go to outputs/<case>/<wandb.name>_<exp_name>/
python scripts/train.py \
    --config=cases/fluid_porous_layer/config/base_line_case_2.py \
    --exp_name=ffv_continuous_pts_seed1 \
    --config.diff.scheme=ffv_continuous \
    --config.diff.use_continuous_mesh=True \
    --config.seed=1
```

Notebooks in `notebooks/` import the same helpers (`porous_pinn.experiment`) and
read/write the same `outputs/<case>/` folders. Reload a run with
`load_saved_model(exp_name, org_config, outputs_dir=case_outputs_dir("<case>"))`.
