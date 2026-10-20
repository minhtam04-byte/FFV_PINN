# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a paper (see `paper_outline.md`) on PINNs for natural convection in porous media, solved with JAX/Flax. It compares automatic differentiation against finite-volume-style stencil residuals (FFV-PINN, CAN-PDE), and uses adaptive pseudo-time stepping (PTS) from Wang et al. 2026. The governing equations (Darcy–Brinkman–Forchheimer momentum, continuity, energy) and boundary conditions are written out in `readme.md`. Its "Directory Structure" section is out of date: `jaxpi2/`, `run_sweep.py` and `analysis.ipynb` no longer exist.

## Environment

- Use the local virtualenv `./venv` (Python 3.14, CUDA JAX): `source venv/bin/activate.fish` (fish shell).
- The core framework installs as an editable package: `pip install -e .`. The distribution is named `jaxpi2` in `pyproject.toml`, but the import name is `jaxpi`.
- There is no test suite, linter config or build step. Code is checked by running experiments.
- `python pack_project.py` concatenates the core sources into `core_codebase.txt` so they can be shared as context.

## Running experiments

Experiments are run from each case's `run_experiment.ipynb`, not from CLI entry points. Each notebook:
1. puts the repo root on `sys.path`;
2. defines `create_batches(config)` to build the collocation sampler;
3. calls `run_ablation_experiment(exp_name, org_config, modify_config_fn)`, which loads a config module's `get_config()`, applies overrides, sets `wandb.name = f"{wandb.name}_{exp_name}"`, and calls the case's `train.train_porous(config, samplers)`;
4. reloads finished runs with `load_saved_model(exp_name, ...)`, which must rebuild the identical config to restore the checkpoint.

Outputs go to `os.getcwd()/outputs/<wandb.name>/`: an Orbax checkpoint in `ckpt/` and a pickled `history.pkl` holding the loss, per-term losses and `tau_<var>` histories. Paths are relative to the working directory, so run notebooks from inside their case directory (for example `nc_porous_layer/`).

## Architecture

### `jaxpi/`: the shared PINN framework (adapted from Sifan Wang's jaxpi)
- `models.py`: `PINN` base class with the subclasses `ForwardBVP` (steady problems, used here) and `ForwardIVP`. It also has the factories `create_arch`, `create_lr_schedule`, `create_optimizer` (SOAP/Adam) and `create_train_state`. `TrainState` carries `loss_weights`, `pts_weights` and `prev_params`.
  - `step`, `update_loss_weights` and `update_pts_weights` are `jax.shard_map` functions over a 1-D `"batch"` device mesh. Batch sizes must be divisible by `jax.device_count()`.
  - Model methods use `@partial(jit, static_argnums=(0,))`. `PINN.__hash__` returns a process-unique id so that a recreated model does not reuse stale traces. Don't replace this hash.
  - **PTS** (`compute_pts_weights`): for each variable, τ = ‖Δresidual‖ / ‖Δsolution‖ between `params` and `prev_params` (a Barzilai–Borwein estimate). It is scaled by a cosine shrink factor driven by the log₁₀ drop in loss (`pseudo_time.shrink`) and clipped to [1e-2, 100].
  - **Loss weighting** (`compute_loss_weights`): gradient-norm balancing across the keys of `losses()`.
- `derivatives.py`: forward-mode derivatives (`value_and_jacfwd`, `hessian_diag_fwd`) that build Laplacians from the Hessian diagonal only.
- `archs.py`: `Mlp`, `ModifiedMlp` and `PirateNet`, with Fourier and periodic embeddings. `samplers.py`: `UniformSampler`, `MeshSampler` and the FVM samplers.

### Name-keyed contracts between a model and its config
These keys are matched **by name** and checked with asserts, so they must stay consistent:
- the model's class attribute `variables = ("u", "v", "p", "T")`;
- the dict returned by `r_net`, keyed by the variable each residual evolves in pseudo-time (momentum-x → `"u"`, continuity → `"p"`, energy → `"T"`);
- the keys of `config.pseudo_time.pts_weights`;
- the keys of `config.loss_weighting.loss_weights`, which must equal the keys returned by `losses()`. Residual terms are automatically named `<var>_res`; boundary terms are named by the model (for example `u_bc`, `T_dirichlet_bc`, `T_neumann_bc`).

### Case directories (each has `models.py`, `train.py`, `config/*.py`, `run_experiment.ipynb`)
- `nc_porous/`: homogeneous porous cavity. It also holds the **shared physics modules** that other cases import:
  - `numerical_pde.py`: the stencil residual schemes, dispatched by `compute_pde_residuals_nd(..., scheme=)` to `ffv`, `ffv_continuous`, `can`, `can_ffv` or `nd`. It also has `compute_porous_and_buoyancy`, which handles the piecewise porous zone `porous_x_start`–`porous_x_end`.
  - `utils.py`: boundary sampling and Nusselt/VTK post-processing. `sampler.py`: `R3CollocationSampler` (residual-based resampling; it did not work, per the paper outline).
- `nc_porous_layer/`: a fluid–porous layer cavity (conjugate convection, `lambda_k` conductivity ratio). It imports from `nc_porous`.
- `nc_porous_1d/`: 1-D conjugate heat transfer. `mixed_convection/`: an older case.
- `old_code/`: legacy code; ignore it unless asked.

Each model subclasses `ForwardBVP` and implements `neural_net` (4 outputs: u, v, p, T), `r_net`, `losses`, and, for stencil schemes, `sol_net`.

### Differentiation schemes (`config.diff.scheme`)
- `"ad"`: pointwise residuals from forward-mode autodiff, with collocation points sampled uniformly over the whole domain.
- Stencil schemes (`ffv`, `ffv_continuous`, `can`, `can_ffv`, `nd`): residuals come from neighbour evaluations at spacing `h`. In the notebook, `create_batches` **overwrites `diff.h_stencil = L / n_x`**, requires square cells, and keeps points one stencil reach (two for `nd`) away from the walls. `diff.use_continuous_mesh` selects random points over that inner region instead of fixed cell centres.
- For stencil schemes, `sol_net` returns a stencil-weighted cell average rather than a point value. This makes the solution used in the PTS damping term match the discrete residual.

### Training loop (`<case>/train.py: train_porous`)
Each step does a gradient step, then (on their `update_schedule`) a PTS τ update using only the interior batch `batch["eqn"]`, then a loss-weight update. Logging happens every `logging.log_every_steps` steps, and checkpoints are written every `saving.save_every_steps` steps. Setting `training.transfer_learning` with `pretrained_ckpt_dir` initialises the parameters from an earlier run. `jax_default_matmul_precision` is set to `"highest"`.

## Conventions
- Comments, docstrings and printed messages are mostly in Vietnamese. Follow the surrounding file.
- Configs are `ml_collections.ConfigDict` modules exposing `get_config()`. New variants are added as new files under `<case>/config/`, and notebooks override fields through `modify_config_fn`.
