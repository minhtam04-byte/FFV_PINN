# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a paper (`paper_outline.md`) on PINNs for conjugate natural convection in porous media, solved with JAX/Flax. It combines finite-volume residuals (inspired by FFV-PINN, Wei et al. 2025) with adaptive pseudo-time stepping (PTS) from Wang et al. 2026, whose `jaxpi2` framework is vendored as `jaxpi/`. The benchmarks come from Gao et al. 2017 (IJHMT 105). The governing equations are in `README.md`. Reference PDFs are in `references/` (git-ignored).

Open research tasks, known bugs and agreed paper framing are tracked in **`ROADMAP.md`**. Read it before starting work, and update it when an item is done.

## Environment

- Virtualenv `./venv` (Python 3.14, CUDA JAX); activate with `source venv/bin/activate.fish`. Entry points add the repo root to `sys.path` themselves, so no install is needed. The editable install in the venv still points at the removed `jaxpi2/`; if you reinstall, use `pip install -e . --no-deps`, because `pyproject.toml` pins a different JAX/CUDA build than the venv has.
- There is no test suite or linter.

## Running

- CLI: `python scripts/train.py --config=cases/<case>/config/<file>.py --exp_name=<suffix> [--config.a.b=value ...]`. The case comes from the config path. The run goes to `outputs/<case>/<wandb.name>_<exp_name>/` with a `run_info.json` (resolved config, git commit, dirty flag). The script refuses to overwrite an existing run.
- Notebooks: `notebooks/<case>.ipynb`. Cell 0 imports the helpers and binds them to the case with `functools.partial`: `run_ablation_experiment(exp_name, org_config, modify_config_fn)` and `load_saved_model(exp_name, org_config, step=None)`, both pointing at `outputs/<case>/`. Analysis and plotting functions are still defined inside the notebooks.
- Run folders keep their historical names (`PorousNC_layer_<exp>`, `1D_Conjugate_Conduction_<exp>`, ...), so `load_saved_model(exp_name, ...)` still finds old runs.

## Architecture

- `jaxpi/` is the framework. Keep it close to upstream; algorithm changes belong in `porous_pinn/` or `cases/`.
  - `models.py`: `PINN` base class with `ForwardBVP`/`ForwardIVP`, factories (`create_arch`, `create_optimizer` for SOAP/Adam, `create_train_state`), and `TrainState` (`loss_weights`, `pts_weights`, `prev_params`).
  - `step`, `update_loss_weights` and `update_pts_weights` are `jax.shard_map` over a 1-D `"batch"` mesh. Batch sizes must be divisible by `jax.device_count()`.
  - Model methods use `@partial(jit, static_argnums=(0,))`. `PINN.__hash__` returns a process-unique id; do not replace it.
  - PTS (`compute_pts_weights`): τ = ‖Δresidual‖/‖Δsol_net‖ between `params` and `prev_params` (Barzilai–Borwein). It is multiplied by a cosine shrink on the log₁₀ loss drop and clipped to [1e-2, 100]. Loss weighting is grad-norm balancing.
- `porous_pinn/` holds the shared 2D code.
  - `numerical_pde.py`: porous coefficients (`compute_porous_and_buoyancy`, which handles the piecewise porous zone `porous_x_start`–`porous_x_end`) and the FV schemes, dispatched by `compute_pde_residuals_nd(..., scheme=)` to `ffv`, `ffv_continuous` or `can_ffv`.
  - `models.py`: `PorousNaturalConvection2D`, used by both 2D cases (fluid–porous–fluid is just a config with `porous_x_start/end`).
  - `batching.py`: `create_batches` overwrites `diff.h_stencil = L / n_x`, requires square cells, and keeps stencil centres one reach from the walls. Fixed grid uses cell centres; `use_continuous_mesh` samples uniform random centres instead.
  - `experiment.py`: run, reload and outputs-path helpers.
- `cases/<case>/` contains `train.py` (case trainer, `outputs_dir=None` defaults to `<cwd>/outputs`) and `config/*.py` (`ml_collections` `get_config()`). `conduction_1d` also has its own `models.py`, `numerical_pde.py` and `experiment.py`.
- `legacy/` and `notebooks/legacy/` hold old, non-runnable code. Ignore them unless asked.

### Name-keyed contracts (enforced by asserts)
`variables = ("u", "v", "p", "T")` on the model; `r_net` returns a dict keyed by the variable each residual evolves in pseudo-time (momentum-x → `u`, continuity → `p`, energy → `T`); `config.pseudo_time.pts_weights` uses the same keys; `config.loss_weighting.loss_weights` keys must equal those of `losses()` (residuals are auto-named `<var>_res`).

### Schemes (`config.diff.scheme`)
- `ad`: pointwise forward-mode AD residuals. Note that the energy equation uses `k·∇²T` piecewise and has no flux-continuity condition at the porous interface.
- FV schemes evaluate the network at faces/neighbours at spacing `h`. `sol_net` returns a stencil-weighted cell average so that the PTS damping term matches the discrete residual.

## Conventions and gotchas

- Matmul precision: `cases/fluid_porous_layer/train.py` sets `jax_default_matmul_precision="highest"` when imported; `cases/porous_cavity/train.py` does not. Evaluating the same checkpoint with and without it changes ψ_max by ~1% (see `ROADMAP.md`). When writing analysis code, set precision explicitly.
- When asked to restructure without changing behaviour, verify it numerically: compare residuals, losses and batches before and after on fixed inputs. GPU `jnp.dot` can differ by 1 ULP between runs, so compare against a repeated run.
- Comments, docstrings and printed messages are mostly in Vietnamese; follow the surrounding file.
- New config variants go in as new files under `cases/<case>/config/`. Notebooks override fields through `modify_config_fn`; the CLI does it with `--config.x=...`.
