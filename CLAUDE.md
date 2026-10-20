# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a paper (`paper_outline.md`) on PINNs for conjugate natural convection in porous media, written in JAX/Flax. It combines finite-volume residuals (inspired by FFV-PINN, Wei et al. 2025) with adaptive pseudo-time stepping (PTS) from Wang et al. 2026, whose `jaxpi2` framework is vendored as `jaxpi/`. The authors' guides for that framework are in `docs/` (`guide/`, `methods/`). Benchmarks come from Gao et al. 2017 (IJHMT 105). The governing equations are in `README.md`, and reference PDFs are in `references/` (git-ignored).

Open research tasks, known issues and the agreed paper framing are tracked in **`ROADMAP.md`**. Read it before starting work, and update it when an item is done.

## Environment

- Virtualenv `./venv` (Python 3.14, CUDA JAX): `source venv/bin/activate.fish`. Entry points put the repo root on `sys.path` themselves. If you reinstall, use `pip install -e . --no-deps`, because `pyproject.toml` pins a different JAX/CUDA build than the venv has.
- W&B is disabled unless `WANDB_MODE` is set (`cases/experiment.py` sets the default to `disabled`).

## Commands

- Train: `python scripts/train.py --config=cases/<case>/config/<variant>.py --exp_name=<suffix> [--config.a.b=value ...]`. The case comes from the config path. Continue an interrupted run with `--config.training.resume=True`; without that flag, the script refuses to reuse an existing run directory.
- Tests: `pytest` runs the fast suite (~30 s, CPU). `pytest -m slow` runs the training tests (~2 min). Run a single test with `pytest tests/test_fv_schemes.py::TestInterfaceConservation -q`.
- After an **intended** numerical change, regenerate the golden file with `python tests/make_golden.py` and say so in the commit message. Otherwise `tests/test_regression.py` failing means numerics changed unintentionally.

## Architecture

- `jaxpi/` is the framework. Keep it close to upstream; changes belong in `porous_pinn/` or `cases/`.
  - `models.py`: the `PINN` base class with `ForwardBVP`/`ForwardIVP`, `create_model`, and `TrainState` (`loss_weights`, `pts_weights`, `prev_params`).
  - `training.py`: `train`/`train_loop`. Each iteration does step → PTS update → GradNorm update, calls the evaluator every `logging.log_every_steps`, and handles checkpoints at `<saving.ckpt_path>/<wandb.name>/ckpt`, resume, and `stop_fn`.
  - `step` and the weight updates are `jax.shard_map` over a 1-D `"batch"` mesh, so batch sizes must be divisible by `jax.device_count()`.
  - Model methods use `@partial(jit, static_argnums=(0,))`, and `PINN.__hash__` is a process-unique id. Do not replace it.
  - PTS: τ = ‖Δresidual‖/‖Δsol_net‖ between `params` and `prev_params`, multiplied by a cosine shrink on the log₁₀ loss drop and clipped to [1e-2, 100].
- `porous_pinn/` is the physics library for the 2D cases. It has no training orchestration.
  - `numerical_pde.py`: porous coefficients and the FV schemes `ffv`, `ffv_continuous` and `can_ffv`. All three are derivative-free except `can_ffv`.
  - `models.py`: `PorousNaturalConvection2D`, shared by both 2D cases. The fluid–porous–fluid case is just `porous_x_start/end` in its config.
  - `batching.py`: `create_batches` overwrites `diff.h_stencil = L / n_x`, keeps stencil centres one reach from the walls, and samples either fixed cell centres or random centres (`use_continuous_mesh`).
  - `evaluators.py`: `HistoryEvaluator` (keeps every logged dict plus `loss/total`) and `ConvectionEvaluator` (ψ_max, wall Nu, `error/mass_imbalance`, `error/Nu_imbalance`, `error/*_vs_ref` from `config.reference`).
- `cases/` holds the problem definitions and the runner.
  - `experiment.py` contains the `CASES` registry (case → model, batch builder, evaluator), `train_case(case, config)`, the notebook helpers `run_experiment(case, exp_name, org_config, modify_config_fn)` and `load_run(case, exp_name, org_config, step)`, and `apply_runtime`, which applies `config.jax_matmul_precision`.
  - `cases/<case>/config/`: `base.py` holds the full tree; variants are thin overrides (`fluid_porous_layer/{aluminum,glass}.py`). A run lives in `outputs/<case>/<wandb.name>/` with `ckpt/`, `history.pkl` and `run_info.json`.
  - `conduction_1d` has its own `models.py`, `numerical_pde.py`, `batching.py` and `evaluators.py`.
- `notebooks/<case>.ipynb`: cell 0 binds `run_ablation_experiment` and `load_saved_model` to the case with `functools.partial`. `legacy/` and `notebooks/legacy/` hold retired code (R3, vRBA, old configs and trainers); ignore them unless asked.

### Name-keyed contracts (asserted; checked by `tests/test_configs.py`)
- The model declares `variables = ("u", "v", "p", "T")`.
- `r_net` returns a dict keyed by the variable each residual evolves in pseudo-time (momentum-x → `u`, continuity → `p`, energy → `T`).
- `pseudo_time.pts_weights` uses the same keys.
- `loss_weighting.loss_weights` keys equal the keys returned by `losses()`; residual losses are auto-named `<var>_res`.

### Schemes (`config.diff.scheme`)
- `ad`: pointwise forward-mode AD. The energy equation uses `k·∇²T` piecewise, with no flux-continuity condition at the porous interface; `tests/test_fv_schemes.py` demonstrates this.
- FV schemes are second order (verified by manufactured-solution tests). `ffv_continuous` is exact for a flux-continuous temperature at any interface offset; `ffv` is exact only when the interface lies on a cell face. `sol_net` returns a stencil-weighted cell average so that the PTS damping term matches the discrete residual.

## Conventions

- Comments, docstrings and printed messages are mostly in Vietnamese; follow the surrounding file.
- New experiments are new thin config variants or `--config.x=...` overrides, never copies of `base.py`.
- Evaluate saved runs through `cases.experiment.load_run`, which applies the run's matmul precision. Evaluating the same checkpoint at TF32 instead of "highest" changes ψ_max by ~1%.
