# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a paper (`paper_outline.md`) on PINNs for conjugate natural convection in porous media, written in JAX/Flax. It combines finite-volume residuals (inspired by FFV-PINN, Wei et al. 2025) with adaptive pseudo-time stepping (PTS) from Wang et al. 2026, whose `jaxpi2` framework is vendored as `jaxpi/`. The authors' guides for that framework are in `docs/` (`guide/`, `methods/`). Benchmarks come from Gao et al. 2017 (IJHMT 105). The governing equations are in `README.md`, and reference PDFs are in `references/` (git-ignored).

Open research tasks, known issues and the agreed paper framing are tracked in **`ROADMAP.md`** (current status and priorities at the top). Session handoff notes are in `notes/`; the latest is `notes/2026-10-21_session.md`, which holds the key results, the registry of valid and obsolete runs, and the figure index. Read ROADMAP and the latest note before starting work, and update ROADMAP when an item is done.

## Environment

- Virtualenv `./venv` (Python 3.14, CUDA JAX): `source venv/bin/activate.fish`. Entry points put the repo root on `sys.path` themselves. If you reinstall, use `pip install -e . --no-deps`, because `pyproject.toml` pins a different JAX/CUDA build than the venv has.
- W&B is disabled unless `WANDB_MODE` is set (`cases/experiment.py` sets the default to `disabled`).

## Commands

- Train: `python scripts/train.py --config=cases/<case>/config/<variant>.py --exp_name=<suffix> [--config.a.b=value ...]`. The case comes from the config path. Continue an interrupted run with `--config.training.resume=True`; without that flag, the script refuses to reuse an existing run directory.
- Tests: `pytest` runs the fast suite (~30 s, CPU). `pytest -m slow` runs the training tests (~2 min). Run a single test with `pytest tests/test_fv_schemes.py::TestInterfaceConservation -q`.
- Sweeps: `python scripts/sweep.py --config=... --prefix=<name> [--set KEY=V ...] [--grid KEYS=V1,V2 ...]` runs `train.py` subprocesses one after another. It skips finished runs (`history.pkl` present) and resumes partial ones, writing logs to `outputs/<case>/logs/`. It must not import JAX, which would preallocate 75% of the GPU memory in the parent process.
- Analysis of `fluid_porous_layer` runs (figures go to `outputs/fluid_porous_layer/figures/`):
  - `scripts/residual_maps.py [--tag=x] "label=run_suffix" ...` plots residual maps, residual-vs-x profiles, interface profiles and per-region RMS in a CSV. It evaluates each run with its own scheme, taken from git for runs trained before the drag fix.
  - `scripts/plot_fields.py --tag=x "label=run_suffix" ...` plots streamlines and isotherms.
  - `scripts/plot_beckermann_fig3.py` compares runs with the digitised Fig. 3 of Beckermann et al. (1988), stored in `reference_data/`.
  - `run_suffix` is the part after `fluid_porous_layer_aluminum_`, or a full run-directory name.
- Paper experiments keep their launchers (`.sh`) and analysis scripts in `scripts/<experiment>/`, with a README describing the protocol. Each one resolves the repo root itself. `scripts/table3/` covers Table 3 (homogeneous porous cavity vs Liu et al. 2021) and the speed benchmark. Do not leave experiment scripts in `~` or in a scratchpad.
- After an **intended** numerical change, regenerate the golden file with `python tests/make_golden.py` and say so in the commit message. Otherwise `tests/test_regression.py` failing means numerics changed unintentionally.

## Architecture

- `jaxpi/` is the framework. Keep it close to upstream; changes belong in `porous_pinn/` or `cases/`.
  - `models.py`: the `PINN` base class with `ForwardBVP`/`ForwardIVP`, `create_model`, and `TrainState` (`loss_weights`, `pts_weights`, `prev_params`).
  - `training.py`: `train`/`train_loop`. Each iteration does step → PTS update → GradNorm update, calls the evaluator every `logging.log_every_steps`, and handles checkpoints at `<saving.ckpt_path>/<wandb.name>/ckpt`, resume, and `stop_fn`.
  - `step` and the weight updates are `jax.shard_map` over a 1-D `"batch"` mesh, so batch sizes must be divisible by `jax.device_count()`.
  - Model methods use `@partial(jit, static_argnums=(0,))`, and `PINN.__hash__` is a process-unique id. Do not replace it.
  - PTS: τ = ‖Δresidual‖/‖Δsol_net‖ between `params` and `prev_params`, multiplied by a cosine shrink on the log₁₀ loss drop and clipped to [1e-2, 100].
- `porous_pinn/` is the physics library for the 2D cases. It has no training orchestration.
  - `numerical_pde.py`: porous coefficients and the FV schemes `ffv`, `ffv_continuous`, `ffv_compact` and `can_ffv`. All are derivative-free except `can_ffv`. The module docstring states the governing equations and the references. `physics.momentum_form` (`"gao"` default, or `"hirata"`) is implemented only in `ffv_compact`.
  - `models.py`: `PorousNaturalConvection2D`, shared by both 2D cases. The fluid–porous–fluid case is just `porous_x_start/end` in its config. With `config.model.interface_features = True` (default off) the network input becomes (x, y, |x − x_start|, |x − x_end|), so it can represent gradient kinks at the interfaces; `cases.experiment.build_model` then initialises a 4-input network while `config.input_dim` stays 2. `config.arch` is passed verbatim to the Flax module, so new model options must not go there.
  - `batching.py`: `create_batches` overwrites `diff.h_stencil = L / n_x`, keeps stencil centres one stencil reach from the walls (h, or h/2 for `ffv_compact`, so its control volumes cover the whole cavity), and samples either fixed cell centres or random centres (`use_continuous_mesh`).
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
- `ffv_compact` (recommended; 5 points at h/2 plus 2 drag Gauss points) is second order in the bulk and in cut cells, including the drag. `ffv_continuous` has first-order drag in cut cells (a strict xfail in `TestDragCellAverage`).
- FV schemes are second order in the bulk (verified by manufactured-solution tests). `ffv_continuous` is exact for a flux-continuous temperature at any interface offset; `ffv` is exact only when the interface lies on a cell face. `sol_net` returns a stencil-weighted cell average so that the PTS damping term matches the discrete residual.
- `diff.fuse_pts_stencil=True` (`ffv_compact` only, default off) takes the PTS cell average from the residual's own stencil evaluation (`PorousNaturalConvection2D.r_and_sol_net`). Losses are bitwise equal and gradients agree to ~1e-7 (`tests/test_fused_pts.py`). On an RTX 3060 it cuts the cost of a step with PTS by 30%, from 1.71 to 1.21 s per 100 steps, which is 16% faster than AD with PTS.

## Conventions

- Docstrings are in English, Google style (summary, Args, Returns, Raises). Wherever they use a formula or a result from a paper, they cite it with equation or section numbers checked against `references/*.pdf`, and they mark sources that are not in `references/`. Inline comments and printed messages are mostly in Vietnamese; follow the surrounding file.
- New experiments are new thin config variants or `--config.x=...` overrides, never copies of `base.py`.
- Evaluate saved runs through `cases.experiment.load_run`, which applies the run's matmul precision. Evaluating the same checkpoint at TF32 instead of "highest" changes ψ_max by ~1%.
