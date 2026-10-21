"""Residual maps and interface profiles of trained fluid_porous_layer runs.

For each run, the FV residual the run was trained on (same scheme, h, epsilon and
momentum form, read from run_info.json) is evaluated on a dense grid of control-volume
centres. Runs trained before the drag-quadrature fix (commit bc791b5) are evaluated with
the scheme of that time, taken from git. Writes to outputs/fluid_porous_layer/figures/:
  residual_maps.{png,pdf}          log10 |residual| per equation and run
  residual_x_profiles.{png,pdf}    RMS over y of each residual as a function of x
  interface_profiles.{png,pdf}     u, v, p, T, dT/dx across the left interface
  residual_regions.csv             RMS residual per region (fluid / interface band / porous / walls)

  python scripts/residual_maps.py
"""
import csv
import json
import os
import subprocess
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cases import experiment
from cases.fluid_porous_layer.config import aluminum
from porous_pinn import numerical_pde

FIG_DIR = os.path.join(REPO_ROOT, "outputs", "fluid_porous_layer", "figures")
DRAG_FIX_COMMIT = "bc791b5"  # ffv_compact drag quadrature changed here
RUNS = [  # (label, run directory name suffix after "fluid_porous_layer_aluminum_")
    ("gao, ε=0.4, h=1/80, drag fixed", "dragfix_seed0_n80"),
    ("gao, ε=0.4, h=1/80", "hconv_seed0_n80"),
    ("gao, ε=0.4, h=1/160", "hconv_seed0_n160"),
    ("hirata, ε=0.385, h=1/80", "eps0385_seed0_momentum_formhirata"),
]
EQS = [("u", "x-momentum"), ("v", "y-momentum"), ("p", "continuity"), ("T", "energy")]
N_GRID = 400
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SERIES = [("#2a78d6", "-"), ("#eb6834", "--"), ("#1baf7a", "-."), ("#eda100", ":")]


def scheme_module(run_dir):
  """numerical_pde as it was when the run was trained."""
  info = json.load(open(os.path.join(run_dir, "run_info.json")))
  commit = info.get("git_commit")
  is_old = subprocess.run(["git", "merge-base", "--is-ancestor", DRAG_FIX_COMMIT, commit],
                          cwd=REPO_ROOT).returncode != 0
  if not is_old:
    return numerical_pde, info
  src = subprocess.run(["git", "show", f"{commit}:porous_pinn/numerical_pde.py"], cwd=REPO_ROOT,
                       capture_output=True, text=True, check=True).stdout
  mod = types.ModuleType(f"numerical_pde_{commit[:7]}")
  exec(src, mod.__dict__)
  return mod, info


def load(suffix):
  run_name = f"fluid_porous_layer_aluminum_{suffix}"
  run_dir = os.path.join(experiment.case_outputs_dir("fluid_porous_layer"), run_name)
  module, info = scheme_module(run_dir)
  cfg = ml_collections.ConfigDict(info["config"])
  model, _, _ = experiment.load_run("fluid_porous_layer", suffix, aluminum)
  return model, cfg, module, info


def residual_field(model, cfg, module):
  h = float(cfg.diff.h_stencil)
  c = np.linspace(0.5 * h, 1.0 - 0.5 * h, N_GRID)  # every CV fully inside the cavity
  X, Y = np.meshgrid(c, c)
  phys = cfg.physics
  f = jax.jit(jax.vmap(lambda a, b: module.compute_pde_residuals_nd(
      model.neural_net, model.state.params, a, b, h, phys, cfg.diff.scheme)))
  out = {k: [] for k, _ in EQS}
  xs, ys = X.ravel(), Y.ravel()
  for i in range(0, xs.size, 20000):
    r = f(jnp.asarray(xs[i:i + 20000]), jnp.asarray(ys[i:i + 20000]))
    for k, _ in EQS:
      out[k].append(np.asarray(r[k]))
  return X, Y, {k: np.concatenate(v).reshape(X.shape) for k, v in out.items()}, h


def regions(X, Y, h, phys):
  xi = (phys.porous_x_start, phys.porous_x_end)
  d_int = np.minimum(np.abs(X - xi[0]), np.abs(X - xi[1]))
  d_wall = np.minimum.reduce([X, 1 - X, Y, 1 - Y])
  porous = (X > xi[0]) & (X < xi[1])
  band = d_int < h
  wall = (d_wall < 2 * h) & ~band
  return {"fluid bulk": ~porous & ~band & ~wall, "interface band (|x-x_i|<h)": band,
          "porous bulk": porous & ~band & ~wall, "near walls (<2h)": wall}


def main():
  experiment.apply_runtime(aluminum.get_config())
  os.makedirs(FIG_DIR, exist_ok=True)
  plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
                       "axes.edgecolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
  loaded = [(label, *load(suffix)) for label, suffix in RUNS]

  # ---------------- residual maps ----------------
  fig, axes = plt.subplots(len(loaded), len(EQS), figsize=(11, 2.75 * len(loaded)), constrained_layout=True)
  rows = []
  fields = {}
  for i, (label, model, cfg, module, info) in enumerate(loaded):
    X, Y, R, h = residual_field(model, cfg, module)
    fields[label] = (X, Y, R, h, cfg)
    for k, name in EQS:
      for reg, mask in regions(X, Y, h, cfg.physics).items():
        rows.append(dict(run=label, equation=name, region=reg, rms=float(np.sqrt(np.mean(R[k][mask] ** 2))),
                         max_abs=float(np.max(np.abs(R[k][mask])))))
    for j, (k, name) in enumerate(EQS):
      ax = axes[i, j]
      L = np.log10(np.abs(R[k]) + 1e-12)
      hi = np.percentile(L, 99.9)
      im = ax.imshow(L, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()], cmap=BLUES,
                     vmin=hi - 4, vmax=hi, interpolation="nearest")
      for xi in (cfg.physics.porous_x_start, cfg.physics.porous_x_end):
        ax.axvline(xi, color=INK_2, lw=0.8, ls="--")
      ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
      if i == 0:
        ax.set_title(f"{name}: " + r"$\log_{10}|r_{%s}|$" % k, color=INK, fontsize=9.5)
      if j == 0:
        ax.set_ylabel(label, color=INK, fontsize=9)
      cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
      cb.ax.tick_params(labelsize=7, colors=INK_2)
  fig.suptitle("FV residual each run was trained on, evaluated at 400×400 control-volume centres "
               "(dashed: fluid/porous interfaces; colour range = top 4 decades)", color=INK, fontsize=10)
  for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"residual_maps.{ext}"), dpi=200)
  plt.close(fig)

  with open(os.path.join(FIG_DIR, "residual_regions.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["run", "equation", "region", "rms", "max_abs"])
    w.writeheader(); w.writerows(rows)

  # ---------------- RMS over y of each residual, as a function of x ----------------
  fig, axes = plt.subplots(1, len(EQS), figsize=(13, 3.4), constrained_layout=True, sharex=True)
  for (label, (X, Y, R, h, cfg)), (color, ls) in zip(fields.items(), SERIES):
    for ax, (k, name) in zip(axes, EQS):
      ax.semilogy(X[0], np.sqrt(np.mean(R[k] ** 2, axis=0)), color=color, ls=ls, lw=1.2, label=label)
  for ax, (k, name) in zip(axes, EQS):
    for xi in (1 / 3, 2 / 3):
      ax.axvline(xi, color=INK_2, lw=0.8, ls="--")
    ax.set_title(f"{name}: RMS over y of " + r"$r_{%s}$" % k, color=INK, fontsize=9.5)
    ax.set_xlabel("x"); ax.grid(color=GRID, lw=0.5, which="both")
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8.5)
  for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"residual_x_profiles.{ext}"), dpi=200)
  plt.close(fig)

  # ---------------- interface profiles (sub-h resolution) ----------------
  x_i = loaded[0][2].physics.porous_x_start
  xs = np.linspace(x_i - 0.06, x_i + 0.06, 1201)
  heights = (0.25, 0.5, 0.75)
  quantities = [("u", 0), ("v", 1), ("p", 2), ("T", 3), (r"$\partial T/\partial x$", None)]
  fig, axes = plt.subplots(len(heights), len(quantities), figsize=(12.5, 7.6), constrained_layout=True)
  for (label, model, cfg, module, info), (color, ls) in zip(loaded, SERIES):
    net = jax.jit(jax.vmap(lambda a, b: jnp.stack(model.neural_net(model.state.params, a, b))))
    dTdx = jax.jit(jax.vmap(jax.grad(lambda a, b: model.neural_net(model.state.params, a, b)[3], argnums=0)))
    for r, yv in enumerate(heights):
      Q = np.asarray(net(jnp.asarray(xs), jnp.full(xs.shape, yv)))
      G = np.asarray(dTdx(jnp.asarray(xs), jnp.full(xs.shape, yv)))
      for c, (qname, idx) in enumerate(quantities):
        ax = axes[r, c]
        ax.plot(xs, Q[:, idx] if idx is not None else G, color=color, ls=ls, lw=1.3, label=label)
  for r, yv in enumerate(heights):
    for c, (qname, _) in enumerate(quantities):
      ax = axes[r, c]
      ax.axvline(x_i, color=INK_2, lw=0.8, ls="--")
      ax.grid(color=GRID, lw=0.5)
      for side in ("top", "right"):
        ax.spines[side].set_visible(False)
      if r == 0:
        ax.set_title(qname, color=INK)
      if c == 0:
        ax.set_ylabel(f"y = {yv}", color=INK)
      if r == len(heights) - 1:
        ax.set_xlabel("x  (interface at x = 1/3)")
  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8.5)
  fig.suptitle("Profiles across the left fluid/porous interface (1201 points over ±0.06, i.e. sub-h resolution)",
               color=INK, fontsize=10)
  for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"interface_profiles.{ext}"), dpi=200)
  plt.close(fig)

  # ---------------- summary ----------------
  print(f"{'run':34s} {'equation':11s} " + "  ".join(f"{r:>22s}" for r in regions(np.zeros(1), np.zeros(1), 1, aluminum.get_config().physics)))
  for label, *_ in loaded:
    for _, name in EQS:
      sel = {r["region"]: r for r in rows if r["run"] == label and r["equation"] == name}
      print(f"{label:34s} {name:11s} " + "  ".join(f"{sel[k]['rms']:10.2e} ({sel[k]['max_abs']:8.1e})" for k in sel))


if __name__ == "__main__":
  main()
