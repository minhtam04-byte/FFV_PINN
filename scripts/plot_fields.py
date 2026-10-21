"""Streamlines and isotherms of trained fluid_porous_layer runs (Gao et al. 2017, Fig. 7 style).

  python scripts/plot_fields.py "label=run_suffix" [...] [--tag=name]
run_suffix is the part after "fluid_porous_layer_aluminum_". Writes
outputs/fluid_porous_layer/figures/fields{_tag}.{png,pdf}.
"""
import json
import os
import sys

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

from cases import experiment
from cases.fluid_porous_layer.config import aluminum

FIG_DIR = os.path.join(REPO_ROOT, "outputs", "fluid_porous_layer", "figures")
INK, INK_2, POROUS = "#0b0b0b", "#52514e", "#ecebe6"
N = 401


def load(suffix):
  run_dir = os.path.join(experiment.case_outputs_dir("fluid_porous_layer"), f"fluid_porous_layer_aluminum_{suffix}")
  cfg = ml_collections.ConfigDict(json.load(open(os.path.join(run_dir, "run_info.json")))["config"])
  features = bool(cfg.get("model", {}).get("interface_features", False))

  def same_network(c):
    c.model.interface_features = features
    return c

  model, _, _ = experiment.load_run("fluid_porous_layer", suffix, aluminum, modify_config_fn=same_network)
  return model, cfg


def fields(model, cfg):
  x = np.linspace(0, 1, N)
  X, Y = np.meshgrid(x, x)
  out = jax.jit(jax.vmap(lambda a, b: jnp.stack(model.neural_net(model.state.params, a, b))))(
      jnp.asarray(X.ravel()), jnp.asarray(Y.ravel()))
  u, v, p, T = (np.asarray(out[:, k]).reshape(N, N) for k in range(4))
  scale = np.sqrt(cfg.physics.Ra * cfg.physics.Pr)  # alpha_f / L velocity units
  dy = x[1] - x[0]
  psi = np.concatenate([np.zeros((1, N)), np.cumsum(0.5 * (u[1:] + u[:-1]) * dy, axis=0)]) * scale
  return X, Y, psi, T


def main():
  args = sys.argv[1:]
  tag = next((a.split("=", 1)[1] for a in args if a.startswith("--tag=")), "")
  runs = [tuple(a.split("=", 1)) for a in args if not a.startswith("--tag=")]
  experiment.apply_runtime(aluminum.get_config())
  plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
                       "axes.edgecolor": INK_2})
  fig, axes = plt.subplots(2, len(runs), figsize=(4.3 * len(runs), 8.4), constrained_layout=True, squeeze=False)
  for j, (label, suffix) in enumerate(runs):
    model, cfg = load(suffix)
    X, Y, psi, T = fields(model, cfg)
    xs, xe = cfg.physics.porous_x_start, cfg.physics.porous_x_end
    psi_max = np.abs(psi).max()

    ax = axes[0, j]
    ax.axvspan(xs, xe, color=POROUS, zorder=0)
    levels = np.linspace(-psi_max, psi_max, 31)[1:-1]
    levels = levels[np.abs(levels) > 1e-9]
    ax.contour(X, Y, psi, levels=levels, colors=INK, linewidths=0.7, negative_linestyles="solid")
    for side, (x0, x1) in (("left", (0.0, xs)), ("right", (xe, 1.0))):
      mask = (X >= x0) & (X <= x1)
      k = np.argmax(np.where(mask, np.abs(psi), -1))
      ax.plot(X.flat[k], Y.flat[k], marker="+", color=INK, ms=7)
      ax.text(X.flat[k], Y.flat[k] - 0.06, f"|ψ|max = {np.abs(psi.flat[k]):.2f}", ha="center", va="top", fontsize=8,
              color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    ax.set_title(f"{label}\nstreamlines (30 equal increments)", color=INK, fontsize=9.5)

    ax = axes[1, j]
    ax.axvspan(xs, xe, color=POROUS, zorder=0)
    cs = ax.contour(X, Y, T, levels=np.linspace(0, 1, 17)[1:-1], colors=INK, linewidths=0.7)
    ax.clabel(cs, levels=[0.5], fmt="0.5", fontsize=8, inline=True)
    ax.set_title("isotherms θ = 0.0625 … 0.9375 (step 1/16)", color=INK, fontsize=9.5)

    for ax in axes[:, j]:
      ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
      ax.set_xticks([0, 1 / 3, 2 / 3, 1]); ax.set_xticklabels(["0", "1/3", "2/3", "1"]); ax.set_yticks([0, 0.5, 1])
      for xi in (xs, xe):
        ax.axvline(xi, color=INK_2, lw=0.6, ls="--")
    print(f"{label}: |psi|max = {psi_max:.3f}")
  fig.suptitle("Aluminium beads (Gao et al. 2017, Sec. 4.3)\nRa = 3.7e6, Pr = 6.44, Da = 1.534e-5, k_e/k_f = 37.47; "
               "shaded: porous layer", color=INK, fontsize=10)
  name = f"fields_{tag}" if tag else "fields"
  for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=200)


if __name__ == "__main__":
  main()
