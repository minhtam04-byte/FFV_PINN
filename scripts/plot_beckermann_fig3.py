"""Compare trained runs with the measured temperature profiles of Beckermann et al. (1988), Fig. 3.

Reads reference_data/beckermann1988_fig3.csv and the runs listed in RUNS (skipped if
missing), writes
  outputs/fluid_porous_layer/figures/beckermann1988_fig3_comparison.{png,pdf}
  outputs/fluid_porous_layer/figures/beckermann1988_fig3_errors.csv
and prints RMS / max |theta_model - theta_exp| per run and height.

  python scripts/plot_beckermann_fig3.py
"""
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cases import experiment
from cases.fluid_porous_layer.config import exp3_glycerin_glass, exp4_glycerin_aluminum

DATA = os.path.join(REPO_ROOT, "reference_data", "beckermann1988_fig3.csv")
FIG_DIR = os.path.join(REPO_ROOT, "outputs", "fluid_porous_layer", "figures")

EXPERIMENTS = {3: ("(a) Experiment 3: glass beads", exp3_glycerin_glass),
               4: ("(b) Experiment 4: aluminium beads", exp4_glycerin_aluminum)}
RUNS = [  # (label, exp_name suffix, line style)
    ("Gao form (Eq. 2, Gao et al. 2017)", "fig3_seed0_momentum_formgao", "-"),
    ("Hirata form (interface-consistent)", "fig3_seed0_momentum_formhirata", "--"),
]
HEIGHTS = {0.912: ("s", "#2a78d6"), 0.554: ("^", "#eb6834"), 0.204: ("o", "#1baf7a")}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_data():
  rows = [r for r in csv.reader(open(DATA)) if r and not r[0].startswith("#")]
  header, rows = rows[0], rows[1:]
  data = {}
  for r in rows:
    d = dict(zip(header, r))
    data.setdefault(int(d["experiment"]), {}).setdefault(float(d["y_over_L"]), []).append(
        (float(d["xi"]), float(d["theta"])))
  return {e: {h: np.array(sorted(v)) for h, v in hs.items()} for e, hs in data.items()}


def temperature(model, x, y):
  f = jax.jit(jax.vmap(lambda xx, yy: model.neural_net(model.state.params, xx, yy)[3]))
  return np.asarray(f(np.asarray(x, float), np.asarray(y, float)))


def main():
  data = load_data()
  os.makedirs(FIG_DIR, exist_ok=True)
  plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 10,
                       "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                       "ytick.color": INK_2, "xtick.direction": "in", "ytick.direction": "in"})
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, constrained_layout=True)
  errors = []
  xi_line = np.linspace(0.0, 1.0, 401)

  for ax, (exp_id, (title, cfg_mod)) in zip(axes, EXPERIMENTS.items()):
    phys = cfg_mod.get_config().physics
    ax.axvspan(phys.porous_x_start, phys.porous_x_end, color="#f1f0ec", zorder=0)
    ax.text(0.5 * (phys.porous_x_start + phys.porous_x_end), 0.02, "porous layer",
            ha="center", va="bottom", color=INK_2, fontsize=8)
    for label, exp_name, ls in RUNS:
      try:
        model, _, _ = experiment.load_run("fluid_porous_layer", exp_name, cfg_mod)
      except FileNotFoundError:
        print(f"[skip] experiment {exp_id}: run '{exp_name}' not found")
        continue
      for h, (marker, color) in HEIGHTS.items():
        ax.plot(xi_line, temperature(model, xi_line, np.full_like(xi_line, h)),
                ls=ls, lw=1.6, color=color, zorder=2)
        pts = data[exp_id][h]
        diff = temperature(model, pts[:, 0], np.full(len(pts), h)) - pts[:, 1]
        errors.append(dict(experiment=exp_id, model=label, y_over_L=h,
                           rms=float(np.sqrt(np.mean(diff**2))), max_abs=float(np.max(np.abs(diff)))))
    for h, (marker, color) in HEIGHTS.items():
      pts = data[exp_id][h]
      ax.plot(pts[:, 0], pts[:, 1], ls="none", marker=marker, ms=7, mfc="white", mec=color,
              mew=1.6, zorder=3)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\xi = x/L$")
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
  axes[0].set_ylabel(r"$\theta = (T - T_c)/(T_h - T_c)$")

  # legend: heights (marker + colour) and models (line style); never colour alone
  from matplotlib.lines import Line2D
  handles = [Line2D([], [], ls="none", marker=m, ms=7, mfc="white", mec=c, mew=1.6, label=f"exp. $y/L$ = {h}")
             for h, (m, c) in HEIGHTS.items()]
  handles += [Line2D([], [], ls=ls, color=INK_2, lw=1.6, label=label) for label, _, ls in RUNS]
  fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=8.5)
  fig.suptitle("Beckermann et al. (1988), configuration 2, glycerin: measured vs. PINN-FV temperature profiles",
               color=INK, fontsize=11)
  for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"beckermann1988_fig3_comparison.{ext}"), dpi=300)

  with open(os.path.join(FIG_DIR, "beckermann1988_fig3_errors.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["experiment", "model", "y_over_L", "rms", "max_abs"])
    w.writeheader(); w.writerows(errors)
  print(f"{'exp':>3s}  {'model':38s} {'y/L':>6s} {'RMS dtheta':>10s} {'max |dtheta|':>12s}")
  for e in errors:
    print(f"{e['experiment']:>3d}  {e['model']:38s} {e['y_over_L']:6.3f} {e['rms']:10.4f} {e['max_abs']:12.4f}")
  for exp_id in EXPERIMENTS:
    for label, _, _ in RUNS:
      sel = [e for e in errors if e["experiment"] == exp_id and e["model"] == label]
      if sel:
        print(f"exp {exp_id} {label}: overall RMS = {np.sqrt(np.mean([e['rms']**2 for e in sel])):.4f}, "
              f"max = {max(e['max_abs'] for e in sel):.4f}")


if __name__ == "__main__":
  main()
