"""Run a grid of experiments sequentially through scripts/train.py.

Each run is named <prefix>_<key><value>_..., saved like any other run under
outputs/<case>/, and logged to outputs/<case>/logs/<run>.log. Finished runs
(history.pkl present) are skipped and interrupted ones are resumed, so the
sweep can be stopped and restarted at any time.

  python scripts/sweep.py \
      --config=cases/fluid_porous_layer/config/aluminum.py --prefix=hconv \
      --set diff.scheme=ffv_compact --set diff.use_continuous_mesh=True \
      --grid seed,training.sampler_seed=0,1,2 \
      --grid diff.n_x,diff.n_y=40,60,80,120,160

--grid KEYS=VALUES: comma-separated values; several comma-separated keys take
the same value (e.g. n_x and n_y). The first --grid varies slowest, so put the
seed first to get one complete h-series before the next seed.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_grid(spec):
  keys, values = spec.split("=", 1)
  return keys.split(","), values.split(",")


def case_from_config_path(path):
  """cases/<case>/config/<file>.py -> <case>.

  Deliberately does not import scripts.train or anything that imports JAX: this
  process lives for the whole sweep, and JAX would preallocate ~75% of the GPU
  memory here, leaving the training subprocesses only the remainder.
  """
  parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
  if "cases" not in parts or not os.path.isdir(os.path.join(REPO_ROOT, "cases", parts[parts.index("cases") + 1], "config")):
    raise ValueError(f"Config must live under cases/<case>/config/: {path}")
  return parts[parts.index("cases") + 1]


def short_name(keys):
  """'seed,training.sampler_seed' -> 'seed'; 'diff.n_x,diff.n_y' -> 'n'."""
  last = keys[0].split(".")[-1]
  return "n" if last in ("n_x", "n_y") else last


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--config", required=True)
  parser.add_argument("--prefix", required=True)
  parser.add_argument("--set", action="append", default=[], help="fixed override KEY=VALUE")
  parser.add_argument("--grid", action="append", default=[], help="swept override KEYS=V1,V2,...")
  parser.add_argument("--dry_run", action="store_true")
  args = parser.parse_args()

  sys.path.insert(0, REPO_ROOT)
  case = case_from_config_path(args.config)
  case_dir = os.path.join(REPO_ROOT, "outputs", case)
  log_dir = os.path.join(case_dir, "logs")
  os.makedirs(log_dir, exist_ok=True)

  # run-folder prefix = <config.wandb.name>_
  config_mod = os.path.relpath(os.path.abspath(args.config), REPO_ROOT)[:-3].replace(os.sep, ".")
  import importlib
  wandb_name = importlib.import_module(config_mod).get_config().wandb.name

  grids = [parse_grid(g) for g in args.grid]
  combos = list(itertools.product(*[[(keys, v) for v in values] for keys, values in grids]))
  print(f"{len(combos)} runs in outputs/{case}/ (logs in {os.path.relpath(log_dir, REPO_ROOT)})", flush=True)

  for i, combo in enumerate(combos, 1):
    exp_name = "_".join([args.prefix] + [f"{short_name(keys)}{v}" for keys, v in combo])
    run_dir = os.path.join(case_dir, f"{wandb_name}_{exp_name}")
    overrides = [f"--config.{kv}" for kv in args.set]
    overrides += [f"--config.{k}={v}" for keys, v in combo for k in keys]
    if os.path.exists(os.path.join(run_dir, "history.pkl")):
      print(f"[{i}/{len(combos)}] skip   {exp_name} (finished)", flush=True)
      continue
    if os.path.exists(run_dir):
      overrides.append("--config.training.resume=True")
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "train.py"),
           f"--config={args.config}", f"--exp_name={exp_name}"] + overrides
    if args.dry_run:
      print(f"[{i}/{len(combos)}] " + " ".join(cmd[1:]), flush=True)
      continue

    start = time.time()
    print(f"[{i}/{len(combos)}] {time.strftime('%H:%M:%S')} start  {exp_name}", flush=True)
    with open(os.path.join(log_dir, f"{exp_name}.log"), "a") as log:
      rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    status = "done" if rc == 0 else f"FAILED rc={rc}"
    print(f"[{i}/{len(combos)}] {time.strftime('%H:%M:%S')} {status:6s} {exp_name} "
          f"({(time.time() - start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
  main()
