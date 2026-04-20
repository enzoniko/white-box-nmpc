#!/usr/bin/env python3
"""
Minimal Synthetic Evaluation (In-Manifold vs Out-of-Manifold) — Paper-Ready

This experiment evaluates *trained surrogate* prediction quality relative to the
ORCA-scale physics oracle on time-consistent trajectories.

We report one-step prediction errors under:
  - In-manifold: mu_scale(t) stays within the trained library grid.
  - Out-of-manifold: mu_scale(t) steps outside that grid.

Notes:
  - This is a prediction-quality experiment (not full MPC tracking).
  - True dynamics use the BayesRace-consistent simplified Pacejka (no E).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from .orca_physics import OrcaParams, accelerations_numpy
from .orca_library import load_orca_library, pick_in_out_mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./paper_synth_results")
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--lib_dir", type=str, default="./orca_library_trained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu_step", type=int, default=150)
    parser.add_argument("--mu_ramp", type=int, default=0, help="0=step change, >0=ramp duration")
    parser.add_argument("--out_multiplier", type=float, default=1.3, help="mu_out = out_multiplier * mu_max")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    device = torch.device("cpu")

    lib = load_orca_library(args.lib_dir, device=device)
    mu_min, mu_max = float(np.min(lib.mu_scale_grid)), float(np.max(lib.mu_scale_grid))
    mu_in, mu_out = pick_in_out_mu(mu_min, mu_max, out_multiplier=float(args.out_multiplier))

    # Build a time-consistent control sequence and mu schedules
    T = int(args.T)
    dt = float(args.dt)

    # simple excitation controls
    delta = rng.uniform(-0.30, 0.30, size=(T,))
    pwm = rng.uniform(-0.40, 0.90, size=(T,))
    u = np.column_stack([delta, pwm]).astype(np.float64)

    mu_in_sched = np.ones((T,), dtype=np.float64) * mu_in
    mu_out_sched = np.ones((T,), dtype=np.float64) * mu_in
    if args.mu_ramp <= 0:
        mu_in_sched[args.mu_step:] = mu_in
        mu_out_sched[args.mu_step:] = mu_out
    else:
        ramp = int(args.mu_ramp)
        mu_out_sched[args.mu_step:] = mu_out
        # ramp in-manifold is optional; keep constant for clarity

    orca = OrcaParams()
    static = lib.static_params.astype(np.float64)

    def rollout_true(mu_sched: np.ndarray):
        x = np.zeros((T, 3), dtype=np.float64)
        dx = np.zeros((T, 3), dtype=np.float64)
        x[0] = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        for t in range(T - 1):
            dx[t] = accelerations_numpy(x[t], u[t], static, orca, mu_scale=float(mu_sched[t]))
            x[t + 1] = x[t] + dt * dx[t]
            x[t + 1, 0] = max(x[t + 1, 0], 0.05)
        dx[-1] = accelerations_numpy(x[-1], u[-1], static, orca, mu_scale=float(mu_sched[-1]))
        return x, dx

    x_in, dx_in = rollout_true(mu_in_sched)
    x_out, dx_out = rollout_true(mu_out_sched)

    def surrogate_one_step_errors(x: np.ndarray, mu_sched: np.ndarray):
        errs = []
        dx_errs = []
        vx_errs = []
        dvx_errs = []
        for t in range(T - 1):
            # Mode A: pass mu_current (mu_scale) to compute weights internally (RBF).
            st = torch.from_numpy(x[t].astype(np.float32)).view(1, 3)
            ct = torch.from_numpy(u[t].astype(np.float32)).view(1, 2)
            ss = torch.from_numpy(lib.static_params.astype(np.float32)).view(1, 2)
            with torch.no_grad():
                pred_dx = lib.ensemble(st, ct, ss, mu_current=float(mu_sched[t])).cpu().numpy().reshape(-1)
            x_pred = x[t] + dt * pred_dx
            errs.append(float(np.linalg.norm(x_pred - x[t + 1])))
            dx_errs.append(float(np.linalg.norm(pred_dx - (x[t + 1] - x[t]) / dt)))
            vx_errs.append(float(abs(x_pred[0] - x[t + 1, 0])))
            dvx_errs.append(float(abs(pred_dx[0] - (x[t + 1, 0] - x[t, 0]) / dt)))
        return (
            float(np.mean(errs)),
            float(np.percentile(errs, 95)),
            float(np.mean(dx_errs)),
            float(np.percentile(dx_errs, 95)),
            float(np.mean(vx_errs)),
            float(np.percentile(vx_errs, 95)),
            float(np.mean(dvx_errs)),
            float(np.percentile(dvx_errs, 95)),
        )

    # Compare surrogate prediction against the true rollout state transitions
    xrmse_in, xp95_in, dxrmse_in, dxp95_in, vxrmse_in, vxp95_in, dvxrmse_in, dvxp95_in = surrogate_one_step_errors(x_in, mu_in_sched)
    xrmse_out, xp95_out, dxrmse_out, dxp95_out, vxrmse_out, vxp95_out, dvxrmse_out, dvxp95_out = surrogate_one_step_errors(x_out, mu_out_sched)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(x_in[:, 0], label="vx (in-manifold)")
    plt.plot(x_out[:, 0], label="vx (out-of-manifold)", alpha=0.8)
    plt.axvline(int(args.mu_step), color="k", linestyle="--", linewidth=1, label="mu step")
    plt.title("Synthetic Trajectory (vx)")
    plt.xlabel("t")
    plt.ylabel("m/s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "synthetic_vx.png", dpi=150)
    plt.close(fig)

    results = {
        "config": {
            "T": T,
            "dt": dt,
            "seed": int(args.seed),
            "lib_dir": str(Path(args.lib_dir).resolve()),
            "mu_grid": lib.mu_scale_grid.tolist(),
            "mu_in": float(mu_in),
            "mu_out": float(mu_out),
            "mu_step": int(args.mu_step),
        },
        "one_step_state_error": {
            "rmse_in_manifold": xrmse_in,
            "p95_in_manifold": xp95_in,
            "rmse_out_of_manifold": xrmse_out,
            "p95_out_of_manifold": xp95_out,
        },
        "one_step_vx_error": {
            "rmse_in_manifold": vxrmse_in,
            "p95_in_manifold": vxp95_in,
            "rmse_out_of_manifold": vxrmse_out,
            "p95_out_of_manifold": vxp95_out,
        },
        "one_step_accel_error": {
            "rmse_in_manifold": dxrmse_in,
            "p95_in_manifold": dxp95_in,
            "rmse_out_of_manifold": dxrmse_out,
            "p95_out_of_manifold": dxp95_out,
        },
        "one_step_dvx_error": {
            "rmse_in_manifold": dvxrmse_in,
            "p95_in_manifold": dvxp95_in,
            "rmse_out_of_manifold": dvxrmse_out,
            "p95_out_of_manifold": dvxp95_out,
        },
    }
    with open(out_dir / "synthetic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


