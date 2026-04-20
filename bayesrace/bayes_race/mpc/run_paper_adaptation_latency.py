#!/usr/bin/env python3
"""
Paper Experiment B: Adaptation Latency (BayesRace/DeepDynamics-style)

We emulate the common pattern used in deep_dynamics scripts:
  - estimate/update dynamics parameters online
  - rebuild Dynamic model + rebuild NMPC (expensive)

And compare to our surrogate Mode B:
  - keep NMPC fixed
  - update only specialist weights (fast) via OnlineLinearRegressor

This script is intentionally minimal:
  - uses ORCA params + ETHZ track
  - runs a short loop
  - measures:
      * rebuild time spike at change step (physics baseline)
      * per-step solve time
      * Mode B update time per step (surrogate)
"""

import time
import numpy as np
import json
import torch

from bayes_race.params import ORCA
from bayes_race.models import Dynamic
from bayes_race.tracks import ETHZ
from bayes_race.mpc.planner import ConstantSpeed
from bayes_race.mpc.nmpc import setupNLP
from bayes_race.mpc.nmpc_adaptive import setupNLPAdaptive

# Ensure repo root is on PYTHONPATH so `s2gpt_pinn` can be imported when running from bayesrace/
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../NMPC
sys.path.insert(0, str(_REPO_ROOT))

from bayes_race.models.neural_surrogate import NeuralSurrogateDynamic, SurrogateParams

from s2gpt_pinn.orca_library import load_orca_library
from s2gpt_pinn.calibration import OnlineLinearRegressor


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=str, default=None, help="Optional path to write JSON results")
    parser.add_argument("--n_steps", type=int, default=120)
    parser.add_argument("--change_step", type=int, default=60)
    parser.add_argument("--lib_dir", type=str, default=str((_REPO_ROOT / "s2gpt_pinn" / "orca_library_trained").resolve()))
    args = parser.parse_args()

    Ts = 0.02
    horizon = 13
    COST_Q = np.diag([1, 1])
    COST_P = np.diag([0, 0])
    COST_R = np.diag([5 / 1000, 1])

    params = ORCA(control="pwm")
    track = ETHZ(reference="optimal", longer=True)

    # Base physics model
    model_phys = Dynamic(**params)
    nlp_phys = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model_phys, track, track_cons=False)

    # Surrogate (weights_param) adaptive NMPC (NO rebuild): load trained ORCA library.
    lib = load_orca_library(args.lib_dir, device=torch.device("cpu"))
    ensemble = lib.ensemble
    sparams = SurrogateParams(static_mass=params["mass"], static_Iz=params["Iz"], mu_fixed=1.0, top_k=4, n_specialists=ensemble.n_specialists)
    model_sur = NeuralSurrogateDynamic(ensemble=ensemble, params=sparams, backend="native", mode="weights_param")
    nlp_sur = setupNLPAdaptive(horizon, Ts, COST_Q, COST_P, COST_R, params, model_sur, track, dyn_dim=ensemble.n_specialists, track_cons=False)

    reg = OnlineLinearRegressor(ensemble, window_size=40, regularization=1e-4, device="cpu")
    w = np.ones(ensemble.n_specialists) / ensemble.n_specialists

    # scenario: parameter jump at step
    change_step = int(args.change_step)
    Df_before, Df_after = params["Df"], params["Df"] * 0.6
    Dr_before, Dr_after = params["Dr"], params["Dr"] * 0.6

    # init
    n_states = model_phys.n_states
    projidx = 0
    x0 = np.zeros(n_states)
    x0[0], x0[1] = track.x_init, track.y_init
    x0[2] = track.psi_init
    x0[3] = track.vx_init
    uprev = np.zeros((2, 1))

    # timings
    rebuild_times = []
    solve_phys = []
    solve_sur = []
    update_ms = []

    for t in range(int(args.n_steps)):
        # change physics parameters and REBUILD at change step (baseline)
        if t == change_step:
            params2 = dict(params)
            params2["Df"] = Df_after
            params2["Dr"] = Dr_after
            t0 = time.perf_counter()
            model_phys = Dynamic(**params2)
            nlp_phys = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params2, model_phys, track, track_cons=False)
            rebuild_times.append((time.perf_counter() - t0) * 1000.0)
            params = params2

        xref, projidx = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)

        # Physics solve
        t0 = time.perf_counter()
        uopt, _, _ = nlp_phys.solve(x0=x0, xref=xref[:2, :], uprev=uprev)
        solve_phys.append((time.perf_counter() - t0) * 1000.0)
        u_apply = uopt[:, 0].reshape(2, 1)

        # Simulate "measured" accelerations from finite difference of true model integration
        x_traj, dxdt_traj = model_phys.sim_continuous(x0, u_apply, [0, Ts])
        x1 = x_traj[:, -1]
        dv = (x1[3:6] - x0[3:6]) / Ts  # [dvx,dvy,domega]

        # Mode B update (weights) using measured dv
        # Surrogate inputs: state_dyn=[vx,vy,omega], control=[steer,pwm]=[u[1],u[0]]
        state_dyn = x0[3:6].astype(np.float32)
        control_dyn = np.array([float(u_apply[1, 0]), float(u_apply[0, 0])], dtype=np.float32)
        static = np.array([params["mass"], params["Iz"]], dtype=np.float32)
        t0 = time.perf_counter()
        reg.add_observation(state_dyn, control_dyn, static, dv.astype(np.float32))
        update_ms.append((time.perf_counter() - t0) * 1000.0)
        w = reg.get_weights()

        # Surrogate solve (NO rebuild) with weights in p
        t0 = time.perf_counter()
        uopt2, _, _ = nlp_sur.solve(x0=x0, xref=xref[:2, :], uprev=uprev, dyn=w.reshape(-1, 1))
        solve_sur.append((time.perf_counter() - t0) * 1000.0)

        # advance plant
        x0 = x1
        uprev = u_apply

    print("=== Adaptation Latency Summary ===")
    if rebuild_times:
        print(f"Physics rebuild spike: {rebuild_times[0]:.2f} ms")
    print(f"Physics solve mean: {np.mean(solve_phys):.2f} ms (p95 {np.percentile(solve_phys,95):.2f})")
    print(f"Surrogate solve mean: {np.mean(solve_sur):.2f} ms (p95 {np.percentile(solve_sur,95):.2f})")
    print(f"Mode B update mean: {np.mean(update_ms):.3f} ms (p95 {np.percentile(update_ms,95):.3f})")

    results = {
        "physics_rebuild_ms": float(rebuild_times[0]) if rebuild_times else None,
        "physics_solve_ms": {"mean": float(np.mean(solve_phys)), "p95": float(np.percentile(solve_phys, 95)), "max": float(np.max(solve_phys))},
        "surrogate_solve_ms": {"mean": float(np.mean(solve_sur)), "p95": float(np.percentile(solve_sur, 95)), "max": float(np.max(solve_sur))},
        "mode_b_update_ms": {"mean": float(np.mean(update_ms)), "p95": float(np.percentile(update_ms, 95)), "max": float(np.max(update_ms))},
        "change_step": change_step,
        "n_steps": int(args.n_steps),
        "notes": {
            "physics_baseline": "rebuild Dynamic + setupNLP at change step (matches deep_dynamics scripts)",
            "surrogate": "native CasADi export with weights as NLP parameters (Mode B), no rebuild",
            "surrogate_weights": f"computed online from measured dv via OnlineLinearRegressor (trained library {args.lib_dir})",
        },
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()


