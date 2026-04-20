#!/usr/bin/env python3
"""
Paper Experiment A: Efficiency Boundary (BayesRace harness)

Goal:
  Compare solve-time overhead in BayesRace NMPC between:
    1) Symbolic physics (BayesRace Dynamic model, CasADi SX)
    2) Neural surrogate via CasADi-native exported graph
    3) Neural surrogate via Python callback (Top-K optional)

This is a MINIMAL benchmark:
  - we do NOT change parameters during the run
  - we only measure build time (nlpsol construction) and solve time distribution

Note on data:
  Uses BayesRace track planner `ConstantSpeed` to generate consistent xref.
  This is not "real sensor data", it's the BayesRace simulator/planner pipeline.
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

# Ensure repo root is on PYTHONPATH so `s2gpt_pinn` can be imported when running from bayesrace/
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../NMPC
sys.path.insert(0, str(_REPO_ROOT))

from bayes_race.models.neural_surrogate import NeuralSurrogateDynamic, SurrogateParams

from s2gpt_pinn.orca_library import load_orca_library


class setupNLPCallback(setupNLP):
    """
    Same as BayesRace setupNLP, but uses limited-memory Hessian approximation.
    This avoids requiring second derivatives through Python callbacks.
    """

    def __init__(self, horizon, Ts, Q, P, R, params, model, track, track_cons=False):
        # Copy/paste minimal modifications from bayes_race.mpc.nmpc.setupNLP.__init__
        self.horizon = horizon
        self.params = params
        self.model = model
        self.track = track
        self.track_cons = track_cons

        n_states = model.n_states
        n_inputs = model.n_inputs
        xref_size = 2

        import casadi as cs
        from bayes_race.mpc.constraints import Boundary

        x0 = cs.SX.sym("x0", n_states, 1)
        xref = cs.SX.sym("xref", xref_size, horizon + 1)
        uprev = cs.SX.sym("uprev", 2, 1)
        x = cs.SX.sym("x", n_states, horizon + 1)
        u = cs.SX.sym("u", n_inputs, horizon)
        dxdtc = cs.SX.sym("dxdt", n_states, 1)

        if track_cons:
            eps = cs.SX.sym("eps", 2, horizon)
            Aineq = cs.SX.sym("Aineq", 2 * horizon, 2)
            bineq = cs.SX.sym("bineq", 2 * horizon, 1)

        cost_tracking = 0
        cost_actuation = 0
        cost_violation = 0

        cost_tracking += (x[:xref_size, -1] - xref[:xref_size, -1]).T @ P @ (x[:xref_size, -1] - xref[:xref_size, -1])
        constraints = x[:, 0] - x0

        for idh in range(horizon):
            dxdt = model.casadi(x[:, idh], u[:, idh], dxdtc)
            constraints = cs.vertcat(constraints, x[:, idh + 1] - x[:, idh] - Ts * dxdt)

        for idh in range(horizon):
            if idh == 0:
                deltaU = u[:, idh] - uprev
            else:
                deltaU = u[:, idh] - u[:, idh - 1]

            cost_tracking += (x[:xref_size, idh + 1] - xref[:xref_size, idh + 1]).T @ Q @ (x[:xref_size, idh + 1] - xref[:xref_size, idh + 1])
            cost_actuation += deltaU.T @ R @ deltaU

            if track_cons:
                cost_violation += 1e6 * (eps[:, idh].T @ eps[:, idh])

            constraints = cs.vertcat(constraints, u[:, idh] - params["max_inputs"])
            constraints = cs.vertcat(constraints, -u[:, idh] + params["min_inputs"])
            constraints = cs.vertcat(constraints, deltaU[1] - params["max_rates"][1] * Ts)
            constraints = cs.vertcat(constraints, -deltaU[1] + params["min_rates"][1] * Ts)

            if track_cons:
                constraints = cs.vertcat(constraints, Aineq[2 * idh : 2 * idh + 2, :] @ x[:2, idh + 1] - bineq[2 * idh : 2 * idh + 2, :] - eps[:, idh])

        cost = cost_tracking + cost_actuation + cost_violation

        xvars = cs.vertcat(cs.reshape(x, -1, 1), cs.reshape(u, -1, 1))
        if track_cons:
            xvars = cs.vertcat(xvars, cs.reshape(eps, -1, 1))

        pvars = cs.vertcat(cs.reshape(x0, -1, 1), cs.reshape(xref, -1, 1), cs.reshape(uprev, -1, 1))
        if track_cons:
            pvars = cs.vertcat(pvars, cs.reshape(Aineq, -1, 1), cs.reshape(bineq, -1, 1))

        nlp = {"x": xvars, "p": pvars, "f": cost, "g": constraints}
        options = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "print_timing_statistics": "no",
                "max_iter": 100,
                "hessian_approximation": "limited-memory",
            },
        }
        self.problem = cs.nlpsol("nmpc_cb", "ipopt", nlp, options)


def time_build_and_solves(name: str, nlp, model, track, plant_model, n_steps: int = 50, horizon: int = 13, Ts: float = 0.02):
    n_states = model.n_states
    n_inputs = model.n_inputs

    # init
    projidx = 0
    x0 = np.zeros(n_states)
    x0[0], x0[1] = track.x_init, track.y_init
    x0[2] = track.psi_init
    x0[3] = track.vx_init
    uprev = np.zeros((2, 1))

    # warm + measure
    solve_times = []
    for k in range(n_steps):
        xref, projidx = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)
        t0 = time.perf_counter()
        uopt, fval, _ = nlp.solve(x0=x0, xref=xref[:2, :], uprev=uprev)
        dt = (time.perf_counter() - t0) * 1000.0
        solve_times.append(dt)
        # advance a tiny bit with model (for consistent x0 evolution)
        uprev = uopt[:, 0].reshape(2, 1)
        # Always advance the plant using the physics model to keep x0 evolution consistent
        x_traj, _ = plant_model.sim_continuous(x0, uprev, [0, Ts])
        x0 = x_traj[:, -1]

    solve_times = np.array(solve_times)
    print(f"[{name}] solve mean={solve_times.mean():.2f}ms p95={np.percentile(solve_times,95):.2f}ms max={solve_times.max():.2f}ms")
    return {
        "mean_ms": float(solve_times.mean()),
        "p95_ms": float(np.percentile(solve_times, 95)),
        "max_ms": float(solve_times.max()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=str, default=None, help="Optional path to write JSON results")
    parser.add_argument("--n_steps", type=int, default=60)
    parser.add_argument("--lib_dir", type=str, default=str((_REPO_ROOT / "s2gpt_pinn" / "orca_library_trained").resolve()))
    args = parser.parse_args()

    Ts = 0.02
    horizon = 13
    COST_Q = np.diag([1, 1])
    COST_P = np.diag([0, 0])
    COST_R = np.diag([5 / 1000, 1])

    params = ORCA(control="pwm")
    track = ETHZ(reference="optimal", longer=True)

    # Baseline physics model + NLP
    model_phys = Dynamic(**params)
    t0 = time.perf_counter()
    nlp_phys = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model_phys, track, track_cons=False)
    build_phys = (time.perf_counter() - t0) * 1000.0
    print(f"[physics] build {build_phys:.1f}ms")

    # Load trained ORCA library (weights don't affect solve-time materially, but avoids "toy" caveat).
    lib = load_orca_library(args.lib_dir, device=torch.device("cpu"))
    ensemble = lib.ensemble
    sparams = SurrogateParams(static_mass=params["mass"], static_Iz=params["Iz"], mu_fixed=1.0, top_k=4, n_specialists=ensemble.n_specialists)

    # Native graph surrogate
    model_nat = NeuralSurrogateDynamic(ensemble=ensemble, params=sparams, backend="native", mode="fixed_mu")
    t0 = time.perf_counter()
    nlp_nat = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model_nat, track, track_cons=False)
    build_nat = (time.perf_counter() - t0) * 1000.0
    print(f"[surrogate/native] build {build_nat:.1f}ms")

    # Callback surrogate (Top-K)
    model_cb = NeuralSurrogateDynamic(ensemble=ensemble, params=sparams, backend="callback", mode="fixed_mu")
    t0 = time.perf_counter()
    nlp_cb = setupNLPCallback(horizon, Ts, COST_Q, COST_P, COST_R, params, model_cb, track, track_cons=False)
    build_cb = (time.perf_counter() - t0) * 1000.0
    print(f"[surrogate/callback topK={sparams.top_k}] build {build_cb:.1f}ms")

    # Solve timing
    results = {
        "build_ms": {
            "physics": float(build_phys),
            "surrogate_native": float(build_nat),
            "surrogate_callback": float(build_cb),
        },
        "solve_ms": {
            "physics": time_build_and_solves("physics", nlp_phys, model_phys, track, plant_model=model_phys, n_steps=args.n_steps, horizon=horizon, Ts=Ts),
            "surrogate_native": time_build_and_solves("surrogate_native", nlp_nat, model_nat, track, plant_model=model_phys, n_steps=args.n_steps, horizon=horizon, Ts=Ts),
            "surrogate_callback": time_build_and_solves("surrogate_callback", nlp_cb, model_cb, track, plant_model=model_phys, n_steps=args.n_steps, horizon=horizon, Ts=Ts),
        },
        "notes": {
            "callback_solver": "limited-memory Hessian (no 2nd derivatives through callback)",
            "surrogate": f"trained library loaded from {args.lib_dir} (timing benchmark; accuracy not evaluated here)",
        },
    }

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()


