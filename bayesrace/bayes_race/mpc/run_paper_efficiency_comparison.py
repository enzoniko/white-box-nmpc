#!/usr/bin/env python3
"""
Paper Experiment A-Prime: Identical OCP Efficiency Comparison (BayesRace harness)

Motivation
----------
We observed a large solve-time gap between:
  - a minimal 3-state toy NMPC microbenchmark (~23 ms), and
  - the full BayesRace NMPC benchmark (~677 ms) for the neural surrogate.

This script resolves the discrepancy by running *identical* NMPC formulations with
only the dynamics model swapped:
  1) Symbolic physics (BayesRace Dynamic)
  2) Neural surrogate (CasADi-native export; trained library)

Deliverables
------------
- JSON artifact with:
    * build time, solve time distribution
    * pure dynamics eval time (forward + Jacobian)
    * constraint Jacobian / Hessian evaluation time
    * sparsity % of constraint Jacobian and Hessian
    * IPOPT timing stats (parsed from IPOPT output file), including linear solver time
    * full metadata (horizon, Ts, cost weights, ipopt options, git-like provenance)
- PNG sparsity heatmaps for constraint Jacobian (physics vs surrogate)

Run (example)
-------------
cd /home/enzo/HYDRA/NMPC/bayesrace
MPLBACKEND=Agg python -m bayes_race.mpc.run_paper_efficiency_comparison \
  --n_steps 40 \
  --horizon 10 \
  --lib_dir /home/enzo/HYDRA/NMPC/s2gpt_pinn/orca_library_trained \
  --out_dir /home/enzo/HYDRA/NMPC/s2gpt_pinn/paper_results_bayesrace_efficiency_comparison \
  --out_json /home/enzo/HYDRA/NMPC/s2gpt_pinn/paper_results_bayesrace_efficiency_comparison.json
"""

from __future__ import annotations

import argparse
import json
import contextlib
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import casadi as cs
import torch

from bayes_race.params import ORCA
from bayes_race.models import Dynamic
from bayes_race.tracks import ETHZ
from bayes_race.mpc.planner import ConstantSpeed

# Ensure repo root is on PYTHONPATH so `s2gpt_pinn` can be imported when running from bayesrace/
import sys
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../NMPC
sys.path.insert(0, str(_REPO_ROOT))

from bayes_race.models.neural_surrogate import NeuralSurrogateDynamic, SurrogateParams
from s2gpt_pinn.orca_library import load_orca_library


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _redirect_c_stdout_stderr(to_path: Path):
    """
    Redirect OS-level stdout/stderr (FD 1/2) so IPOPT/CasADi C++ prints are captured.
    Python-level redirect_stdout is insufficient for C libraries.
    """
    to_path = Path(to_path)
    _mkdir(to_path.parent)
    f = open(to_path, "w")
    try:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        yield
    finally:
        try:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
        finally:
            os.close(old_stdout)
            os.close(old_stderr)
            f.close()


def _plot_sparsity(sp: cs.Sparsity, out_png: Path, title: str) -> Dict[str, float]:
    """
    Save a sparsity "heatmap" (spy plot). CasADi sparsity uses (row, col) triplets.
    """
    nrow = int(sp.size1())
    ncol = int(sp.size2())
    nnz = int(sp.nnz())

    # Convert to coordinate list for plotting
    rows, cols = sp.get_triplet()[0], sp.get_triplet()[1]
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)

    plt.figure(figsize=(6, 5))
    plt.scatter(cols, rows, s=1.0, marker="s")
    plt.gca().invert_yaxis()
    plt.xlabel("col")
    plt.ylabel("row")
    plt.title(title + f"\n{nnz}/{nrow*ncol} nnz ({100.0*nnz/(nrow*ncol):.3f}%)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return {
        "rows": float(nrow),
        "cols": float(ncol),
        "nnz": float(nnz),
        "density": float(nnz / max(nrow * ncol, 1)),
        "density_percent": float(100.0 * nnz / max(nrow * ncol, 1)),
    }


def _parse_ipopt_timing_stats(text: str) -> Dict[str, Any]:
    """
    Parse IPOPT timing statistics from the IPOPT output file.
    We keep this tolerant across IPOPT versions by using regex patterns.
    """
    out: Dict[str, Any] = {"raw_found": False}
    if not text:
        return out

    # A few common summary lines/keys (vary across IPOPT versions)
    patterns = {
        # newer wording
        "total_ipopt_cpu_no_f_eval_s": r"Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([0-9eE\.\+\-]+)",
        "total_linear_solver_cpu_s": r"Total CPU secs in linear solver\s*=\s*([0-9eE\.\+\-]+)",
        "total_nlp_f_eval_cpu_s": r"Total CPU secs in NLP function evaluations\s*=\s*([0-9eE\.\+\-]+)",
        "total_nlp_f_eval_wall_s": r"Total wall-clock secs in NLP function evaluations\s*=\s*([0-9eE\.\+\-]+)",
        "total_ipopt_wall_s": r"Total wall-clock secs in IPOPT\s*=\s*([0-9eE\.\+\-]+)",
        # older wording (often emitted via CasADi builds)
        "total_ipopt_no_f_eval_s": r"Total seconds in IPOPT \(w/o function evaluations\)\s*=\s*([0-9eE\.\+\-]+)",
        "total_nlp_f_eval_s": r"Total seconds in NLP function evaluations\s*=\s*([0-9eE\.\+\-]+)",
    }

    found_any = False
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[k] = float(m.group(1))
            found_any = True

    # Parse "Timing Statistics" block into a dict. This is the most reliable
    # way to recover linear solver and factorization time.
    # Example line:
    #   PDSystemSolverTotal.................:      0.011 (sys:      0.001 wall:      0.011)
    timing_lines = re.findall(
        r"^([A-Za-z0-9_]+)\.+:\s*([0-9eE\.\+\-]+)\s*\(sys:\s*([0-9eE\.\+\-]+)\s*wall:\s*([0-9eE\.\+\-]+)\)",
        text,
        flags=re.MULTILINE,
    )
    if timing_lines:
        out["timing_statistics"] = {}
        for name, cpu_s, sys_s, wall_s in timing_lines:
            out["timing_statistics"][name] = {
                "cpu_s": float(cpu_s),
                "sys_s": float(sys_s),
                "wall_s": float(wall_s),
            }
        found_any = True

    # Parse IPOPT-reported nonzeros if present
    m = re.search(r"Number of nonzeros in equality constraint Jacobian\.*:\s*([0-9]+)", text)
    if m:
        out["ipopt_reported_nnz_eq_jac"] = int(m.group(1))
        found_any = True
    m = re.search(r"Number of nonzeros in inequality constraint Jacobian\.*:\s*([0-9]+)", text)
    if m:
        out["ipopt_reported_nnz_ineq_jac"] = int(m.group(1))
        found_any = True
    m = re.search(r"Number of nonzeros in Lagrangian Hessian\.*:\s*([0-9]+)", text)
    if m:
        out["ipopt_reported_nnz_hess"] = int(m.group(1))
        found_any = True

    out["raw_found"] = found_any
    return out


def _build_identical_nlp(
    *,
    horizon: int,
    Ts: float,
    Q: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    params: dict,
    model,
    track,
    track_cons: bool,
    ipopt_print_level: int,
    ipopt_max_iter: int,
    ipopt_print_timing_statistics: bool,
    ipopt_output_file: Optional[str],
) -> Tuple[cs.Function, Dict[str, Any], Dict[str, Any]]:
    """
    Clone of BayesRace `setupNLP.__init__` (with minimal extensions):
    - Return the constructed nlpsol
    - Return the symbolic `nlp` dict and basic structural metadata
    - Add IPOPT timing stats / output file options

    This ensures the OCP formulation is identical across dynamics models.
    """
    n_states = model.n_states
    n_inputs = model.n_inputs
    xref_size = 2

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

    for k in range(horizon):
        dxdt = model.casadi(x[:, k], u[:, k], dxdtc)
        constraints = cs.vertcat(constraints, x[:, k + 1] - x[:, k] - Ts * dxdt)

    for k in range(horizon):
        if k == 0:
            deltaU = u[:, k] - uprev
        else:
            deltaU = u[:, k] - u[:, k - 1]

        cost_tracking += (x[:xref_size, k + 1] - xref[:xref_size, k + 1]).T @ Q @ (x[:xref_size, k + 1] - xref[:xref_size, k + 1])
        cost_actuation += deltaU.T @ R @ deltaU

        if track_cons:
            cost_violation += 1e6 * (eps[:, k].T @ eps[:, k])

        constraints = cs.vertcat(constraints, u[:, k] - params["max_inputs"])
        constraints = cs.vertcat(constraints, -u[:, k] + params["min_inputs"])
        constraints = cs.vertcat(constraints, deltaU[1] - params["max_rates"][1] * Ts)
        constraints = cs.vertcat(constraints, -deltaU[1] + params["min_rates"][1] * Ts)

        if track_cons:
            constraints = cs.vertcat(
                constraints,
                Aineq[2 * k : 2 * k + 2, :] @ x[:2, k + 1] - bineq[2 * k : 2 * k + 2, :] - eps[:, k],
            )

    cost = cost_tracking + cost_actuation + cost_violation

    xvars = cs.vertcat(cs.reshape(x, -1, 1), cs.reshape(u, -1, 1))
    if track_cons:
        xvars = cs.vertcat(xvars, cs.reshape(eps, -1, 1))

    pvars = cs.vertcat(cs.reshape(x0, -1, 1), cs.reshape(xref, -1, 1), cs.reshape(uprev, -1, 1))
    if track_cons:
        pvars = cs.vertcat(pvars, cs.reshape(Aineq, -1, 1), cs.reshape(bineq, -1, 1))

    nlp = {"x": xvars, "p": pvars, "f": cost, "g": constraints}

    ipoptoptions: Dict[str, Any] = {
        "print_level": int(ipopt_print_level),
        "print_timing_statistics": "yes" if ipopt_print_timing_statistics else "no",
        "max_iter": int(ipopt_max_iter),
    }
    if ipopt_output_file:
        # IPOPT writes to this path (relative/absolute accepted).
        ipoptoptions["output_file"] = str(ipopt_output_file)
        # IPOPT only writes to file if file_print_level > 0
        ipoptoptions["file_print_level"] = int(ipopt_print_level)

    options = {
        "expand": True,
        "print_time": False,
        "ipopt": ipoptoptions,
    }

    solver = cs.nlpsol("nmpc_identical", "ipopt", nlp, options)

    meta = {
        "horizon": int(horizon),
        "Ts": float(Ts),
        "n_states": int(n_states),
        "n_inputs": int(n_inputs),
        "track_cons": bool(track_cons),
        "n_decision": int(xvars.size1()),
        "n_constraints": int(constraints.size1()),
        "ipopt": ipoptoptions,
        "casadi": {"expand": True},
    }
    return solver, nlp, meta


def _bench_dynamics_eval_and_jacobian(model, n_eval: int, seed: int = 0) -> Dict[str, Any]:
    """
    Benchmark pure dynamics evaluation and its Jacobian (wrt [x;u]).
    Uses a CasADi Function built from model.casadi().
    """
    rng = np.random.default_rng(int(seed))
    n_states = int(model.n_states)
    n_inputs = int(model.n_inputs)

    x = cs.SX.sym("x", n_states, 1)
    u = cs.SX.sym("u", n_inputs, 1)
    dxdtc = cs.SX.sym("dxdt", n_states, 1)
    dx = model.casadi(x, u, dxdtc)
    f = cs.Function("f_dyn", [x, u], [dx])
    z = cs.vertcat(x, u)
    J = cs.jacobian(dx, z)
    jf = cs.Function("jac_dyn", [x, u], [J])

    # Warmup
    for _ in range(10):
        xv = rng.standard_normal((n_states, 1))
        uv = rng.standard_normal((n_inputs, 1))
        _ = f(xv, uv)
        _ = jf(xv, uv)

    # Timed
    t0 = _now_ms()
    for _ in range(int(n_eval)):
        xv = rng.standard_normal((n_states, 1))
        uv = rng.standard_normal((n_inputs, 1))
        _ = f(xv, uv)
    f_ms = (_now_ms() - t0) / max(int(n_eval), 1)

    t0 = _now_ms()
    for _ in range(int(n_eval)):
        xv = rng.standard_normal((n_states, 1))
        uv = rng.standard_normal((n_inputs, 1))
        _ = jf(xv, uv)
    j_ms = (_now_ms() - t0) / max(int(n_eval), 1)

    sp_j = J.sparsity()
    jac_meta = {
        "rows": int(sp_j.size1()),
        "cols": int(sp_j.size2()),
        "nnz": int(sp_j.nnz()),
        "density": float(sp_j.nnz() / max(sp_j.size1() * sp_j.size2(), 1)),
    }

    return {
        "eval_ms_per_call": float(f_ms),
        "jac_ms_per_call": float(j_ms),
        "jac_sparsity": jac_meta,
    }


def _bench_symbolic_forms(nlp: Dict[str, Any], n_eval: int, seed: int = 0) -> Dict[str, Any]:
    """
    Benchmark:
      - constraint evaluation time g(x,p)
      - constraint Jacobian eval time Jg(x,p)
      - Hessian of Lagrangian eval time H(x,p,lam,obj_factor)

    This approximates the overhead sources seen inside IPOPT.
    """
    rng = np.random.default_rng(int(seed))
    x = nlp["x"]
    p = nlp["p"]
    g = nlp["g"]
    f = nlp["f"]

    g_fun = cs.Function("g_fun", [x, p], [g])
    Jg = cs.jacobian(g, x)
    Jg_fun = cs.Function("Jg_fun", [x, p], [Jg])

    lam = cs.SX.sym("lam", int(g.size1()), 1)
    obj_factor = cs.SX.sym("obj_factor", 1, 1)
    L = obj_factor * f + lam.T @ g
    H, _ = cs.hessian(L, x)  # dense SX; CasADi uses sparsity internally.
    H_fun = cs.Function("H_fun", [x, p, lam, obj_factor], [H])

    nx = int(x.size1())
    np_ = int(p.size1())
    ng = int(g.size1())

    def rand_xp():
        xv = rng.standard_normal((nx, 1))
        pv = rng.standard_normal((np_, 1))
        return xv, pv

    # warmup
    for _ in range(5):
        xv, pv = rand_xp()
        _ = g_fun(xv, pv)
        _ = Jg_fun(xv, pv)
        _ = H_fun(xv, pv, rng.standard_normal((ng, 1)), cs.DM([1.0]))

    # time g
    t0 = _now_ms()
    for _ in range(int(n_eval)):
        xv, pv = rand_xp()
        _ = g_fun(xv, pv)
    g_ms = (_now_ms() - t0) / max(int(n_eval), 1)

    # time Jg
    t0 = _now_ms()
    for _ in range(int(n_eval)):
        xv, pv = rand_xp()
        _ = Jg_fun(xv, pv)
    jg_ms = (_now_ms() - t0) / max(int(n_eval), 1)

    # time H
    t0 = _now_ms()
    for _ in range(int(n_eval)):
        xv, pv = rand_xp()
        lamv = rng.standard_normal((ng, 1))
        _ = H_fun(xv, pv, lamv, cs.DM([1.0]))
    h_ms = (_now_ms() - t0) / max(int(n_eval), 1)

    sp_jg = Jg.sparsity()
    sp_h = H.sparsity()
    return {
        "g_ms_per_call": float(g_ms),
        "jac_g_x_ms_per_call": float(jg_ms),
        "hess_lag_ms_per_call": float(h_ms),
        "jac_g_x_sparsity": {
            "rows": int(sp_jg.size1()),
            "cols": int(sp_jg.size2()),
            "nnz": int(sp_jg.nnz()),
            "density": float(sp_jg.nnz() / max(sp_jg.size1() * sp_jg.size2(), 1)),
        },
        "hess_lag_sparsity": {
            "rows": int(sp_h.size1()),
            "cols": int(sp_h.size2()),
            "nnz": int(sp_h.nnz()),
            "density": float(sp_h.nnz() / max(sp_h.size1() * sp_h.size2(), 1)),
        },
    }


def _time_solves(
    *,
    solver: cs.Function,
    horizon: int,
    Ts: float,
    track,
    model_phys_for_plant,
    n_steps: int,
    seed: int,
    ipopt_output_file: Optional[Path],
) -> Dict[str, Any]:
    """
    Run a short BayesRace loop to measure solve time distribution.
    Keep the plant evolution consistent by always advancing using the physics plant model.
    """
    rng = np.random.default_rng(int(seed))
    n_states = int(model_phys_for_plant.n_states)

    projidx = 0
    x0 = np.zeros(n_states)
    x0[0], x0[1] = track.x_init, track.y_init
    x0[2] = track.psi_init
    x0[3] = track.vx_init
    uprev = np.zeros((2, 1))

    solve_ms: List[float] = []

    def _run_loop():
        for _k in range(int(n_steps)):
            nonlocal projidx, x0, uprev
            xref, projidx = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)
            t0 = time.perf_counter()

            # BayesRace packs p as [x0; xref(:); uprev; (Aineq;bineq)] but we use track_cons=False.
            # The solver we built expects pvars with xref in column-major time ordering (same as BayesRace solve()).
            p = np.concatenate([x0.reshape(-1, 1), xref.T.reshape(-1, 1), uprev.reshape(-1, 1)], axis=0)

            # Bounds: mirror BayesRace solve() convention
            idx_g = solver.index_out("g")
            n_g = int(solver.size1_out(idx_g))
            n_eq = n_states * (horizon + 1)
            n_ineq = n_g - n_eq
            arg = {
                "p": p,
                "x0": np.zeros((int(solver.size1_in(solver.index_in("x0"))), 1)),
                "lbx": -np.inf * np.ones((n_states * (horizon + 1) + 2 * horizon, 1)),
                "ubx": np.inf * np.ones((n_states * (horizon + 1) + 2 * horizon, 1)),
                "lbg": np.concatenate([np.zeros((n_eq, 1)), -np.inf * np.ones((n_ineq, 1))], axis=0),
                "ubg": np.concatenate([np.zeros((n_eq, 1)), np.zeros((n_ineq, 1))], axis=0),
            }

            sol = solver(**arg)
            solve_ms.append((time.perf_counter() - t0) * 1000.0)

            # Extract first optimal control and advance plant with physics for consistent evolution.
            wopt = np.array(sol["x"]).reshape(-1, 1)
            offset_u = n_states * (horizon + 1)
            uopt = wopt[offset_u : offset_u + 2 * horizon].reshape(horizon, 2).T  # (2, horizon)
            uprev = uopt[:, 0].reshape(2, 1)

            x_traj, _ = model_phys_for_plant.sim_continuous(x0, uprev, [0, Ts])
            x0 = x_traj[:, -1]

    if ipopt_output_file is not None:
        with _redirect_c_stdout_stderr(Path(ipopt_output_file)):
            _run_loop()
    else:
        _run_loop()

    solve_ms_np = np.asarray(solve_ms, dtype=np.float64)

    # parse IPOPT timing stats from file (if present)
    ipopt_stats: Dict[str, Any] = {}
    if ipopt_output_file is not None and Path(ipopt_output_file).exists():
        try:
            text = Path(ipopt_output_file).read_text(errors="ignore")
        except Exception:
            text = ""
        ipopt_stats = _parse_ipopt_timing_stats(text)

    return {
        "mean_ms": float(np.mean(solve_ms_np)),
        "p95_ms": float(np.percentile(solve_ms_np, 95)),
        "max_ms": float(np.max(solve_ms_np)),
        "min_ms": float(np.min(solve_ms_np)),
        "std_ms": float(np.std(solve_ms_np)),
        "n_steps": int(n_steps),
        "ipopt_timing_stats": ipopt_stats,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default=str((_REPO_ROOT / "s2gpt_pinn" / "paper_results_bayesrace_efficiency_comparison").resolve()))
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--lib_dir", type=str, default=str((_REPO_ROOT / "s2gpt_pinn" / "orca_library_trained").resolve()))
    p.add_argument("--n_steps", type=int, default=40)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--Ts", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mu_fixed", type=float, default=1.0)
    p.add_argument("--ipopt_max_iter", type=int, default=100)
    p.add_argument("--ipopt_print_level", type=int, default=5)
    p.add_argument("--no_ipopt_timing_stats", action="store_true")
    p.add_argument(
        "--timing_stats_solves",
        type=int,
        default=1,
        help="Number of solves to run with verbose IPOPT output to capture timing stats (kept small to avoid huge logs).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    _mkdir(out_dir)

    horizon = int(args.horizon)
    Ts = float(args.Ts)
    seed = int(args.seed)

    COST_Q = np.diag([1.0, 1.0])
    COST_P = np.diag([0.0, 0.0])
    COST_R = np.diag([5.0 / 1000.0, 1.0])

    params = ORCA(control="pwm")
    track = ETHZ(reference="optimal", longer=True)

    # Plant model (physics) used only for stable state evolution during solve timing.
    plant_model = Dynamic(**params)

    # Models to compare (same OCP, swapped dynamics)
    model_phys = Dynamic(**params)
    lib = load_orca_library(args.lib_dir, device=torch.device("cpu"))
    ensemble = lib.ensemble
    sparams = SurrogateParams(
        static_mass=params["mass"],
        static_Iz=params["Iz"],
        mu_fixed=float(args.mu_fixed),
        top_k=None,  # native backend cannot truly skip compute; keep None for clean comparison
        n_specialists=ensemble.n_specialists,
    )
    model_sur = NeuralSurrogateDynamic(ensemble=ensemble, params=sparams, backend="native", mode="fixed_mu")

    comparisons = [
        ("physics", model_phys),
        ("surrogate_native", model_sur),
    ]

    results: Dict[str, Any] = {
        "config": {
            "out_dir": str(out_dir),
            "lib_dir": str(Path(args.lib_dir).resolve()),
            "seed": seed,
            "horizon": horizon,
            "Ts": Ts,
            "mu_fixed": float(args.mu_fixed),
            "cost_Q": COST_Q.tolist(),
            "cost_P": COST_P.tolist(),
            "cost_R": COST_R.tolist(),
            "params_name": "ORCA(control=pwm)",
            "track": "ETHZ(reference=optimal,longer=True)",
        },
        "models": {},
        "notes": {
            "purpose": "identical BayesRace OCP, swapped dynamics only",
            "track_cons": False,
            "ipopt_timing_stats": "enabled" if not args.no_ipopt_timing_stats else "disabled",
        },
    }

    for tag, model in comparisons:
        ipopt_log = out_dir / f"ipopt_{tag}.log"

        # Build a QUIET solver for timing distribution (no massive IPOPT logs).
        t0 = time.perf_counter()
        solver_quiet, nlp, meta = _build_identical_nlp(
            horizon=horizon,
            Ts=Ts,
            Q=COST_Q,
            P=COST_P,
            R=COST_R,
            params=params,
            model=model,
            track=track,
            track_cons=False,
            ipopt_print_level=0,
            ipopt_max_iter=int(args.ipopt_max_iter),
            ipopt_print_timing_statistics=False,
            ipopt_output_file=None,
        )
        build_ms = (time.perf_counter() - t0) * 1000.0

        # Build a VERBOSE solver only to capture IPOPT timing statistics / linear solver timing.
        t0 = time.perf_counter()
        solver_verbose, _nlp2, _meta2 = _build_identical_nlp(
            horizon=horizon,
            Ts=Ts,
            Q=COST_Q,
            P=COST_P,
            R=COST_R,
            params=params,
            model=model,
            track=track,
            track_cons=False,
            ipopt_print_level=int(args.ipopt_print_level),
            ipopt_max_iter=int(args.ipopt_max_iter),
            ipopt_print_timing_statistics=not bool(args.no_ipopt_timing_stats),
            ipopt_output_file=str(ipopt_log),
        )
        build_verbose_ms = (time.perf_counter() - t0) * 1000.0

        # Sparsity heatmap for constraint Jacobian (wrt decision vars)
        Jg = cs.jacobian(nlp["g"], nlp["x"])
        jac_sp = Jg.sparsity()
        sp_png = out_dir / f"jacobian_sparsity_{tag}.png"
        jac_sp_meta = _plot_sparsity(jac_sp, sp_png, title=f"Constraint Jacobian sparsity ({tag})")

        # Additional symbolic function timings
        sym_bench = _bench_symbolic_forms(nlp, n_eval=20, seed=seed)

        # Dynamics eval timings
        dyn_bench = _bench_dynamics_eval_and_jacobian(model, n_eval=200, seed=seed)

        # Solve timing distribution (and parse IPOPT timing stats)
        solve_stats_quiet = _time_solves(
            solver=solver_quiet,
            horizon=horizon,
            Ts=Ts,
            track=track,
            model_phys_for_plant=plant_model,
            n_steps=int(args.n_steps),
            seed=seed,
            ipopt_output_file=None,
        )
        solve_stats_verbose = _time_solves(
            solver=solver_verbose,
            horizon=horizon,
            Ts=Ts,
            track=track,
            model_phys_for_plant=plant_model,
            n_steps=int(args.timing_stats_solves),
            seed=seed,
            ipopt_output_file=ipopt_log,
        )

        results["models"][tag] = {
            "build_ms": float(build_ms),
            "build_verbose_ms": float(build_verbose_ms),
            "nlp_meta": meta,
            "dynamics_bench": dyn_bench,
            "symbolic_bench": sym_bench,
            "jacobian_sparsity": jac_sp_meta,
            "solve_ms": solve_stats_quiet,
            "ipopt_verbose_solve_ms": solve_stats_verbose,
            "artifacts": {
                "ipopt_log": str(ipopt_log),
                "jacobian_sparsity_png": str(sp_png),
            },
        }

    if args.out_json:
        out_json = Path(args.out_json).resolve()
        with out_json.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {out_json}")
    else:
        # default artifact name inside out_dir
        out_json = out_dir / "paper_efficiency_comparison.json"
        with out_json.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()


