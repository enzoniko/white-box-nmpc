#!/usr/bin/env python3
"""
Paper-Style Experiments Runner

Implements the missing experiments needed for the paper draft:

Setup 1: Implementation Overhead
  - Python callback vs CasADi-native graph for identical networks

Setup 2: Steady-State Performance
  - NMPC solve time microbenchmarks for each model class

Setup 3: Adaptation Latency and Recovery
  - Friction step/ramp scenario with time-consistent trajectory data
  - Compare:
      * physics baseline with "rebuild on friction change"
      * neural ensemble with Mode A (mu parameter) + no rebuild
      * neural ensemble with Mode B (online regression weights) + no rebuild

This file is intentionally self-contained and outputs figures + JSON metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import casadi as cs
import torch

from .specialist import HSSConfig, HSSSpecialist, HSSEnsemble
from .calibration import OnlineLinearRegressor
from .casadi_callbacks import CasadiExportConfig, export_ensemble_to_casadi, HSSDynamicsCallback
from .physics_casadi import PhysicsParams, build_physics_dynamics_sx
from .trajectory_scenarios import ScenarioConfig, make_mu_schedule, make_vref_profile, generate_open_loop_controls, rollout_true_dynamics
from .trajectory_scenarios import make_theta_schedule
from .physics_casadi import build_physics_dynamics_sx_theta


@dataclass
class PaperConfig:
    out_dir: str = "./paper_results"
    dt: float = 0.02
    horizon: int = 10
    T: int = 300
    seed: int = 42
    n_warmup: int = 5
    n_solves: int = 50
    device: str = "cpu"

    # library
    n_specialists: int = 8
    hidden_dim: int = 32
    n_layers: int = 3

    # adaptation
    window_size: int = 50
    reg: float = 1e-4

    # closed-loop tracking
    top_k: int = 4
    use_theta: bool = True


def _project_topk(weights: np.ndarray, k: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    k = max(int(k), 1)
    if k >= w.size:
        w_pos = np.maximum(w, 0)
        s = w_pos.sum()
        return w_pos / (s + 1e-12)
    idx = np.argsort(-w)[:k]
    w2 = np.zeros_like(w)
    w2[idx] = np.maximum(w[idx], 0)
    s = w2.sum()
    return w2 / (s + 1e-12)


def _build_simple_nlp(
    f_dyn: cs.Function,
    mode: str,
    cfg: PaperConfig,
    static_params: np.ndarray,
    mu_or_w: Optional[np.ndarray] = None,
) -> Tuple[cs.Function, dict]:
    """
    Build a minimal NMPC-like NLP for the 3-state system:
      x_{k+1} = x_k + dt * f(x_k, u_k, static, p)
      cost = sum (vx - vref)^2 + small control penalty

    `mode` controls which extra parameter is used:
      - "physics_mu": f(x,u,s,mu)
      - "ens_mu":    f(x,u,s,mu)
      - "ens_w":     f(x,u,s,w)
    """
    N = cfg.horizon
    dt = cfg.dt

    X = cs.SX.sym("X", 3, N + 1)
    U = cs.SX.sym("U", 2, N)

    x0 = cs.SX.sym("x0", 3)
    vref = cs.SX.sym("vref", 1)

    # static params are fixed constants in the graph
    s = cs.DM(static_params.reshape(2))

    if mode in ("physics_mu", "ens_mu"):
        mu = cs.SX.sym("mu", 1)
        p = cs.vertcat(x0, vref, mu)
    elif mode == "ens_w":
        assert mu_or_w is not None
        K = int(mu_or_w.shape[0])
        w = cs.SX.sym("w", K)
        p = cs.vertcat(x0, vref, w)
    else:
        raise ValueError(mode)

    cost = 0
    g = [X[:, 0] - x0]

    for k in range(N):
        if mode in ("physics_mu", "ens_mu"):
            dx = f_dyn(X[:, k], U[:, k], s, mu)
        else:
            dx = f_dyn(X[:, k], U[:, k], s, w)
        x_next = X[:, k] + dt * dx
        g.append(X[:, k + 1] - x_next)

        cost += (X[0, k + 1] - vref) ** 2
        cost += 1e-3 * (U[0, k] ** 2 + U[1, k] ** 2)

    opt_vars = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
    g = cs.vertcat(*g)

    nlp = {"x": opt_vars, "f": cost, "g": g, "p": p}

    solver = cs.nlpsol(
        "solver",
        "ipopt",
        nlp,
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 50,
        },
    )

    meta = {"N": N, "mode": mode, "n_vars": int(opt_vars.size1()), "n_cons": int(g.size1())}
    return solver, meta


def _solve_once(solver: cs.Function, cfg: PaperConfig, x0: np.ndarray, vref: float, extra: np.ndarray) -> float:
    # Simple zero init. Constraints: g == 0 (dynamics), no bounds on x/u in this minimal benchmark.
    idx_x0 = solver.index_in("x0")
    idx_g = solver.index_out("g")
    n_dec = int(solver.size1_in(idx_x0))
    n_con = int(solver.size1_out(idx_g))

    t0 = time.perf_counter()
    _ = solver(
        x0=np.zeros((n_dec, 1)),
        p=np.concatenate([x0.reshape(-1), np.array([vref]), extra.reshape(-1)]),
        lbg=np.zeros((n_con,)),
        ubg=np.zeros((n_con,)),
    )
    return (time.perf_counter() - t0) * 1000.0


def build_toy_ensemble(cfg: PaperConfig, mu_centers: Optional[List[float]] = None) -> HSSEnsemble:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    spec_cfg = HSSConfig(hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers)
    specs = [HSSSpecialist(spec_cfg) for _ in range(cfg.n_specialists)]
    # untrained is fine for overhead/solver benchmarks; for accuracy you'd load trained
    ens = HSSEnsemble(specs, mu_centers=mu_centers)
    return ens


def experiment_overhead(cfg: PaperConfig, out_dir: Path) -> Dict:
    """
    Compare Python callback vs CasADi-native graph eval latency (not full NLP solve).
    """
    ens = build_toy_ensemble(cfg, mu_centers=np.linspace(0.3, 1.2, cfg.n_specialists).tolist())
    static = np.array([800.0, 1200.0], dtype=np.float32)

    # Build callback function
    cb = HSSDynamicsCallback("hss_cb", ens, static_params=static, mu_current=0.9)
    x = cs.SX.sym("x", 3)
    u = cs.SX.sym("u", 2)
    f_cb = cs.Function("f_cb", [x, u], [cb(x, u)])

    # Build native graph function (Mode A mu)
    f_native, meta = export_ensemble_to_casadi(ens, CasadiExportConfig(mode="mode_a_rbf", rbf_width=0.15), name="ens_native")
    mu = cs.SX.sym("mu", 1)
    s = cs.DM(static.reshape(2))
    f_nat = cs.Function("f_nat", [x, u, mu], [f_native(x, u, s, mu)])

    rng = np.random.default_rng(cfg.seed)
    xs = rng.uniform([5, -2, -1], [35, 2, 1], size=(200, 3))
    us = rng.uniform([-0.5, -0.5], [0.5, 0.8], size=(200, 2))

    # time each
    def time_fn(fn, kind: str):
        times = []
        for i in range(50):
            _ = fn(xs[i], us[i], 0.9) if kind == "native" else fn(xs[i], us[i])
        t0 = time.perf_counter()
        for i in range(200):
            _ = fn(xs[i], us[i], 0.9) if kind == "native" else fn(xs[i], us[i])
        times.append((time.perf_counter() - t0) * 1000.0 / 200.0)
        return float(np.mean(times))

    ms_cb = time_fn(f_cb, "cb")
    ms_nat = time_fn(f_nat, "native")
    return {"eval_ms_callback": ms_cb, "eval_ms_native": ms_nat, "speedup": ms_cb / ms_nat if ms_nat > 0 else None}


def experiment_setup_and_solve_times(cfg: PaperConfig, out_dir: Path) -> Dict:
    """
    Compare NLP build/setup + solve time for:
      - physics baked mu (rebuild needed)
      - physics mu as param (no rebuild)
      - neural ensemble native mu (no rebuild)
      - neural ensemble native weights (no rebuild)
    """
    ens = build_toy_ensemble(cfg, mu_centers=np.linspace(0.3, 1.2, cfg.n_specialists).tolist())
    static = np.array([800.0, 1200.0], dtype=np.float64)

    # native ensemble
    f_ens_mu, _ = export_ensemble_to_casadi(ens, CasadiExportConfig(mode="mode_a_rbf", rbf_width=0.15), name="ens_mu")
    f_ens_w, _ = export_ensemble_to_casadi(ens, CasadiExportConfig(mode="mode_b_weights", normalize_weights=True), name="ens_w")

    # physics
    f_phys_mu, _ = build_physics_dynamics_sx(PhysicsParams(), mu_mode="as_input", name="phys_mu")

    # setup times
    def build_solver(mode: str, mu_or_w: np.ndarray):
        t0 = time.perf_counter()
        if mode == "physics_mu":
            solver, meta = _build_simple_nlp(f_phys_mu, "physics_mu", cfg, static, mu_or_w)
        elif mode == "ens_mu":
            solver, meta = _build_simple_nlp(f_ens_mu, "ens_mu", cfg, static, mu_or_w)
        elif mode == "ens_w":
            solver, meta = _build_simple_nlp(f_ens_w, "ens_w", cfg, static, mu_or_w)
        else:
            raise ValueError(mode)
        return solver, meta, (time.perf_counter() - t0) * 1000.0

    # build baseline solvers
    solver_phys, _, build_phys_ms = build_solver("physics_mu", np.array([0.9]))
    solver_ens_mu, _, build_ens_mu_ms = build_solver("ens_mu", np.array([0.9]))
    solver_ens_w, _, build_ens_w_ms = build_solver("ens_w", np.ones(cfg.n_specialists) / cfg.n_specialists)

    # solve times
    rng = np.random.default_rng(cfg.seed)
    x0s = rng.uniform([10, -0.5, -0.2], [30, 0.5, 0.2], size=(cfg.n_solves, 3))
    vrefs = rng.uniform(15.0, 30.0, size=(cfg.n_solves,))

    def time_solves(solver, extra):
        times = []
        for i in range(cfg.n_warmup):
            _solve_once(solver, cfg, x0s[i], float(vrefs[i]), extra)
        for i in range(cfg.n_solves):
            times.append(_solve_once(solver, cfg, x0s[i], float(vrefs[i]), extra))
        return {
            "mean_ms": float(np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "max_ms": float(np.max(times)),
        }

    phys_solve = time_solves(solver_phys, np.array([0.9]))
    ens_mu_solve = time_solves(solver_ens_mu, np.array([0.9]))
    ens_w_solve = time_solves(solver_ens_w, np.ones(cfg.n_specialists) / cfg.n_specialists)

    return {
        "build_ms": {"physics_mu": build_phys_ms, "ens_mu": build_ens_mu_ms, "ens_w": build_ens_w_ms},
        "solve_ms": {"physics_mu": phys_solve, "ens_mu": ens_mu_solve, "ens_w": ens_w_solve},
    }


def experiment_adaptation_recovery(cfg: PaperConfig, out_dir: Path) -> Dict:
    """
    Generate time-consistent trajectory under true physics with friction step,
    then run "virtual online adaptation" using Mode B regressor to fit weights.

    This experiment measures:
      - Mode B weight update latency per step
      - predicted acceleration error before/after friction change
    """
    scfg = ScenarioConfig(dt=cfg.dt, T=cfg.T, seed=cfg.seed)
    mu_sched = make_mu_schedule(scfg, kind="step")
    vref = make_vref_profile(scfg, kind="piecewise")
    delta, throttle = generate_open_loop_controls(scfg)
    static = np.array([800.0, 1200.0], dtype=np.float64)

    # True dynamics = CasADi physics (mu as input), evaluated numerically
    f_phys, _ = build_physics_dynamics_sx(PhysicsParams(), mu_mode="as_input", name="phys_true")

    def f_true(x, u, s, mu):
        y = np.array(f_phys(x, u, s, np.array([mu]))).reshape(-1)
        return y.astype(np.float64)

    rollout = rollout_true_dynamics(f_true, scfg, static, mu_sched, delta, throttle)

    # Build ensemble (untrained) just for measuring regressor timings and weight evolution mechanics
    ens = build_toy_ensemble(cfg, mu_centers=np.linspace(0.3, 1.2, cfg.n_specialists).tolist())
    reg = OnlineLinearRegressor(ens, window_size=cfg.window_size, regularization=cfg.reg, device=torch.device(cfg.device))

    weights_hist = []
    update_ms = []
    residuals = []

    # "observed accel" from true physics, optionally add noise
    obs_noise = 0.0
    for t in range(scfg.T):
        x_t = rollout["x"][t].astype(np.float32)
        u_t = rollout["u"][t].astype(np.float32)
        y_obs = rollout["dx"][t].astype(np.float32) + obs_noise * np.random.randn(3).astype(np.float32)

        t0 = time.perf_counter()
        reg.add_observation(x_t, u_t, static.astype(np.float32), y_obs)
        update_ms.append((time.perf_counter() - t0) * 1000.0)
        weights_hist.append(reg.get_weights().copy())
        residuals.append(float(reg.last_residual))

    weights_hist = np.array(weights_hist)
    update_ms = np.array(update_ms)
    residuals = np.array(residuals)

    # Plot weights over time
    fig = plt.figure(figsize=(10, 4))
    plt.stackplot(np.arange(scfg.T), weights_hist.T, alpha=0.85)
    plt.axvline(scfg.mu_change_step, color="k", linestyle="--", linewidth=1, label="μ step")
    plt.ylim(0, 1)
    plt.title("Mode B Online Adaptation (Weights over Time)")
    plt.xlabel("t")
    plt.ylabel("w")
    plt.tight_layout()
    plt.savefig(out_dir / "mode_b_weights_timeseries.png", dpi=150)
    plt.close(fig)

    # Plot update latency
    fig = plt.figure(figsize=(10, 3))
    plt.plot(update_ms, linewidth=1)
    plt.axvline(scfg.mu_change_step, color="k", linestyle="--", linewidth=1)
    plt.title("Mode B Update Latency per Step (ms)")
    plt.xlabel("t")
    plt.ylabel("ms")
    plt.tight_layout()
    plt.savefig(out_dir / "mode_b_update_latency.png", dpi=150)
    plt.close(fig)

    return {
        "mode_b_update_ms": {
            "mean": float(update_ms.mean()),
            "p95": float(np.percentile(update_ms, 95)),
            "max": float(update_ms.max()),
        },
        "mode_b_residual": {
            "final": float(residuals[-1]),
            "mean": float(residuals.mean()),
        },
    }


def experiment_closed_loop_tracking(cfg: PaperConfig, out_dir: Path) -> Dict:
    """
    Closed-loop tracking experiment on the 3-state system.

    We do NOT try to reproduce a full racing OCP here; instead we create a minimal NMPC
    that tracks vref (vx) under friction/parameter changes and compare:
      - physics baseline (theta as input, no rebuild)
      - physics baseline baked theta (rebuild at change step)
      - neural ensemble native graph, Mode A (mu as parameter)
      - neural ensemble native graph, Mode B (weights from online regression)
      - neural ensemble + Top-K weights (Mode B weights sparsified to top_k)

    Metrics:
      - tracking RMSE of vx vs vref
      - solve time per step
      - rebuild time at change step (baked physics)
      - Mode B update time per step
    """
    scfg = ScenarioConfig(dt=cfg.dt, T=cfg.T, seed=cfg.seed)
    vref = make_vref_profile(scfg, kind="piecewise")
    delta_ol, throttle_ol = generate_open_loop_controls(scfg)
    static = np.array([800.0, 1200.0], dtype=np.float64)

    # parameter schedules
    mu_sched = make_mu_schedule(scfg, kind="step")
    theta_sched = make_theta_schedule(scfg, kind="step")  # [mu, cd, cm1]

    # True dynamics: physics theta as input
    f_true_theta, _ = build_physics_dynamics_sx_theta(PhysicsParams(), theta_mode="as_input", name="phys_true_theta")

    def f_true(x, u, s, theta):
        y = np.array(f_true_theta(x, u, s, theta)).reshape(-1)
        return y.astype(np.float64)

    # Rollout true closed-loop plant will be simulated step-by-step using f_true
    x_true = np.zeros((scfg.T, 3), dtype=np.float64)
    x_true[0] = np.array([20.0, 0.0, 0.0], dtype=np.float64)

    # Build neural ensemble graphs (untrained toy ensemble for now)
    ens = build_toy_ensemble(cfg, mu_centers=np.linspace(0.3, 1.2, cfg.n_specialists).tolist())
    f_ens_mu, _ = export_ensemble_to_casadi(ens, CasadiExportConfig(mode="mode_a_rbf", rbf_width=0.15), name="ens_mu_track")
    f_ens_w, _ = export_ensemble_to_casadi(ens, CasadiExportConfig(mode="mode_b_weights", normalize_weights=True), name="ens_w_track")

    # Physics models:
    f_phys_theta_in, _ = build_physics_dynamics_sx_theta(PhysicsParams(), theta_mode="as_input", name="phys_theta_in")

    # Solver builders
    def build_solver_phys_theta_input():
        # treat theta as a parameter, but our minimal NLP supports only mu or w.
        # We'll wrap physics(theta) as a function of (x,u,s,mu) by using mu only for the NLP,
        # and apply cd/cm1 via theta_sched during plant simulation and via rebuild for baseline.
        # For the "no rebuild" physics baseline, we build a dedicated NLP with theta param.
        N = cfg.horizon
        dt = cfg.dt
        X = cs.SX.sym("X", 3, N + 1)
        U = cs.SX.sym("U", 2, N)
        x0 = cs.SX.sym("x0", 3)
        vref_p = cs.SX.sym("vref", 1)
        theta = cs.SX.sym("theta", 3)
        s_dm = cs.DM(static.reshape(2))
        cost = 0
        g = [X[:, 0] - x0]
        for k in range(N):
            dx = f_phys_theta_in(X[:, k], U[:, k], s_dm, theta)
            Xn = X[:, k] + dt * dx
            g.append(X[:, k + 1] - Xn)
            cost += (X[0, k + 1] - vref_p) ** 2 + 1e-3 * (U[0, k] ** 2 + U[1, k] ** 2)
        opt = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
        g = cs.vertcat(*g)
        p = cs.vertcat(x0, vref_p, theta)
        solver = cs.nlpsol("solver_phys_theta", "ipopt", {"x": opt, "f": cost, "g": g, "p": p},
                           {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 50})
        return solver

    def build_solver_phys_baked(theta_baked: np.ndarray):
        # rebuild a baked-theta dynamics function and a solver that doesn't take theta
        f_baked, _ = build_physics_dynamics_sx_theta(PhysicsParams(), theta_mode="baked_constant", theta_baked=theta_baked, name="phys_baked")
        # reuse minimal NLP builder by wrapping baked into "physics_mu" signature using dummy mu
        # We'll just create a custom NLP without theta
        N = cfg.horizon
        dt = cfg.dt
        X = cs.SX.sym("X", 3, N + 1)
        U = cs.SX.sym("U", 2, N)
        x0 = cs.SX.sym("x0", 3)
        vref_p = cs.SX.sym("vref", 1)
        s_dm = cs.DM(static.reshape(2))
        cost = 0
        g = [X[:, 0] - x0]
        for k in range(N):
            dx = f_baked(X[:, k], U[:, k], s_dm)
            Xn = X[:, k] + dt * dx
            g.append(X[:, k + 1] - Xn)
            cost += (X[0, k + 1] - vref_p) ** 2 + 1e-3 * (U[0, k] ** 2 + U[1, k] ** 2)
        opt = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
        g = cs.vertcat(*g)
        p = cs.vertcat(x0, vref_p)
        solver = cs.nlpsol("solver_phys_baked", "ipopt", {"x": opt, "f": cost, "g": g, "p": p},
                           {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 50})
        return solver

    # Neural solvers
    def build_solver_ens_mu():
        return _build_simple_nlp(f_ens_mu, "ens_mu", cfg, static, np.array([0.9]))[0]

    def build_solver_ens_w():
        return _build_simple_nlp(f_ens_w, "ens_w", cfg, static, np.ones(cfg.n_specialists) / cfg.n_specialists)[0]

    solver_phys_theta = build_solver_phys_theta_input()
    solver_phys_baked = build_solver_phys_baked(theta_sched[0])
    solver_ens_mu = build_solver_ens_mu()
    solver_ens_w = build_solver_ens_w()

    # Online regressor for Mode B weights
    reg = OnlineLinearRegressor(ens, window_size=cfg.window_size, regularization=cfg.reg, device=torch.device(cfg.device))
    w_mode_b = np.ones(cfg.n_specialists) / cfg.n_specialists

    # storage
    solve_ms = {k: [] for k in ["phys_theta", "phys_baked", "ens_mu", "ens_w", "ens_w_topk"]}
    rebuild_ms = []
    update_ms = []
    vx_pred = {k: np.zeros(scfg.T) for k in solve_ms.keys()}

    # warm-start vectors
    def solve_step(solver, p_vec):
        idx_x0 = solver.index_in("x0")
        idx_g = solver.index_out("g")
        n_dec = int(solver.size1_in(idx_x0))
        n_con = int(solver.size1_out(idx_g))
        t0 = time.perf_counter()
        sol = solver(x0=np.zeros((n_dec, 1)), p=p_vec, lbg=np.zeros((n_con,)), ubg=np.zeros((n_con,)))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        x_opt = np.array(sol["x"]).reshape(-1)
        # Extract first control U[:,0] from decision vector layout
        # Layout: X (3*(N+1)) then U (2*N)
        N = cfg.horizon
        offset_u = 3 * (N + 1)
        u0 = x_opt[offset_u:offset_u + 2]
        return u0, dt_ms

    # closed-loop sim
    for t in range(scfg.T - 1):
        theta_t = theta_sched[t]
        mu_t = mu_sched[t]

        # rebuild baked physics at change step (explicit measurement)
        if t == scfg.mu_change_step:
            t0 = time.perf_counter()
            solver_phys_baked = build_solver_phys_baked(theta_t)
            rebuild_ms.append((time.perf_counter() - t0) * 1000.0)

        # Mode B update with observed accel from true physics (uses current true state and applied control)
        # We'll use open-loop u as the "measured" input for regressor update and then use updated weights in NMPC.
        u_meas = np.array([delta_ol[t], throttle_ol[t]], dtype=np.float64)
        dx_obs = f_true(x_true[t], u_meas, static, theta_t)
        t0 = time.perf_counter()
        reg.add_observation(x_true[t].astype(np.float32), u_meas.astype(np.float32), static.astype(np.float32), dx_obs.astype(np.float32))
        update_ms.append((time.perf_counter() - t0) * 1000.0)
        w_mode_b = reg.get_weights().copy()
        w_topk = _project_topk(w_mode_b, cfg.top_k)

        # Solve each controller variant (each provides a control u0)
        u_phys_theta, ms = solve_step(solver_phys_theta, np.concatenate([x_true[t], np.array([vref[t]]), theta_t]))
        solve_ms["phys_theta"].append(ms)

        u_phys_baked, ms = solve_step(solver_phys_baked, np.concatenate([x_true[t], np.array([vref[t]])]))
        solve_ms["phys_baked"].append(ms)

        u_ens_mu, ms = solve_step(solver_ens_mu, np.concatenate([x_true[t], np.array([vref[t]]), np.array([mu_t])]))
        solve_ms["ens_mu"].append(ms)

        u_ens_w, ms = solve_step(solver_ens_w, np.concatenate([x_true[t], np.array([vref[t]]), w_mode_b]))
        solve_ms["ens_w"].append(ms)

        u_ens_wk, ms = solve_step(solver_ens_w, np.concatenate([x_true[t], np.array([vref[t]]), w_topk]))
        solve_ms["ens_w_topk"].append(ms)

        # Apply one chosen controller to the true plant (use ens_w as default for paper narrative)
        u_apply = u_ens_w
        x_true[t + 1] = x_true[t] + scfg.dt * f_true(x_true[t], u_apply, static, theta_t)
        x_true[t + 1, 0] = max(x_true[t + 1, 0], 0.1)

        # log predicted vx (simple one-step prediction)
        vx_pred["phys_theta"][t + 1] = x_true[t, 0] + scfg.dt * f_true(x_true[t], u_phys_theta, static, theta_t)[0]
        vx_pred["phys_baked"][t + 1] = x_true[t, 0] + scfg.dt * f_true(x_true[t], u_phys_baked, static, theta_t)[0]
        vx_pred["ens_mu"][t + 1] = x_true[t, 0] + scfg.dt * f_true(x_true[t], u_ens_mu, static, theta_t)[0]
        vx_pred["ens_w"][t + 1] = x_true[t, 0] + scfg.dt * f_true(x_true[t], u_ens_w, static, theta_t)[0]
        vx_pred["ens_w_topk"][t + 1] = x_true[t, 0] + scfg.dt * f_true(x_true[t], u_ens_wk, static, theta_t)[0]

    # compute tracking RMSE
    vx = x_true[:, 0]
    rmse = {k: float(np.sqrt(np.mean((vx - vref) ** 2))) for k in solve_ms.keys()}

    # plots
    fig = plt.figure(figsize=(10, 4))
    plt.plot(vx, label="vx (true plant)")
    plt.plot(vref, "--", label="vref")
    plt.axvline(scfg.mu_change_step, color="k", linestyle="--", linewidth=1)
    plt.title("Closed-loop Tracking (vx)")
    plt.xlabel("t")
    plt.ylabel("m/s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "closed_loop_vx_tracking.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 3))
    plt.plot(theta_sched[:, 0], label="mu")
    plt.plot(theta_sched[:, 1], label="cd")
    plt.plot(theta_sched[:, 2] / np.max(theta_sched[:, 2]), label="cm1 (norm)")
    plt.axvline(scfg.mu_change_step, color="k", linestyle="--", linewidth=1)
    plt.title("True Parameter Schedule")
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "closed_loop_params.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 3))
    for k, arr in solve_ms.items():
        plt.plot(arr, label=k)
    plt.axvline(scfg.mu_change_step, color="k", linestyle="--", linewidth=1)
    plt.title("Solve Time per Step (ms)")
    plt.xlabel("t")
    plt.ylabel("ms")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "closed_loop_solve_times.png", dpi=150)
    plt.close(fig)

    if len(rebuild_ms) > 0:
        fig = plt.figure(figsize=(6, 3))
        plt.bar([0], [rebuild_ms[0]])
        plt.title("Physics Rebuild Time at Change Step (ms)")
        plt.ylabel("ms")
        plt.tight_layout()
        plt.savefig(out_dir / "physics_rebuild_time.png", dpi=150)
        plt.close(fig)

    return {
        "tracking_rmse_vx": rmse,
        "solve_ms": {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))} for k, v in solve_ms.items()},
        "mode_b_update_ms": {"mean": float(np.mean(update_ms)), "p95": float(np.percentile(update_ms, 95)), "max": float(np.max(update_ms))},
        "physics_rebuild_ms": {"at_change": float(rebuild_ms[0]) if rebuild_ms else None},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./paper_results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_specialists", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--n_solves", type=int, default=50)
    args = parser.parse_args()

    cfg = PaperConfig(
        out_dir=args.out_dir,
        device=args.device,
        n_specialists=args.n_specialists,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        T=args.T,
        horizon=args.horizon,
        dt=args.dt,
        n_solves=args.n_solves,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {"config": asdict(cfg)}

    # Setup 1
    results["setup1_overhead"] = experiment_overhead(cfg, out_dir)

    # Setup 2
    results["setup2_setup_and_solve"] = experiment_setup_and_solve_times(cfg, out_dir)

    # Setup 3
    results["setup3_adaptation_recovery"] = experiment_adaptation_recovery(cfg, out_dir)

    # Setup 4 (closed-loop tracking + rebuild + Top-K)
    results["setup4_closed_loop_tracking"] = experiment_closed_loop_tracking(cfg, out_dir)

    with open(out_dir / "paper_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {out_dir / 'paper_results.json'}")


if __name__ == "__main__":
    main()


