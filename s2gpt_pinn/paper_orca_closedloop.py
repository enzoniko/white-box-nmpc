#!/usr/bin/env python3
"""
Paper Experiment: ORCA Closed-Loop Tracking (trained library, discriminative RMSE)

This addresses two paper-readiness gaps:
  1) uses a *trained* specialist library (not toy/untrained)
  2) computes tracking RMSE per controller by simulating separate plant rollouts

Plant:
  ORCA-scale bicycle accel dynamics (BayesRace-consistent simplified Pacejka).

Controllers (all solve the same minimal 3-state NMPC):
  - phys_mu:    exact physics model with mu_scale as a solver parameter (no rebuild on change)
  - phys_baked: exact physics model but mu_scale baked as a constant (explicit rebuild at change)
  - ens_mu:     trained ensemble, Mode A (mu_scale -> RBF weights inside graph)
  - ens_w:      trained ensemble, Mode B weights as NLP parameter (weights updated online)
  - ens_w_topk: same as ens_w but weights projected to Top-K before passing to NMPC

We report:
  - tracking RMSE vs vref for vx, and optionally full state norms
  - solve-time distributions
  - rebuild spike for phys_baked at change step
  - Mode-B weight update latency

Optional:
  - JIT/codegen rebuild timing for phys_baked via CasADi's JIT (if compiler available).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import casadi as cs
import torch

from .orca_physics import OrcaParams, accelerations_numpy, build_orca_dynamics_sx_mu
from .orca_library import load_orca_library, pick_in_out_mu
from .calibration import OnlineLinearRegressor
from .casadi_callbacks import CasadiExportConfig, export_ensemble_to_casadi
from .friction_estimators import RLSMuEstimator, RLSConfig, UKFMuEstimator, UKFConfig

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


@dataclass
class OrcaClosedLoopConfig:
    out_dir: str = "./paper_orca_closedloop_results"
    lib_dir: str = "./orca_library_trained"
    seed: int = 42
    dt: float = 0.02
    T: int = 300
    horizon: int = 10
    top_k: int = 4
    # vref profile (m/s)
    vref_low: float = 1.5
    vref_high: float = 3.0
    vref_change_step: int = 120
    # mu schedule
    mu_change_step: int = 150
    out_multiplier: float = 1.25
    # NMPC bounds
    delta_max: float = 0.35
    pwm_min: float = -0.5
    pwm_max: float = 1.0
    # Solver options
    ipopt_max_iter: int = 80
    # JIT compilation options for rebuild measurement
    use_jit_for_baked: bool = False

    # Estimator configs (for phys_mu baselines)
    rls_lambda: float = 0.98
    rls_initial_P: float = 100.0
    rls_adaptation_window: int = 50  # used for convergence reporting, not required for RLS math
    ukf_alpha: float = 1e-3
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    ukf_process_noise: float = 1e-4
    ukf_measurement_noise: float = 0.1

    # Failure-mode controls / study knobs
    failure_case: str = "normal"  # normal|no_excitation|short_window|extreme_mu
    estimator_max_update_steps: Optional[int] = None  # if set: stop estimator updates after N steps
    modeb_window_size: int = 40
    modeb_regularization: float = 1e-4

    # Progress / UX
    show_progress: bool = False


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


def _build_nlp_3state(
    f_dyn: cs.Function,
    mode: str,
    cfg: OrcaClosedLoopConfig,
    static_params: np.ndarray,
    extra_dim: int,
    jit: bool = False,
    solver_name: str = "solver",
) -> cs.Function:
    """
    Minimal NMPC for 3-state system:
      x_{k+1} = x_k + dt * f(x_k, u_k, static, extra)
      cost = sum (vx - vref)^2 + small control penalty

    f_dyn signature:
      - mode="mu": expects f(x,u,s,mu)
      - mode="w":  expects f(x,u,s,w)
    """
    N = int(cfg.horizon)
    dt = float(cfg.dt)

    X = cs.SX.sym("X", 3, N + 1)
    U = cs.SX.sym("U", 2, N)
    x0 = cs.SX.sym("x0", 3)
    vref = cs.SX.sym("vref", 1)
    extra = cs.SX.sym("extra", extra_dim)

    s = cs.DM(np.asarray(static_params, dtype=np.float64).reshape(2))

    cost = 0
    g = [X[:, 0] - x0]
    for k in range(N):
        dx = f_dyn(X[:, k], U[:, k], s, extra)
        g.append(X[:, k + 1] - (X[:, k] + dt * dx))
        cost += (X[0, k + 1] - vref) ** 2
        cost += 1e-3 * (U[0, k] ** 2) + 1e-4 * (U[1, k] ** 2)

    opt_vars = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
    g = cs.vertcat(*g)
    p = cs.vertcat(x0, vref, extra)

    nlp = {"x": opt_vars, "f": cost, "g": g, "p": p}

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": int(cfg.ipopt_max_iter),
    }
    if jit:
        # May require a working compiler toolchain. If not available, CasADi throws.
        opts.update({"jit": True, "compiler": "shell", "jit_options": {"flags": "-O2"}})

    solver = cs.nlpsol(solver_name, "ipopt", nlp, opts)
    return solver


def _decision_bounds(cfg: OrcaClosedLoopConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lbx/ubx for decision vector [X(:); U(:)].
    """
    N = int(cfg.horizon)
    nX = 3 * (N + 1)
    nU = 2 * N
    n = nX + nU
    lbx = -np.inf * np.ones((n,), dtype=np.float64)
    ubx = np.inf * np.ones((n,), dtype=np.float64)

    # State bounds (keep numerically stable)
    # X = [vx, vy, omega]
    for k in range(N + 1):
        i = 3 * k
        lbx[i + 0] = 0.05
        ubx[i + 0] = 8.0
        lbx[i + 1] = -3.0
        ubx[i + 1] = 3.0
        lbx[i + 2] = -30.0
        ubx[i + 2] = 30.0

    # Control bounds
    off_u = nX
    for k in range(N):
        j = off_u + 2 * k
        lbx[j + 0] = -float(cfg.delta_max)
        ubx[j + 0] = float(cfg.delta_max)
        lbx[j + 1] = float(cfg.pwm_min)
        ubx[j + 1] = float(cfg.pwm_max)

    return lbx, ubx


def _solve_step(
    solver: cs.Function,
    x0: np.ndarray,
    vref: float,
    extra: np.ndarray,
    lbx: np.ndarray,
    ubx: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, float]:
    idx_x0 = solver.index_in("x0")
    idx_g = solver.index_out("g")
    n_dec = int(solver.size1_in(idx_x0))
    n_con = int(solver.size1_out(idx_g))

    t0 = time.perf_counter()
    sol = solver(
        x0=np.zeros((n_dec, 1)),
        p=np.concatenate([x0.reshape(-1), np.array([vref]), extra.reshape(-1)]),
        lbg=np.zeros((n_con,)),
        ubg=np.zeros((n_con,)),
        lbx=lbx,
        ubx=ubx,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    x_opt = np.array(sol["x"]).reshape(-1)

    # decision layout: X then U
    N = int(horizon)
    off_u = 3 * (N + 1)
    u0 = x_opt[off_u : off_u + 2]
    return u0, dt_ms


def _make_vref(cfg: OrcaClosedLoopConfig) -> np.ndarray:
    vref = np.ones((cfg.T,), dtype=np.float64) * float(cfg.vref_low)
    vref[int(cfg.vref_change_step) :] = float(cfg.vref_high)
    return vref


def _make_mu_schedule(cfg: OrcaClosedLoopConfig, mu_in: float, mu_out: float, kind: str) -> np.ndarray:
    mu = np.ones((cfg.T,), dtype=np.float64) * float(mu_in)
    if kind == "in":
        # step but stays in-manifold
        mu[int(cfg.mu_change_step) :] = float(mu_in)
        return mu
    if kind == "out":
        mu[int(cfg.mu_change_step) :] = float(mu_out)
        return mu
    raise ValueError(kind)


def run_closed_loop(cfg: OrcaClosedLoopConfig, mu_sched: np.ndarray, out_dir: Path, tag: str) -> Dict:
    rng = np.random.default_rng(int(cfg.seed))

    # Load trained library
    lib = load_orca_library(cfg.lib_dir, device=torch.device("cpu"))
    orca = OrcaParams()
    static = lib.static_params.astype(np.float64)

    mu_min, mu_max = float(np.min(lib.mu_scale_grid)), float(np.max(lib.mu_scale_grid))
    mu_in, mu_out = pick_in_out_mu(mu_min, mu_max, out_multiplier=float(cfg.out_multiplier))

    vref = _make_vref(cfg)

    # Failure-mode: remove lateral excitation by constraining delta and keeping vref constant.
    if str(cfg.failure_case).lower() == "no_excitation":
        vref = np.ones_like(vref) * float(cfg.vref_low)

    # Build CasADi dynamics (physics)
    f_phys_mu = build_orca_dynamics_sx_mu("orca_phys_mu", static_params=static, params=orca, mu_mode="as_input")
    # Wrap to match signature f(x,u,s,mu) by ignoring s inside wrapper
    x = cs.SX.sym("x", 3)
    u = cs.SX.sym("u", 2)
    s = cs.SX.sym("s", 2)
    mu = cs.SX.sym("mu", 1)
    f_phys_mu_wrap = cs.Function("phys_mu_wrap", [x, u, s, mu], [f_phys_mu(x, u, mu)], ["x", "u", "s", "mu"], ["dx"])

    # Baked physics builder (rebuild on change)
    def build_phys_baked_solver(mu_baked: float, solver_name: str):
        f_baked = build_orca_dynamics_sx_mu(solver_name + "_f", static_params=static, params=orca, mu_mode="baked_constant", mu_baked=float(mu_baked))
        f_baked_wrap = cs.Function("phys_baked_wrap_" + solver_name, [x, u, s, mu], [f_baked(x, u)], ["x", "u", "s", "mu"], ["dx"])
        try:
            return _build_nlp_3state(
                f_dyn=f_baked_wrap,
                mode="mu",
                cfg=cfg,
                static_params=static,
                extra_dim=1,
                jit=bool(cfg.use_jit_for_baked),
                solver_name=solver_name,
            )
        except Exception:
            # Fallback if toolchain for JIT isn't available.
            return _build_nlp_3state(
                f_dyn=f_baked_wrap,
                mode="mu",
                cfg=cfg,
                static_params=static,
                extra_dim=1,
                jit=False,
                solver_name=solver_name,
            )

    # Surrogate native graphs
    f_ens_mu, _ = export_ensemble_to_casadi(
        lib.ensemble,
        CasadiExportConfig(mode="mode_a_rbf", rbf_width=float(getattr(lib.ensemble, "rbf_width", 0.08))),
        name="ens_mu_orca",
    )
    f_ens_w, _ = export_ensemble_to_casadi(
        lib.ensemble,
        CasadiExportConfig(mode="mode_b_weights", normalize_weights=True),
        name="ens_w_orca",
    )

    # Build solvers
    solver_phys_mu = _build_nlp_3state(f_phys_mu_wrap, "mu", cfg, static, extra_dim=1, jit=False, solver_name="sol_phys_mu_" + tag)
    solver_phys_baked = build_phys_baked_solver(mu_baked=float(mu_sched[0]), solver_name="sol_phys_baked_" + tag + "_0")
    solver_ens_mu = _build_nlp_3state(f_ens_mu, "mu", cfg, static, extra_dim=1, jit=False, solver_name="sol_ens_mu_" + tag)
    solver_ens_w = _build_nlp_3state(f_ens_w, "w", cfg, static, extra_dim=int(lib.ensemble.n_specialists), jit=False, solver_name="sol_ens_w_" + tag)

    lbx, ubx = _decision_bounds(cfg)

    # Controllers to simulate separately
    names = ["phys_mu", "phys_mu_rls", "phys_mu_ukf", "phys_baked", "ens_mu", "ens_w", "ens_w_topk"]
    x_traj = {k: np.zeros((cfg.T, 3), dtype=np.float64) for k in names}
    u_traj = {k: np.zeros((cfg.T, 2), dtype=np.float64) for k in names}
    solve_ms = {k: [] for k in names}
    rebuild_ms: Dict[str, Optional[float]] = {"phys_baked": None}
    update_ms = {k: [] for k in ["ens_w", "ens_w_topk", "phys_mu_rls", "phys_mu_ukf"]}

    # Estimator time-series (for convergence + conditioning diagnostics)
    mu_est = {k: np.zeros((cfg.T,), dtype=np.float64) for k in ["phys_mu_rls", "phys_mu_ukf"]}
    ukf_condS = np.full((cfg.T,), np.nan, dtype=np.float64)
    rls_denom = np.full((cfg.T,), np.nan, dtype=np.float64)

    # init states
    for k in names:
        x_traj[k][0] = np.array([2.0, 0.0, 0.0], dtype=np.float64)

    # Mode-B regressors (separate per controller rollout)
    # Failure-mode: short adaptation window (<10) for Mode-B
    modeb_window = int(cfg.modeb_window_size)
    if str(cfg.failure_case).lower() == "short_window":
        modeb_window = min(modeb_window, 8)
    reg_w = OnlineLinearRegressor(lib.ensemble, window_size=modeb_window, regularization=float(cfg.modeb_regularization), device=torch.device("cpu"))
    reg_wk = OnlineLinearRegressor(lib.ensemble, window_size=modeb_window, regularization=float(cfg.modeb_regularization), device=torch.device("cpu"))
    w_modeb = np.ones(lib.ensemble.n_specialists, dtype=np.float64) / float(lib.ensemble.n_specialists)
    w_modeb_k = w_modeb.copy()

    # Initialize mu estimators (start at mu_in)
    mu0 = float(mu_sched[0])
    rls = RLSMuEstimator(
        mu0=mu0,
        cfg=RLSConfig(lam=float(cfg.rls_lambda), initial_P=float(cfg.rls_initial_P), max_update_steps=cfg.estimator_max_update_steps),
    )
    ukf = UKFMuEstimator(
        mu0=mu0,
        cfg=UKFConfig(
            alpha=float(cfg.ukf_alpha),
            beta=float(cfg.ukf_beta),
            kappa=float(cfg.ukf_kappa),
            process_noise=float(cfg.ukf_process_noise),
            measurement_noise=float(cfg.ukf_measurement_noise),
            max_update_steps=cfg.estimator_max_update_steps,
        ),
    )
    mu_est["phys_mu_rls"][0] = mu0
    mu_est["phys_mu_ukf"][0] = mu0

    # Track Mode-B weight convergence after change step (diagnostic)
    w_hist = {k: np.zeros((cfg.T, int(lib.ensemble.n_specialists)), dtype=np.float64) for k in ["ens_w", "ens_w_topk"]}
    w_hist["ens_w"][0] = reg_w.get_weights().copy()
    w_hist["ens_w_topk"][0] = _project_topk(reg_wk.get_weights().copy(), cfg.top_k)

    it = range(cfg.T - 1)
    if cfg.show_progress and tqdm is not None:
        it = tqdm(it, desc=f"ORCA closed-loop [{tag}] seed={cfg.seed}", total=int(cfg.T - 1), leave=False)

    for t in it:
        mu_t = float(mu_sched[t])

        # rebuild baked physics solver at change step
        if t == int(cfg.mu_change_step):
            t0 = time.perf_counter()
            solver_phys_baked = build_phys_baked_solver(mu_baked=mu_t, solver_name=f"sol_phys_baked_{tag}_{t}")
            rebuild_ms["phys_baked"] = (time.perf_counter() - t0) * 1000.0

        # 1) phys_mu rollout
        u0, ms = _solve_step(solver_phys_mu, x_traj["phys_mu"][t], float(vref[t]), np.array([mu_t]), lbx, ubx, horizon=cfg.horizon)
        solve_ms["phys_mu"].append(ms)
        u_traj["phys_mu"][t] = u0
        dx = accelerations_numpy(x_traj["phys_mu"][t], u0, static, orca, mu_scale=mu_t)
        x_traj["phys_mu"][t + 1] = x_traj["phys_mu"][t] + cfg.dt * dx
        x_traj["phys_mu"][t + 1, 0] = max(x_traj["phys_mu"][t + 1, 0], 0.05)

        # 1b) phys_mu_rls rollout: NMPC uses mu_est (no rebuild), estimator updates from measured dv
        mu_rls_t = float(mu_est["phys_mu_rls"][t])
        u0, ms = _solve_step(solver_phys_mu, x_traj["phys_mu_rls"][t], float(vref[t]), np.array([mu_rls_t]), lbx, ubx, horizon=cfg.horizon)
        solve_ms["phys_mu_rls"].append(ms)
        u_traj["phys_mu_rls"][t] = u0
        dx = accelerations_numpy(x_traj["phys_mu_rls"][t], u0, static, orca, mu_scale=mu_t)  # plant uses true mu_t
        x_traj["phys_mu_rls"][t + 1] = x_traj["phys_mu_rls"][t] + cfg.dt * dx
        x_traj["phys_mu_rls"][t + 1, 0] = max(x_traj["phys_mu_rls"][t + 1, 0], 0.05)
        t0 = time.perf_counter()
        rls_info = rls.update(
            dv_obs=dx,
            dv_pred_fn=lambda mu_val: accelerations_numpy(x_traj["phys_mu_rls"][t], u0, static, orca, mu_scale=float(mu_val)),
        )
        update_ms["phys_mu_rls"].append((time.perf_counter() - t0) * 1000.0)
        mu_est["phys_mu_rls"][t + 1] = float(rls.mu)
        if "denom_min" in rls_info:
            rls_denom[t] = float(rls_info["denom_min"])

        # 1c) phys_mu_ukf rollout: NMPC uses mu_est (no rebuild), estimator updates from measured dv
        mu_ukf_t = float(mu_est["phys_mu_ukf"][t])
        u0, ms = _solve_step(solver_phys_mu, x_traj["phys_mu_ukf"][t], float(vref[t]), np.array([mu_ukf_t]), lbx, ubx, horizon=cfg.horizon)
        solve_ms["phys_mu_ukf"].append(ms)
        u_traj["phys_mu_ukf"][t] = u0
        dx = accelerations_numpy(x_traj["phys_mu_ukf"][t], u0, static, orca, mu_scale=mu_t)
        x_traj["phys_mu_ukf"][t + 1] = x_traj["phys_mu_ukf"][t] + cfg.dt * dx
        x_traj["phys_mu_ukf"][t + 1, 0] = max(x_traj["phys_mu_ukf"][t + 1, 0], 0.05)
        t0 = time.perf_counter()
        ukf_info = ukf.update(
            dv_obs=dx,
            dv_pred_fn=lambda mu_val: accelerations_numpy(x_traj["phys_mu_ukf"][t], u0, static, orca, mu_scale=float(mu_val)),
        )
        update_ms["phys_mu_ukf"].append((time.perf_counter() - t0) * 1000.0)
        mu_est["phys_mu_ukf"][t + 1] = float(ukf.mu)
        if "cond_S" in ukf_info:
            ukf_condS[t] = float(ukf_info["cond_S"])

        # 2) phys_baked rollout
        u0, ms = _solve_step(solver_phys_baked, x_traj["phys_baked"][t], float(vref[t]), np.array([0.0]), lbx, ubx, horizon=cfg.horizon)
        solve_ms["phys_baked"].append(ms)
        u_traj["phys_baked"][t] = u0
        dx = accelerations_numpy(x_traj["phys_baked"][t], u0, static, orca, mu_scale=mu_t)
        x_traj["phys_baked"][t + 1] = x_traj["phys_baked"][t] + cfg.dt * dx
        x_traj["phys_baked"][t + 1, 0] = max(x_traj["phys_baked"][t + 1, 0], 0.05)

        # 3) ens_mu rollout (Mode A)
        u0, ms = _solve_step(solver_ens_mu, x_traj["ens_mu"][t], float(vref[t]), np.array([mu_t]), lbx, ubx, horizon=cfg.horizon)
        solve_ms["ens_mu"].append(ms)
        u_traj["ens_mu"][t] = u0
        dx = accelerations_numpy(x_traj["ens_mu"][t], u0, static, orca, mu_scale=mu_t)
        x_traj["ens_mu"][t + 1] = x_traj["ens_mu"][t] + cfg.dt * dx
        x_traj["ens_mu"][t + 1, 0] = max(x_traj["ens_mu"][t + 1, 0], 0.05)

        # 4) ens_w rollout (Mode B): use weights from past observations, then update after applying.
        w_modeb = reg_w.get_weights().copy()
        u0, ms = _solve_step(solver_ens_w, x_traj["ens_w"][t], float(vref[t]), w_modeb, lbx, ubx, horizon=cfg.horizon)
        solve_ms["ens_w"].append(ms)
        u_traj["ens_w"][t] = u0
        dx = accelerations_numpy(x_traj["ens_w"][t], u0, static, orca, mu_scale=mu_t)
        x_traj["ens_w"][t + 1] = x_traj["ens_w"][t] + cfg.dt * dx
        x_traj["ens_w"][t + 1, 0] = max(x_traj["ens_w"][t + 1, 0], 0.05)
        t0 = time.perf_counter()
        reg_w.add_observation(
            x_traj["ens_w"][t].astype(np.float32),
            u0.astype(np.float32),
            static.astype(np.float32),
            dx.astype(np.float32),
        )
        update_ms["ens_w"].append((time.perf_counter() - t0) * 1000.0)
        w_hist["ens_w"][t + 1] = reg_w.get_weights().copy()

        # 5) ens_w_topk rollout (Mode B + TopK projection)
        w_modeb_k = _project_topk(reg_wk.get_weights().copy(), cfg.top_k)
        u0, ms = _solve_step(solver_ens_w, x_traj["ens_w_topk"][t], float(vref[t]), w_modeb_k, lbx, ubx, horizon=cfg.horizon)
        solve_ms["ens_w_topk"].append(ms)
        u_traj["ens_w_topk"][t] = u0
        dx = accelerations_numpy(x_traj["ens_w_topk"][t], u0, static, orca, mu_scale=mu_t)
        x_traj["ens_w_topk"][t + 1] = x_traj["ens_w_topk"][t] + cfg.dt * dx
        x_traj["ens_w_topk"][t + 1, 0] = max(x_traj["ens_w_topk"][t + 1, 0], 0.05)
        t0 = time.perf_counter()
        reg_wk.add_observation(
            x_traj["ens_w_topk"][t].astype(np.float32),
            u0.astype(np.float32),
            static.astype(np.float32),
            dx.astype(np.float32),
        )
        update_ms["ens_w_topk"].append((time.perf_counter() - t0) * 1000.0)
        w_hist["ens_w_topk"][t + 1] = _project_topk(reg_wk.get_weights().copy(), cfg.top_k)

    # Metrics
    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def first_convergence_step(series: np.ndarray, target: float, start: int, tol: float = 0.05, hold: int = 5) -> Optional[int]:
        """
        Return first t>=start such that |series[t]-target| <= tol and remains within tol for `hold` steps.
        """
        series = np.asarray(series, dtype=np.float64).reshape(-1)
        for t in range(int(start), max(int(series.size - hold), int(start))):
            if np.all(np.abs(series[t : t + hold] - float(target)) <= float(tol)):
                return int(t)
        return None

    def first_weight_convergence_step(W: np.ndarray, start: int, eps: float = 1e-3, hold: int = 5) -> Optional[int]:
        W = np.asarray(W, dtype=np.float64)
        d = np.linalg.norm(W[1:] - W[:-1], axis=1)
        for t in range(int(start), max(int(d.size - hold), int(start))):
            if np.all(d[t : t + hold] <= float(eps)):
                return int(t)
        return None

    tracking = {}
    for k in names:
        tracking[k] = {
            "rmse_vx": rmse(x_traj[k][:, 0], vref),
            "rmse_vy": rmse(x_traj[k][:, 1], np.zeros_like(vref)),
            "rmse_omega": rmse(x_traj[k][:, 2], np.zeros_like(vref)),
        }

    # Plots (vx)
    fig = plt.figure(figsize=(10, 4))
    for k in names:
        plt.plot(x_traj[k][:, 0], label=k, linewidth=1)
    plt.plot(vref, "k--", label="vref", linewidth=2)
    plt.axvline(int(cfg.mu_change_step), color="k", linestyle=":", linewidth=1)
    plt.title(f"ORCA Closed-loop vx tracking ({tag})")
    plt.xlabel("t")
    plt.ylabel("m/s")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"orca_closedloop_vx_{tag}.png", dpi=150)
    plt.close(fig)

    # Plot mu schedule
    fig = plt.figure(figsize=(10, 2.5))
    plt.plot(mu_sched, label="mu_scale")
    plt.axvline(int(cfg.mu_change_step), color="k", linestyle=":", linewidth=1)
    plt.title(f"mu_scale schedule ({tag})")
    plt.xlabel("t")
    plt.tight_layout()
    plt.savefig(out_dir / f"orca_closedloop_mu_{tag}.png", dpi=150)
    plt.close(fig)

    # Estimator diagnostics / convergence
    mu_true_after = float(mu_sched[min(int(cfg.mu_change_step) + 1, cfg.T - 1)])
    mu_diag = {
        "mu_true_initial": float(mu_sched[0]),
        "mu_true_after_change": float(mu_true_after),
        "mu_change_step": int(cfg.mu_change_step),
    }
    mu_rls_conv = first_convergence_step(mu_est["phys_mu_rls"], target=mu_true_after, start=int(cfg.mu_change_step), tol=0.05, hold=5)
    mu_ukf_conv = first_convergence_step(mu_est["phys_mu_ukf"], target=mu_true_after, start=int(cfg.mu_change_step), tol=0.05, hold=5)
    w_conv = first_weight_convergence_step(w_hist["ens_w"], start=int(cfg.mu_change_step), eps=1e-3, hold=5)
    wk_conv = first_weight_convergence_step(w_hist["ens_w_topk"], start=int(cfg.mu_change_step), eps=1e-3, hold=5)

    return {
        "tag": tag,
        "mu_in": float(mu_in),
        "mu_out": float(mu_out),
        "tracking": tracking,
        "solve_ms": {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))} for k, v in solve_ms.items()},
        "physics_rebuild_ms": rebuild_ms,
        "mode_b_update_ms": {
            "ens_w": {"mean": float(np.mean(update_ms["ens_w"])), "p95": float(np.percentile(update_ms["ens_w"], 95)), "max": float(np.max(update_ms["ens_w"]))},
            "ens_w_topk": {"mean": float(np.mean(update_ms["ens_w_topk"])), "p95": float(np.percentile(update_ms["ens_w_topk"], 95)), "max": float(np.max(update_ms["ens_w_topk"]))},
        },
        "phys_mu_estimator_update_ms": {
            "phys_mu_rls": {"mean": float(np.mean(update_ms["phys_mu_rls"])), "p95": float(np.percentile(update_ms["phys_mu_rls"], 95)), "max": float(np.max(update_ms["phys_mu_rls"]))},
            "phys_mu_ukf": {"mean": float(np.mean(update_ms["phys_mu_ukf"])), "p95": float(np.percentile(update_ms["phys_mu_ukf"], 95)), "max": float(np.max(update_ms["phys_mu_ukf"]))},
        },
        "estimator_diagnostics": {
            **mu_diag,
            "phys_mu_rls": {
                "mu_est": mu_est["phys_mu_rls"].tolist(),
                "rls_denom": rls_denom.tolist(),
                "convergence_step_tol0p05_hold5": mu_rls_conv,
            },
            "phys_mu_ukf": {
                "mu_est": mu_est["phys_mu_ukf"].tolist(),
                "ukf_cond_S": ukf_condS.tolist(),
                "convergence_step_tol0p05_hold5": mu_ukf_conv,
            },
            "mode_b_weights": {
                "ens_w_convergence_step_dW_le_1e-3_hold5": w_conv,
                "ens_w_topk_convergence_step_dW_le_1e-3_hold5": wk_conv,
            },
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="./paper_orca_closedloop_results")
    p.add_argument("--lib_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--mu_change_step", type=int, default=150)
    p.add_argument("--out_multiplier", type=float, default=1.25)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--use_jit_for_baked", action="store_true")

    # Estimator knobs
    p.add_argument("--rls_lambda", type=float, default=0.98)
    p.add_argument("--rls_initial_P", type=float, default=100.0)
    p.add_argument("--rls_adaptation_window", type=int, default=50)
    p.add_argument("--ukf_alpha", type=float, default=1e-3)
    p.add_argument("--ukf_beta", type=float, default=2.0)
    p.add_argument("--ukf_kappa", type=float, default=0.0)
    p.add_argument("--ukf_process_noise", type=float, default=1e-4)
    p.add_argument("--ukf_measurement_noise", type=float, default=0.1)
    p.add_argument("--estimator_max_update_steps", type=int, default=None)

    # Failure mode switches
    p.add_argument("--failure_case", type=str, default="normal", choices=["normal", "no_excitation", "short_window", "extreme_mu"])
    p.add_argument("--modeb_window_size", type=int, default=40)
    p.add_argument("--modeb_regularization", type=float, default=1e-4)
    p.add_argument("--show_progress", action="store_true", help="Show a tqdm progress bar for each closed-loop rollout.")
    args = p.parse_args()

    # Failure-case convenience tweaks (keep explicit CLI overrides working)
    failure_case = str(args.failure_case).lower()
    if failure_case == "short_window" and args.estimator_max_update_steps is None:
        # "short adaptation window (<10 samples)" baseline for estimators
        args.estimator_max_update_steps = 8
    if failure_case == "extreme_mu":
        # Ensure mu_out > 2.0 given library mu_max ~ 1.4 (mu_out = out_multiplier * mu_max)
        args.out_multiplier = max(float(args.out_multiplier), 1.7)

    cfg = OrcaClosedLoopConfig(
        out_dir=args.out_dir,
        lib_dir=args.lib_dir,
        seed=int(args.seed),
        T=int(args.T),
        dt=float(args.dt),
        horizon=int(args.horizon),
        mu_change_step=int(args.mu_change_step),
        out_multiplier=float(args.out_multiplier),
        top_k=int(args.top_k),
        use_jit_for_baked=bool(args.use_jit_for_baked),
        rls_lambda=float(args.rls_lambda),
        rls_initial_P=float(args.rls_initial_P),
        rls_adaptation_window=int(args.rls_adaptation_window),
        ukf_alpha=float(args.ukf_alpha),
        ukf_beta=float(args.ukf_beta),
        ukf_kappa=float(args.ukf_kappa),
        ukf_process_noise=float(args.ukf_process_noise),
        ukf_measurement_noise=float(args.ukf_measurement_noise),
        estimator_max_update_steps=None if args.estimator_max_update_steps is None else int(args.estimator_max_update_steps),
        failure_case=str(args.failure_case),
        modeb_window_size=int(args.modeb_window_size),
        modeb_regularization=float(args.modeb_regularization),
        show_progress=bool(args.show_progress),
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # derive mu ranges from library
    lib = load_orca_library(cfg.lib_dir, device=torch.device("cpu"))
    mu_min, mu_max = float(np.min(lib.mu_scale_grid)), float(np.max(lib.mu_scale_grid))
    mu_in, mu_out = pick_in_out_mu(mu_min, mu_max, out_multiplier=float(cfg.out_multiplier))

    mu_sched_in = _make_mu_schedule(cfg, mu_in=mu_in, mu_out=mu_out, kind="in")
    mu_sched_out = _make_mu_schedule(cfg, mu_in=mu_in, mu_out=mu_out, kind="out")

    results = {"config": asdict(cfg)}
    results["closed_loop_in_manifold"] = run_closed_loop(cfg, mu_sched_in, out_dir, tag="in")
    results["closed_loop_out_of_manifold"] = run_closed_loop(cfg, mu_sched_out, out_dir, tag="out")

    (out_dir / "orca_closedloop_results.json").write_text(json.dumps(results, indent=2))
    print(f"[OK] wrote {out_dir / 'orca_closedloop_results.json'}")


if __name__ == "__main__":
    main()


