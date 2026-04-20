#!/usr/bin/env python3
"""
Trajectory Scenarios for Paper-Style Experiments

We need TIME-CONSISTENT data (trajectories), not just random points.

This module provides:
- Reference trajectory generation (speed profiles)
- Friction schedules (step / ramp)
- Simple rollout using a chosen "true" dynamics model (physics oracle)

The goal is to support:
  - tracking RMSE under sudden friction changes
  - adaptation latency measurements (Mode B weights update)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Optional

import numpy as np


@dataclass
class ScenarioConfig:
    dt: float = 0.02
    T: int = 300  # steps
    vref_low: float = 18.0
    vref_high: float = 28.0
    mu_high: float = 1.0
    mu_low: float = 0.5
    mu_change_step: int = 120
    mu_ramp_len: int = 80
    seed: int = 42


def make_vref_profile(cfg: ScenarioConfig, kind: Literal["constant", "chirp", "piecewise"] = "piecewise") -> np.ndarray:
    t = np.arange(cfg.T) * cfg.dt
    if kind == "constant":
        return np.ones(cfg.T) * ((cfg.vref_low + cfg.vref_high) / 2)
    if kind == "chirp":
        return (cfg.vref_low + cfg.vref_high) / 2 + 4.0 * np.sin(2 * np.pi * (0.2 + 0.8 * t / (cfg.T * cfg.dt)) * t)
    # piecewise
    v = np.ones(cfg.T) * cfg.vref_low
    v[cfg.T // 3: 2 * cfg.T // 3] = cfg.vref_high
    v[2 * cfg.T // 3:] = (cfg.vref_low + cfg.vref_high) / 2
    return v


def make_mu_schedule(cfg: ScenarioConfig, kind: Literal["step", "ramp"] = "step") -> np.ndarray:
    mu = np.ones(cfg.T) * cfg.mu_high
    if kind == "step":
        mu[cfg.mu_change_step:] = cfg.mu_low
        return mu
    # ramp
    start = cfg.mu_change_step
    end = min(cfg.T, start + cfg.mu_ramp_len)
    mu[start:end] = np.linspace(cfg.mu_high, cfg.mu_low, end - start)
    mu[end:] = cfg.mu_low
    return mu


def make_theta_schedule(
    cfg: ScenarioConfig,
    kind: Literal["step", "ramp"] = "step",
    cd_high: float = 0.5,
    cd_low: float = 0.8,
    cm1_high: float = 2000.0,
    cm1_low: float = 1700.0,
) -> np.ndarray:
    """
    Build a time-varying dynamic-parameter schedule:

      theta[t] = [mu[t], cd[t], cm1[t]]

    This matches `build_physics_dynamics_sx_theta`.
    """
    mu = make_mu_schedule(cfg, kind=kind)
    if kind == "step":
        cd = np.ones(cfg.T) * cd_high
        cm1 = np.ones(cfg.T) * cm1_high
        cd[cfg.mu_change_step:] = cd_low
        cm1[cfg.mu_change_step:] = cm1_low
    else:
        # ramp
        cd = np.ones(cfg.T) * cd_high
        cm1 = np.ones(cfg.T) * cm1_high
        start = cfg.mu_change_step
        end = min(cfg.T, start + cfg.mu_ramp_len)
        cd[start:end] = np.linspace(cd_high, cd_low, end - start)
        cm1[start:end] = np.linspace(cm1_high, cm1_low, end - start)
        cd[end:] = cd_low
        cm1[end:] = cm1_low

    theta = np.column_stack([mu, cd, cm1]).astype(np.float64)
    return theta


def generate_open_loop_controls(cfg: ScenarioConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a time-consistent control "excitation" used for rollouts.

    We keep it simple and reproducible:
    - delta: small sinusoid (steering excitation)
    - throttle: PI-like around vref is done in controllers; here we just provide a baseline
    """
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(cfg.T) * cfg.dt
    delta = 0.08 * np.sin(2 * np.pi * 0.8 * t) + 0.02 * np.sin(2 * np.pi * 2.5 * t)
    delta = np.clip(delta, -0.5, 0.5)
    throttle = 0.3 + 0.15 * np.sin(2 * np.pi * 0.4 * t) + 0.05 * rng.standard_normal(cfg.T)
    throttle = np.clip(throttle, -0.5, 0.8)
    return delta, throttle


def rollout_true_dynamics(
    f_true,  # callable: (x,u,static,mu) -> dxdt
    cfg: ScenarioConfig,
    static_params: np.ndarray,
    mu_schedule: np.ndarray,
    u_delta: np.ndarray,
    u_throttle: np.ndarray,
    x0: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Roll out a 3-state bicycle dynamics under a provided control sequence.

    State: [vx, vy, omega]
    """
    x = np.zeros((cfg.T, 3), dtype=np.float64)
    dx = np.zeros((cfg.T, 3), dtype=np.float64)
    u = np.column_stack([u_delta, u_throttle]).astype(np.float64)

    if x0 is None:
        x0 = np.array([20.0, 0.0, 0.0], dtype=np.float64)
    x[0] = x0

    for t in range(cfg.T - 1):
        mu = float(mu_schedule[t])
        dx[t] = f_true(x[t], u[t], static_params, mu)
        x[t + 1] = x[t] + cfg.dt * dx[t]
        # ensure vx stays positive-ish
        x[t + 1, 0] = max(x[t + 1, 0], 0.1)

    dx[-1] = f_true(x[-1], u[-1], static_params, float(mu_schedule[-1]))

    return {"x": x, "dx": dx, "u": u, "mu": mu_schedule}


