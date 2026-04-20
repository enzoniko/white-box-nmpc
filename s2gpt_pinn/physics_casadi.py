#!/usr/bin/env python3
"""
Symbolic Physics Baseline (CasADi SX) for Dynamic Bicycle with Pacejka

This is the "symbolic physics remains fast" baseline used in the paper-style experiments.

State (3):
  x = [vx, vy, omega]
Control (2):
  u = [delta, throttle]
Static params (2):
  s = [m, Iz]

Dynamic parameters are represented by a minimal set used elsewhere in this repo:
  lf, lr, pacejka_{B,C,D,E}_{f,r}, cm1, cm2, cr0, cd

Two build modes:
  - mu as an input parameter (no rebuild needed for friction changes)
  - mu baked in as a constant (simulates "recompile on change" by rebuilding the NLP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import casadi as cs
import numpy as np


@dataclass
class PhysicsParams:
    # geometry
    lf: float = 1.5
    lr: float = 1.0
    # Pacejka
    pacejka_B_f: float = 12.0
    pacejka_C_f: float = 1.5
    pacejka_D_f: float = 1.0  # friction-like (can be overridden by mu)
    pacejka_E_f: float = 0.5
    pacejka_B_r: float = 12.0
    pacejka_C_r: float = 1.5
    pacejka_D_r: float = 1.0  # friction-like (can be overridden by mu)
    pacejka_E_r: float = 0.5
    # drivetrain / aero
    cm1: float = 2000.0
    cm2: float = 10.0
    cr0: float = 200.0
    cd: float = 0.5


def _pacejka(
    alpha: cs.SX,
    B: float,
    C: float,
    D: cs.SX,
    E: float,
    Fz: cs.SX,
) -> cs.SX:
    inner = B * alpha - E * (B * alpha - cs.atan(B * alpha))
    return D * Fz * cs.sin(C * cs.atan(inner))


def build_physics_dynamics_sx(
    params: PhysicsParams = PhysicsParams(),
    mu_mode: Literal["as_input", "baked_constant"] = "as_input",
    mu_baked: float = 0.9,
    name: str = "physics_bicycle",
) -> Tuple[cs.Function, dict]:
    """
    Build CasADi Function for bicycle accelerations.

    Returns (f, meta) where f is:
      - mu_mode=as_input:        f(x,u,s,mu) -> y
      - mu_mode=baked_constant:  f(x,u,s) -> y   (mu baked into D_f/D_r)
    """
    x = cs.SX.sym("x", 3)
    u = cs.SX.sym("u", 2)
    s = cs.SX.sym("s", 2)

    vx = cs.fmax(x[0], 0.1)
    vy = x[1]
    omega = x[2]
    delta = u[0]
    throttle = u[1]
    m = s[0]
    Iz = s[1]

    alpha_f = delta - cs.atan2(vy + params.lf * omega, vx)
    alpha_r = cs.atan2(params.lr * omega - vy, vx)

    Fz_f = m * 9.81 * params.lr / (params.lf + params.lr)
    Fz_r = m * 9.81 * params.lf / (params.lf + params.lr)

    if mu_mode == "as_input":
        mu = cs.SX.sym("mu", 1)
        Df = mu
        Dr = mu
        Ffy = _pacejka(alpha_f, params.pacejka_B_f, params.pacejka_C_f, Df, params.pacejka_E_f, Fz_f)
        Fry = _pacejka(alpha_r, params.pacejka_B_r, params.pacejka_C_r, Dr, params.pacejka_E_r, Fz_r)
        Frx = params.cm1 * throttle - params.cm2 * vx - params.cr0 - params.cd * vx * vx
        dvx = (Frx - Ffy * cs.sin(delta) + m * vy * omega) / m
        dvy = (Fry + Ffy * cs.cos(delta) - m * vx * omega) / m
        domega = (Ffy * params.lf * cs.cos(delta) - Fry * params.lr) / Iz
        y = cs.vertcat(dvx, dvy, domega)
        f = cs.Function(name, [x, u, s, mu], [y], ["x", "u", "s", "mu"], ["y"])
        return f, {"mu_mode": mu_mode}

    if mu_mode == "baked_constant":
        Df = cs.DM([mu_baked])[0]
        Dr = cs.DM([mu_baked])[0]
        Ffy = _pacejka(alpha_f, params.pacejka_B_f, params.pacejka_C_f, Df, params.pacejka_E_f, Fz_f)
        Fry = _pacejka(alpha_r, params.pacejka_B_r, params.pacejka_C_r, Dr, params.pacejka_E_r, Fz_r)
        Frx = params.cm1 * throttle - params.cm2 * vx - params.cr0 - params.cd * vx * vx
        dvx = (Frx - Ffy * cs.sin(delta) + m * vy * omega) / m
        dvy = (Fry + Ffy * cs.cos(delta) - m * vx * omega) / m
        domega = (Ffy * params.lf * cs.cos(delta) - Fry * params.lr) / Iz
        y = cs.vertcat(dvx, dvy, domega)
        f = cs.Function(name, [x, u, s], [y], ["x", "u", "s"], ["y"])
        return f, {"mu_mode": mu_mode, "mu_baked": mu_baked}

    raise ValueError(f"Unsupported mu_mode: {mu_mode}")


def build_physics_dynamics_sx_theta(
    params: PhysicsParams = PhysicsParams(),
    theta_mode: Literal["as_input", "baked_constant"] = "as_input",
    theta_baked: Optional[np.ndarray] = None,
    name: str = "physics_bicycle_theta",
) -> Tuple[cs.Function, dict]:
    """
    Same equations as `build_physics_dynamics_sx`, but exposes a SMALL dynamic-parameter vector:

      theta = [mu, cd, cm1]

    This supports experiments where more than friction changes over time.
    The rest of the parameters remain fixed in `params`.
    """
    x = cs.SX.sym("x", 3)
    u = cs.SX.sym("u", 2)
    s = cs.SX.sym("s", 2)

    vx = cs.fmax(x[0], 0.1)
    vy = x[1]
    omega = x[2]
    delta = u[0]
    throttle = u[1]
    m = s[0]
    Iz = s[1]

    alpha_f = delta - cs.atan2(vy + params.lf * omega, vx)
    alpha_r = cs.atan2(params.lr * omega - vy, vx)

    Fz_f = m * 9.81 * params.lr / (params.lf + params.lr)
    Fz_r = m * 9.81 * params.lf / (params.lf + params.lr)

    if theta_mode == "as_input":
        theta = cs.SX.sym("theta", 3)  # [mu, cd, cm1]
        mu = theta[0]
        cd = theta[1]
        cm1 = theta[2]

        Ffy = _pacejka(alpha_f, params.pacejka_B_f, params.pacejka_C_f, mu, params.pacejka_E_f, Fz_f)
        Fry = _pacejka(alpha_r, params.pacejka_B_r, params.pacejka_C_r, mu, params.pacejka_E_r, Fz_r)

        Frx = cm1 * throttle - params.cm2 * vx - params.cr0 - cd * vx * vx
        dvx = (Frx - Ffy * cs.sin(delta) + m * vy * omega) / m
        dvy = (Fry + Ffy * cs.cos(delta) - m * vx * omega) / m
        domega = (Ffy * params.lf * cs.cos(delta) - Fry * params.lr) / Iz
        y = cs.vertcat(dvx, dvy, domega)
        f = cs.Function(name, [x, u, s, theta], [y], ["x", "u", "s", "theta"], ["y"])
        return f, {"theta_mode": theta_mode}

    if theta_mode == "baked_constant":
        if theta_baked is None:
            theta_baked = np.array([0.9, params.cd, params.cm1], dtype=np.float64)
        theta_baked = np.asarray(theta_baked, dtype=np.float64).reshape(3)
        mu = cs.DM([theta_baked[0]])[0]
        cd = cs.DM([theta_baked[1]])[0]
        cm1 = cs.DM([theta_baked[2]])[0]

        Ffy = _pacejka(alpha_f, params.pacejka_B_f, params.pacejka_C_f, mu, params.pacejka_E_f, Fz_f)
        Fry = _pacejka(alpha_r, params.pacejka_B_r, params.pacejka_C_r, mu, params.pacejka_E_r, Fz_r)

        Frx = cm1 * throttle - params.cm2 * vx - params.cr0 - cd * vx * vx
        dvx = (Frx - Ffy * cs.sin(delta) + m * vy * omega) / m
        dvy = (Fry + Ffy * cs.cos(delta) - m * vx * omega) / m
        domega = (Ffy * params.lf * cs.cos(delta) - Fry * params.lr) / Iz
        y = cs.vertcat(dvx, dvy, domega)
        f = cs.Function(name, [x, u, s], [y], ["x", "u", "s"], ["y"])
        return f, {"theta_mode": theta_mode, "theta_baked": theta_baked.tolist()}

    raise ValueError(f"Unsupported theta_mode: {theta_mode}")


