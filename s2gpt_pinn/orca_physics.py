#!/usr/bin/env python3
"""
ORCA-scale bicycle dynamics (BayesRace-consistent) for training/evaluation.

BayesRace Dynamic model uses a simplified Pacejka form (no E parameter):
  Ffy = Df * sin(Cf * atan(Bf * alpha_f))
  Fry = Dr * sin(Cr * atan(Br * alpha_r))

and drivetrain:
  Frx = (Cm1 - Cm2*vx)*pwm - Cr0 - Cr2*vx^2

State used by our surrogate:
  x_dyn = [vx, vy, omega]
Control used by our surrogate:
  u_dyn = [delta(steer), throttle(pwm)]
Static:
  s = [mass, Iz]

We treat "mu_scale" as scaling applied to Df and Dr to model friction changes:
  Df_eff = mu_scale * Df_nom
  Dr_eff = mu_scale * Dr_nom

This lets us build an "in-manifold" library over mu_scale in [mu_min, mu_max].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import casadi as cs


@dataclass
class OrcaParams:
    # geometry
    lf: float = 0.029
    lr: float = 0.033
    # mass properties
    mass: float = 0.041
    Iz: float = 27.8e-6
    # tire coefficients
    Bf: float = 2.579
    Cf: float = 1.2
    Df: float = 0.192
    Br: float = 3.3852
    Cr: float = 1.2691
    Dr: float = 0.1737
    # drivetrain
    Cm1: float = 0.287
    Cm2: float = 0.0545
    Cr0: float = 0.0518
    Cr2: float = 0.00035


def accelerations_numpy(
    state: np.ndarray,
    control: np.ndarray,
    static: np.ndarray,
    params: OrcaParams,
    mu_scale: float = 1.0,
) -> np.ndarray:
    vx = float(max(state[0], 0.05))
    vy = float(state[1])
    omega = float(state[2])
    delta = float(control[0])
    pwm = float(control[1])
    m = float(static[0])
    Iz = float(static[1])

    # slip angles
    alphaf = delta - np.arctan2((params.lf * omega + vy), abs(vx))
    alphar = np.arctan2((params.lr * omega - vy), abs(vx))

    # forces
    Df = mu_scale * params.Df
    Dr = mu_scale * params.Dr
    Ffy = Df * np.sin(params.Cf * np.arctan(params.Bf * alphaf))
    Fry = Dr * np.sin(params.Cr * np.arctan(params.Br * alphar))
    Frx = (params.Cm1 - params.Cm2 * vx) * pwm - params.Cr0 - params.Cr2 * (vx**2)

    dvx = (Frx - Ffy * np.sin(delta)) / m + vy * omega
    dvy = (Fry + Ffy * np.cos(delta)) / m - vx * omega
    domega = (Ffy * params.lf * np.cos(delta) - Fry * params.lr) / Iz

    return np.array([dvx, dvy, domega], dtype=np.float64)


def accelerations_torch(
    state: torch.Tensor,
    control: torch.Tensor,
    static: torch.Tensor,
    params: OrcaParams,
    mu_scale: float = 1.0,
) -> torch.Tensor:
    # state: [N,3], control:[N,2], static:[N,2]
    vx = torch.clamp(state[:, 0], min=0.05)
    vy = state[:, 1]
    omega = state[:, 2]
    delta = control[:, 0]
    pwm = control[:, 1]
    m = static[:, 0]
    Iz = static[:, 1]

    alphaf = delta - torch.atan2(params.lf * omega + vy, torch.abs(vx))
    alphar = torch.atan2(params.lr * omega - vy, torch.abs(vx))

    Df = mu_scale * params.Df
    Dr = mu_scale * params.Dr
    Ffy = Df * torch.sin(params.Cf * torch.atan(params.Bf * alphaf))
    Fry = Dr * torch.sin(params.Cr * torch.atan(params.Br * alphar))
    Frx = (params.Cm1 - params.Cm2 * vx) * pwm - params.Cr0 - params.Cr2 * (vx**2)

    dvx = (Frx - Ffy * torch.sin(delta)) / m + vy * omega
    dvy = (Fry + Ffy * torch.cos(delta)) / m - vx * omega
    domega = (Ffy * params.lf * torch.cos(delta) - Fry * params.lr) / Iz

    return torch.stack([dvx, dvy, domega], dim=1)


def accelerations_casadi(
    x: cs.SX,
    u: cs.SX,
    s: cs.SX,
    params: OrcaParams,
    mu_scale: cs.SX,
) -> cs.SX:
    vx = cs.fmax(x[0], 0.05)
    vy = x[1]
    omega = x[2]
    delta = u[0]
    pwm = u[1]
    m = s[0]
    Iz = s[1]

    alphaf = delta - cs.atan2(params.lf * omega + vy, cs.fabs(vx))
    alphar = cs.atan2(params.lr * omega - vy, cs.fabs(vx))

    Df = mu_scale * params.Df
    Dr = mu_scale * params.Dr
    Ffy = Df * cs.sin(params.Cf * cs.atan(params.Bf * alphaf))
    Fry = Dr * cs.sin(params.Cr * cs.atan(params.Br * alphar))
    Frx = (params.Cm1 - params.Cm2 * vx) * pwm - params.Cr0 - params.Cr2 * (vx**2)

    dvx = (Frx - Ffy * cs.sin(delta)) / m + vy * omega
    dvy = (Fry + Ffy * cs.cos(delta)) / m - vx * omega
    domega = (Ffy * params.lf * cs.cos(delta) - Fry * params.lr) / Iz
    return cs.vertcat(dvx, dvy, domega)


def build_orca_dynamics_sx_mu(
    name: str,
    static_params: np.ndarray,
    params: OrcaParams | None = None,
    mu_mode: str = "as_input",  # "as_input" or "baked_constant"
    mu_baked: float = 1.0,
) -> cs.Function:
    """
    Build CasADi SX dynamics for the ORCA accel model:
      x = [vx, vy, omega]
      u = [delta, pwm]
      s = [m, Iz] (passed as constant DM to match other code)

    Returns f(x,u,mu) if mu_mode="as_input" else f(x,u).
    """
    p = params or OrcaParams()
    x = cs.SX.sym("x", 3)
    u = cs.SX.sym("u", 2)
    s = cs.DM(np.asarray(static_params, dtype=np.float64).reshape(2))

    if mu_mode == "as_input":
        mu = cs.SX.sym("mu", 1)
        dx = accelerations_casadi(x, u, s, p, mu_scale=mu[0])
        return cs.Function(name, [x, u, mu], [dx], ["x", "u", "mu"], ["dx"])
    if mu_mode == "baked_constant":
        dx = accelerations_casadi(x, u, s, p, mu_scale=cs.DM(float(mu_baked)))
        return cs.Function(name, [x, u], [dx], ["x", "u"], ["dx"])
    raise ValueError(f"Unknown mu_mode={mu_mode}")


