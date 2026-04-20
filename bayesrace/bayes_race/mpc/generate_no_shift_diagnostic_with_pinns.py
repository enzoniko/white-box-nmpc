#!/usr/bin/env python3
"""
generate_no_shift_diagnostic_with_pinns.py

Standalone script to generate the "No-Shift Diagnostic" figure (4-scenario comparison:
Baseline/Adaptive × Shift/NoShift) using:
  1) Idealized ODE specialists (create_ode_specialists)
  2) Neural Network (PINN) specialists (train_pinn_specialists)

Outputs:
  - no_shift_diagnostic_ODE.png
  - no_shift_diagnostic_ODE_noise.png   (ODE + Gaussian noise, aleatoric-only surrogate)
  - no_shift_diagnostic_PINN.png
  - no_shift_diagnostic_metrics.log     (RMSE, degradation %, position errors; all phases)
  - no_shift_diagnostic_metrics.json    (same metrics in JSON for tables)

Usage:
  python3 generate_no_shift_diagnostic_with_pinns.py
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import casadi as cs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Any, Optional

from tqdm import tqdm

# PyTorch for PINN specialists
import torch
import torch.nn as nn

# --- BayesRace ---
from bayes_race.params import ORCA
from bayes_race.models import Dynamic
from bayes_race.models.model import Model
from bayes_race.tracks import ETHZ
from bayes_race.mpc.nmpc_adaptive import setupNLPAdaptive


# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------

def ConstantSpeedFixed(x0, v0, track, N, Ts, projidx):
    """
    Generates a reference trajectory at EXACTLY v0, ignoring the track's
    internal speed profile.
    """
    raceline = track.raceline
    xy, idx = track.project_fast(x=x0[0], y=x0[1], raceline=raceline[:, projidx : projidx + 20])
    projidx = idx + projidx

    xref = np.zeros([2, N + 1])

    if hasattr(track, "spline") and hasattr(track.spline, "s"):
        if projidx >= len(track.spline.s):
            projidx = projidx % len(track.spline.s)
        current_s = track.spline.s[projidx]
        track_length = track.spline.s[-1]
    else:
        dist0 = np.sum(np.linalg.norm(np.diff(raceline[:, : projidx + 1], axis=1), 2, axis=0))
        current_s = dist0
        track_length = np.sum(np.linalg.norm(np.diff(raceline, axis=1), 2, axis=0))

    if hasattr(track, "spline") and hasattr(track.spline, "calc_position"):
        pos = track.spline.calc_position(current_s)
        xref[0, 0], xref[1, 0] = pos[0], pos[1]
    else:
        dists = np.cumsum(np.linalg.norm(np.diff(raceline, axis=1), 2, axis=0))
        closest_idx = np.argmin(np.abs(dists - current_s))
        xref[0, 0], xref[1, 0] = raceline[0, closest_idx], raceline[1, closest_idx]

    for idh in range(1, N + 1):
        current_s += v0 * Ts
        current_s = current_s % track_length
        if hasattr(track, "spline") and hasattr(track.spline, "calc_position"):
            pos = track.spline.calc_position(current_s)
            xref[0, idh], xref[1, idh] = pos[0], pos[1]
        else:
            dists = np.cumsum(np.linalg.norm(np.diff(raceline, axis=1), 2, axis=0))
            closest_idx = np.argmin(np.abs(dists - current_s))
            xref[0, idh], xref[1, idh] = raceline[0, closest_idx], raceline[1, closest_idx]

    return xref, projidx


# -----------------------------------------------------------------------------
# Governor classes
# -----------------------------------------------------------------------------

class Governor:
    """
    Governor module that estimates mixing weights by solving a simplex-constrained
    linear regression over a sliding window of recent measurements.
    """

    def __init__(self, n_specialists: int, window_size: int = 10, regularization: float = 1e-4):
        self.n_specialists = n_specialists
        self.window_size = window_size
        self.regularization = regularization
        self.history: deque = deque(maxlen=window_size)
        self.weights = np.ones(n_specialists) / n_specialists

    def add_measurement(self, x_k: np.ndarray, x_k_minus_1: np.ndarray, u_k_minus_1: np.ndarray) -> None:
        self.history.append((x_k.copy(), x_k_minus_1.copy(), u_k_minus_1.copy()))

    def update_weights(self, specialists: List[Any], dt: float) -> np.ndarray:
        if len(self.history) < 2:
            return self.weights
        n_obs = len(self.history)
        n_dims = 3
        Phi = np.zeros((n_obs * n_dims, self.n_specialists))
        y = np.zeros(n_obs * n_dims)
        for i, (x_k, x_k_minus_1, u_k_minus_1) in enumerate(self.history):
            dx_meas = (x_k[3:6] - x_k_minus_1[3:6]) / dt
            for j, specialist in enumerate(specialists):
                dx_pred = specialist._diffequation(None, x_k_minus_1, u_k_minus_1)
                Phi[i * n_dims : (i + 1) * n_dims, j] = dx_pred[3:6]
            y[i * n_dims : (i + 1) * n_dims] = dx_meas
        def objective(w: np.ndarray) -> float:
            return float(np.sum((Phi @ w - y) ** 2) + self.regularization * np.sum(w ** 2))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * self.n_specialists
        try:
            result = minimize(
                objective, self.weights.copy(), method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            if result.success:
                raw = np.maximum(result.x, 0.0)
                self.weights = raw / (np.sum(raw) + 1e-10)
        except Exception:
            pass
        return self.weights

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()


class GovernorEMA:
    """
    Governor with Exponential Moving Average smoothing for stable adaptation.
    """

    def __init__(
        self,
        n_specialists: int,
        window_size: int = 10,
        alpha: float = 0.2,
        regularization: float = 1e-3,
    ):
        self.n_specialists = n_specialists
        self.window_size = window_size
        self.alpha = alpha
        self.regularization = regularization
        self.history: deque = deque(maxlen=window_size)
        self.weights = np.ones(n_specialists) / n_specialists

    def add_measurement(self, x_k: np.ndarray, x_k_minus_1: np.ndarray, u_k_minus_1: np.ndarray) -> None:
        self.history.append((x_k.copy(), x_k_minus_1.copy(), u_k_minus_1.copy()))

    def update_weights(self, specialists: List[Any], dt: float) -> np.ndarray:
        if len(self.history) < 3:
            return self.weights
        n_obs = len(self.history)
        n_dims = 3
        Phi = np.zeros((n_obs * n_dims, self.n_specialists))
        y = np.zeros(n_obs * n_dims)
        for i, (x_k, x_k_minus_1, u_k_minus_1) in enumerate(self.history):
            dx_meas = (x_k[3:6] - x_k_minus_1[3:6]) / dt
            y[i * n_dims : (i + 1) * n_dims] = dx_meas
            for j, specialist in enumerate(specialists):
                dx_pred = specialist._diffequation(None, x_k_minus_1, u_k_minus_1)
                Phi[i * n_dims : (i + 1) * n_dims, j] = dx_pred[3:6]
        def objective(w: np.ndarray) -> float:
            return float(np.sum((Phi @ w - y) ** 2) + self.regularization * np.sum(w ** 2))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * self.n_specialists
        try:
            result = minimize(
                objective, self.weights, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            if result.success:
                raw = np.maximum(result.x, 0.0)
                raw = raw / (np.sum(raw) + 1e-10)
                self.weights = (1 - self.alpha) * self.weights + self.alpha * raw
        except Exception:
            pass
        return self.weights

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()


# -----------------------------------------------------------------------------
# Adaptive model (weighted ensemble of specialists)
# -----------------------------------------------------------------------------

class AdaptiveModel(Model):
    """Adaptive Model combining multiple specialists via weighted ensemble."""

    def __init__(self, specialists: List[Any], use_dyn_noise: bool = False, horizon: Optional[int] = None):
        super().__init__()
        self.n_states = 6
        self.n_inputs = 2
        self.specialists = specialists
        self.n_specialists = len(specialists)
        self.use_dyn_noise = bool(use_dyn_noise)
        self.horizon = int(horizon) if horizon is not None else None
        self._build_casadi()

    def _build_casadi(self) -> None:
        x = cs.SX.sym("x", 6)
        u = cs.SX.sym("u", 2)
        w = cs.SX.sym("w", self.n_specialists)
        dx_specialists = []
        for specialist in self.specialists:
            dx = specialist.casadi(x, u, cs.SX.zeros(6))
            dx_specialists.append(dx)
        accel_combined = cs.SX.zeros(3)
        for i, dx in enumerate(dx_specialists):
            accel_combined += w[i] * dx[3:6]
        dxdt_eq = cs.vertcat(dx_specialists[0][0:3], accel_combined)
        self._f_comb = cs.Function("f_comb", [x, u, w], [dxdt_eq])

    def casadi(self, x, u, dxdt, dyn_params=None, step_index=None):
        if dyn_params is None:
            w = cs.SX.ones(self.n_specialists) / self.n_specialists
        else:
            if isinstance(dyn_params, (cs.SX, cs.MX, cs.DM)):
                w = dyn_params[: self.n_specialists]
                w = cs.fmax(w, 0.0)
                w = w / (cs.sum1(w) + 1e-10)
            else:
                w = np.asarray(dyn_params).flatten()[: self.n_specialists]
                w = np.maximum(w, 0.0)
                w = w / (np.sum(w) + 1e-10)
        res = self._f_comb(x, u, w)
        dxdt[0], dxdt[1], dxdt[2] = res[0], res[1], res[2]
        dxdt[3], dxdt[4], dxdt[5] = res[3], res[4], res[5]
        if self.use_dyn_noise and self.horizon is not None and step_index is not None:
            n = self.n_specialists
            if isinstance(dyn_params, (cs.SX, cs.MX, cs.DM)):
                noise = dyn_params[n + 6 * step_index : n + 6 * (step_index + 1)]
                for i in range(6):
                    dxdt[i] = dxdt[i] + noise[i, 0]
            else:
                arr = np.asarray(dyn_params).flatten()
                if len(arr) >= n + 6 * self.horizon:
                    noise = arr[n + 6 * step_index : n + 6 * (step_index + 1)]
                    dxdt[0], dxdt[1], dxdt[2] = dxdt[0] + noise[0], dxdt[1] + noise[1], dxdt[2] + noise[2]
                    dxdt[3], dxdt[4], dxdt[5] = dxdt[3] + noise[3], dxdt[4] + noise[4], dxdt[5] + noise[5]
        return dxdt


# -----------------------------------------------------------------------------
# ODE specialists
# -----------------------------------------------------------------------------

def create_ode_specialists(base_params: Dict[str, Any], n_specialists: int = 8) -> Tuple[List[Dynamic], List[Tuple]]:
    """
    Create idealized ODE specialists over the same regimes as the ablation study.
    Returns (specialists, configs) with configs = (f_scale, m_scale, s_scale, d_scale, desc).
    """
    specialists: List[Dynamic] = []
    configs: List[Tuple] = []

    regimes = [
        (1.2, 0.95, 1.1, 0.9, "Optimal: Warm slicks, light"),
        (1.1, 1.0, 1.05, 0.95, "Good: Dry, nominal"),
        (1.0, 1.0, 1.0, 1.0, "Nominal: Baseline"),
        (0.9, 1.05, 0.95, 1.05, "Mild wet"),
        (0.7, 1.1, 0.85, 1.15, "Wet: Heavy rain"),
        (0.6, 1.15, 0.75, 1.25, "Very wet"),
        (0.5, 1.1, 0.7, 1.3, "Ice: Low friction"),
        (0.4, 1.2, 0.6, 1.4, "Extreme: Ice + load"),
    ]

    for i, (f_scale, m_scale, s_scale, d_scale, desc) in enumerate(regimes):
        params = base_params.copy()
        params["Df"] = params["Df"] * f_scale
        params["Dr"] = params["Dr"] * f_scale
        params["mass"] = params["mass"] * m_scale
        params["Iz"] = params["Iz"] * m_scale * 1.1
        params["Cf"] = params["Cf"] * s_scale
        params["Cr"] = params["Cr"] * s_scale
        params["Bf"] = params["Bf"] * (0.8 + 0.4 * s_scale)
        params["Br"] = params["Br"] * (0.8 + 0.4 * s_scale)
        params["Cr0"] = params["Cr0"] * d_scale
        params["Cr2"] = params["Cr2"] * d_scale
        specialists.append(Dynamic(**params))
        configs.append((f_scale, m_scale, s_scale, d_scale, desc))
        print(f"  ODE Specialist {i}: {desc}")

    return specialists, configs


# -----------------------------------------------------------------------------
# ODE specialists with Gaussian noise (aleatoric-only surrogate)
# -----------------------------------------------------------------------------


class NoisyODESpecialist:
    """
    Wraps a Dynamic (ODE) specialist and adds i.i.d. Gaussian noise to the
    predicted state derivative in _diffequation. Used to isolate aleatoric
    noise effects: NNs have both epistemic and aleatoric; ODE+noise has
    only aleatoric (no model uncertainty). CasADi path stays deterministic
    so the NMPC optimization remains well-defined.
    """

    def __init__(self, ode_specialist: Dynamic, noise_std: float = 0.05, rng: Optional[np.random.Generator] = None):
        self._ode = ode_specialist
        self.noise_std = float(noise_std)
        self._rng = rng if rng is not None else np.random.default_rng()

    def casadi(self, x, u, dxdt, *_):
        """Deterministic: delegate to inner ODE (no noise in NMPC)."""
        return self._ode.casadi(x, u, dxdt)

    def _diffequation(self, t: Optional[float], x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """NumPy path: ODE prediction + Gaussian noise (Governor sees aleatoric noise)."""
        dx = self._ode._diffequation(t, x, u)
        noise = self._rng.normal(0, self.noise_std, size=dx.shape)
        return np.asarray(dx, dtype=np.float64) + noise


def create_ode_specialists_with_noise(
    base_params: Dict[str, Any],
    n_specialists: int = 8,
    noise_std: float = 0.05,
    seed: int = 42,
) -> Tuple[List[NoisyODESpecialist], List[Tuple]]:
    """
    Create ODE specialists with added Gaussian noise on predictions (aleatoric-only).
    Same regimes as create_ode_specialists; each specialist wraps an ODE and adds
    i.i.d. N(0, noise_std^2) to dxdt in _diffequation. Returns (specialists, configs).
    """
    specialists_ode, configs = create_ode_specialists(base_params, n_specialists=n_specialists)
    rng = np.random.default_rng(seed)
    specialists: List[NoisyODESpecialist] = [
        NoisyODESpecialist(ode, noise_std=noise_std, rng=np.random.default_rng(seed + i))
        for i, ode in enumerate(specialists_ode)
    ]
    print(f"  ODE+noise specialists: 8 regimes, noise_std={noise_std}")
    return specialists, configs


def apply_regime_shift(params: Dict[str, Any], config: Dict[str, float]) -> Dict[str, Any]:
    """Apply regime shift scales to base params."""
    out = params.copy()
    if "friction" in config:
        out["Df"] = out["Df"] * config["friction"]
        out["Dr"] = out["Dr"] * config["friction"]
    if "mass" in config:
        out["mass"] = out["mass"] * config["mass"]
        out["Iz"] = out["Iz"] * config["mass"] * 1.1
    if "stiffness" in config:
        out["Cf"] = out["Cf"] * config["stiffness"]
        out["Cr"] = out["Cr"] * config["stiffness"]
        out["Bf"] = out["Bf"] * (0.8 + 0.4 * config["stiffness"])
        out["Br"] = out["Br"] * (0.8 + 0.4 * config["stiffness"])
    if "drag" in config:
        out["Cr0"] = out["Cr0"] * config["drag"]
        out["Cr2"] = out["Cr2"] * config["drag"]
    return out


# -----------------------------------------------------------------------------
# Metrics and simulation
# -----------------------------------------------------------------------------

def calculate_metrics(
    log: Dict[str, Any],
    change_time: float,
    label: str,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute RMSE for vx, vy, psi, and position; split Pre/Post event."""
    t = np.array(log["t"])
    vx = np.array(log["vx"])
    vy = np.array(log["vy"])
    psi = np.array(log["psi"])
    x_arr = np.array(log["x"])
    y_arr = np.array(log["y"])
    ref_x_arr = np.array(log["ref_x"])
    ref_y_arr = np.array(log["ref_y"])
    mask_pre = t <= change_time
    mask_post = t > change_time
    ref_vx, ref_vy = 1.5, 0.0
    ref_psi = psi[0] if len(psi) > 0 else 0.0
    position_error = np.sqrt((x_arr - ref_x_arr) ** 2 + (y_arr - ref_y_arr) ** 2)
    metrics = {
        "rmse_vx_pre": np.sqrt(np.mean((vx[mask_pre] - ref_vx) ** 2)),
        "rmse_vy_pre": np.sqrt(np.mean((vy[mask_pre] - ref_vy) ** 2)),
        "rmse_psi_pre": np.sqrt(np.mean((psi[mask_pre] - ref_psi) ** 2)),
        "rmse_vx_post": np.sqrt(np.mean((vx[mask_post] - ref_vx) ** 2)),
        "rmse_vy_post": np.sqrt(np.mean((vy[mask_post] - ref_vy) ** 2)),
        "rmse_psi_post": np.sqrt(np.mean((psi[mask_post] - ref_psi) ** 2)),
        "rmse_position_pre": np.sqrt(np.mean(position_error[mask_pre] ** 2)),
        "rmse_position_post": np.sqrt(np.mean(position_error[mask_post] ** 2)),
    }
    if verbose:
        print(f"\n--- {label} Metrics ---")
        print(f"Pre-Event  (t <= {change_time:.1f}s): RMSE vx {metrics['rmse_vx_pre']:.4f}, vy {metrics['rmse_vy_pre']:.4f}, pos {metrics['rmse_position_pre']:.4f}")
        print(f"Post-Event (t > {change_time:.1f}s): RMSE vx {metrics['rmse_vx_post']:.4f}, vy {metrics['rmse_vy_post']:.4f}, pos {metrics['rmse_position_post']:.4f}")
    return metrics


def _add_degradation(metrics: Dict[str, float]) -> Dict[str, float]:
    """Add degradation (delta and percent) from pre to post for each RMSE."""
    out = dict(metrics)
    for key in ("vx", "vy", "psi", "position"):
        pre = metrics.get(f"rmse_{key}_pre") or 0.0
        post = metrics.get(f"rmse_{key}_post") or 0.0
        out[f"degradation_{key}_delta"] = post - pre
        out[f"degradation_{key}_pct"] = (100.0 * (post - pre) / pre) if pre >= 1e-10 else 0.0
    return out


def compute_diagnostic_metrics_table(
    data_list: List[Tuple[Dict, Dict, Dict, Dict, str]],
    change_time: float = 2.2,
) -> List[Dict[str, Any]]:
    """
    From data_list (log_A, log_B, log_C, log_D, regime_name) per regime, compute
    full metrics (RMSE pre/post, position errors, degradation %) for methods A,B,C,D.
    Returns list of dicts: [ {"regime": name, "A": metrics_A, "B": ..., "C": ..., "D": ... }, ... ]
    """
    table: List[Dict[str, Any]] = []
    method_logs = ("A", "B", "C", "D")
    method_labels = ("Baseline+Shift", "Adaptive+Shift", "Baseline+NoShift", "Adaptive+NoShift")
    for log_A, log_B, log_C, log_D, regime_name in data_list:
        row: Dict[str, Any] = {"regime": regime_name}
        for key, log in zip(method_logs, (log_A, log_B, log_C, log_D)):
            m = calculate_metrics(log, change_time, f"{key}", verbose=False)
            row[key] = _add_degradation(m)
        table.append(row)
    return table


def format_metrics_report(
    phase_name: str,
    table: List[Dict[str, Any]],
    change_time: float = 2.2,
) -> Tuple[str, Dict[str, Any]]:
    """Format metrics table as human-readable text and as a JSON-serializable dict."""
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"DIAGNOSTIC PHASE: {phase_name}")
    lines.append("=" * 80)
    json_out: Dict[str, Any] = {"phase": phase_name, "change_time_s": change_time, "regimes": []}
    for row in table:
        regime = row["regime"]
        lines.append("")
        lines.append(f"--- Regime: {regime} ---")
        json_regime: Dict[str, Any] = {"regime": regime, "methods": {}}
        for method in ("A", "B", "C", "D"):
            label = {"A": "Baseline+Shift", "B": "Adaptive+Shift", "C": "Baseline+NoShift", "D": "Adaptive+NoShift"}[method]
            m = row[method]
            lines.append(f"  [{method}] {label}")
            lines.append(f"      Pre  (t<={change_time}s): RMSE vx={m['rmse_vx_pre']:.4f}, vy={m['rmse_vy_pre']:.4f}, psi={m['rmse_psi_pre']:.4f}, position={m['rmse_position_pre']:.4f}")
            lines.append(f"      Post (t>{change_time}s): RMSE vx={m['rmse_vx_post']:.4f}, vy={m['rmse_vy_post']:.4f}, psi={m['rmse_psi_post']:.4f}, position={m['rmse_position_post']:.4f}")
            lines.append(f"      Degradation (post-pre): vx {m['degradation_vx_delta']:+.4f} ({m['degradation_vx_pct']:+.1f}%), vy {m['degradation_vy_delta']:+.4f} ({m['degradation_vy_pct']:+.1f}%), position {m['degradation_position_delta']:+.4f} ({m['degradation_position_pct']:+.1f}%)")
            json_regime["methods"][method] = {
                "label": label,
                "rmse_vx_pre": float(m["rmse_vx_pre"]), "rmse_vy_pre": float(m["rmse_vy_pre"]),
                "rmse_psi_pre": float(m["rmse_psi_pre"]), "rmse_position_pre": float(m["rmse_position_pre"]),
                "rmse_vx_post": float(m["rmse_vx_post"]), "rmse_vy_post": float(m["rmse_vy_post"]),
                "rmse_psi_post": float(m["rmse_psi_post"]), "rmse_position_post": float(m["rmse_position_post"]),
                "degradation_vx_delta": float(m["degradation_vx_delta"]), "degradation_vx_pct": float(m["degradation_vx_pct"]),
                "degradation_vy_delta": float(m["degradation_vy_delta"]), "degradation_vy_pct": float(m["degradation_vy_pct"]),
                "degradation_psi_delta": float(m["degradation_psi_delta"]), "degradation_psi_pct": float(m["degradation_psi_pct"]),
                "degradation_position_delta": float(m["degradation_position_delta"]), "degradation_position_pct": float(m["degradation_position_pct"]),
            }
        json_out["regimes"].append(json_regime)
    return "\n".join(lines), json_out


def run_simulation_ablation(
    label: str,
    specialists: List[Any],
    friction_scales: np.ndarray,
    params_nominal: Dict[str, Any],
    params_shifted: Dict[str, Any],
    adaptive_mode: bool = True,
    no_shift: bool = False,
    use_dyn_noise: bool = False,
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Run one ablation simulation. no_shift=True keeps plant nominal (no regime shift).
    use_dyn_noise=True contaminates the NMPC surrogate dynamics with i.i.d. Gaussian
    noise at each prediction step (aleatoric-only); noise_std and rng control the noise.
    """
    T_SIM, DT = 5.0, 0.02
    N_STEPS = int(T_SIM / DT)
    CHANGE_TIME, CHANGE_STEP = 2.2, int(2.2 / DT)
    V_REF, HORIZON = 1.5, 15
    Q = np.diag([10.0, 10.0])
    P = np.diag([0.0, 0.0])
    R = np.diag([0.1, 1.0])
    track = ETHZ(reference="optimal", longer=True)
    model_plant = Dynamic(**params_nominal)
    n_spec = len(specialists)
    model_surrogate = AdaptiveModel(
        specialists, use_dyn_noise=use_dyn_noise, horizon=HORIZON if use_dyn_noise else None
    )
    dyn_dim = n_spec + 6 * HORIZON if use_dyn_noise else n_spec
    nlp = setupNLPAdaptive(
        HORIZON, DT, Q, P, R,
        params_nominal, model_surrogate, track,
        dyn_dim=dyn_dim, track_cons=False,
    )
    _rng = rng if rng is not None else np.random.default_rng()
    x0 = np.zeros(6)
    x0[0], x0[1] = track.x_init, track.y_init
    x0[2], x0[3] = track.psi_init, V_REF
    u_prev = np.zeros((2, 1))
    projidx = 0
    idx_nom = np.argmin(np.abs(friction_scales - 1.0))
    if adaptive_mode:
        weights = np.ones(len(specialists)) / len(specialists)
        governor = GovernorEMA(
            n_specialists=len(specialists), window_size=20, alpha=0.1, regularization=1e-3
        )
    else:
        weights = np.zeros(len(specialists))
        weights[idx_nom] = 1.0
        governor = None

    log: Dict[str, Any] = {"t": [], "vx": [], "vy": [], "psi": [], "w": [], "ref_vx": [], "x": [], "y": [], "ref_x": [], "ref_y": []}

    for k in tqdm(range(N_STEPS), desc=label, unit="step", leave=False):
        t_curr = k * DT
        if not no_shift and k == CHANGE_STEP:
            model_plant = Dynamic(**params_shifted)
        xref, projidx = ConstantSpeedFixed(x0=x0[:2], v0=V_REF, track=track, N=HORIZON, Ts=DT, projidx=projidx)
        try:
            if use_dyn_noise:
                noise_samples = _rng.normal(0, noise_std, size=(6 * HORIZON,))
                dyn_vec = np.concatenate([weights.ravel(), noise_samples]).reshape(-1, 1)
            else:
                dyn_vec = weights.reshape(-1, 1)
            u_opt, _, _ = nlp.solve(x0, xref[:2, :], u_prev, dyn=dyn_vec)
            u_apply = u_opt[:, 0].reshape(2, 1)
        except RuntimeError:
            u_apply = np.array([[-0.5], [0]])
        x_traj, _ = model_plant.sim_continuous(x0, u_apply, [0, DT])
        x_next = x_traj[:, -1]
        if adaptive_mode and governor is not None:
            governor.add_measurement(x_next, x0, u_apply.flatten())
            if k > 2:
                weights = governor.update_weights(specialists, DT)
        log["t"].append(t_curr)
        log["vx"].append(x0[3])
        log["vy"].append(x0[4])
        log["psi"].append(x0[2])
        log["w"].append(weights.copy())
        log["ref_vx"].append(V_REF)
        log["x"].append(x0[0])
        log["y"].append(x0[1])
        log["ref_x"].append(xref[0, 0])
        log["ref_y"].append(xref[1, 0])
        x0, u_prev = x_next, u_apply

    return log


# -----------------------------------------------------------------------------
# Neural Specialist (PINN) — CasADi symbolic + NumPy forward
# -----------------------------------------------------------------------------

class NeuralSpecialist:
    """
    Wraps a trained MLP as a specialist: CasADi symbolic forward for the NMPC solver
    and NumPy forward for the Governor's _diffequation.
    Architecture: 3 hidden layers — Linear(8,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,6).
    """

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        input_mean: np.ndarray,
        input_std: np.ndarray,
        output_mean: np.ndarray,
        output_std: np.ndarray,
        hidden_sizes: Tuple[int, ...] = (64, 64, 64),
    ):
        self.input_mean = np.asarray(input_mean, dtype=np.float64).ravel()
        self.input_std = np.asarray(input_std, dtype=np.float64).ravel()
        self.output_mean = np.asarray(output_mean, dtype=np.float64).ravel()
        self.output_std = np.asarray(output_std, dtype=np.float64).ravel()
        self._state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
        self._hidden_sizes = hidden_sizes
        self._n_hidden = len(hidden_sizes)
        assert self.input_mean.size == 8 and self.output_mean.size == 6

    def casadi(self, x, u, dxdt, *_):
        """Symbolic forward pass: Linear -> Tanh (x3 hidden) -> Linear."""
        z = cs.vertcat(x, u)
        in_norm = (z - np.reshape(self.input_mean, (-1, 1))) / (np.reshape(self.input_std, (-1, 1)) + 1e-10)
        h = cs.tanh(cs.mtimes(self._state_dict["lin1.weight"], in_norm) + np.reshape(self._state_dict["lin1.bias"], (-1, 1)))
        h = cs.tanh(cs.mtimes(self._state_dict["lin2.weight"], h) + np.reshape(self._state_dict["lin2.bias"], (-1, 1)))
        h = cs.tanh(cs.mtimes(self._state_dict["lin3.weight"], h) + np.reshape(self._state_dict["lin3.bias"], (-1, 1)))
        out = cs.mtimes(self._state_dict["lin4.weight"], h) + np.reshape(self._state_dict["lin4.bias"], (-1, 1))
        out_denorm = out * np.reshape(self.output_std, (-1, 1)) + np.reshape(self.output_mean, (-1, 1))
        for i in range(6):
            dxdt[i] = out_denorm[i, 0]
        return dxdt

    def _diffequation(self, t: Optional[float], x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """NumPy forward pass for Governor prediction steps."""
        u_flat = np.asarray(u).ravel()
        x_flat = np.asarray(x).ravel()
        z = np.concatenate([x_flat, u_flat[:2]])
        z = (z - self.input_mean) / (self.input_std + 1e-10)
        z = z.astype(np.float64).reshape(-1, 1)
        h = np.tanh(self._state_dict["lin1.weight"] @ z + self._state_dict["lin1.bias"].reshape(-1, 1))
        h = np.tanh(self._state_dict["lin2.weight"] @ h + self._state_dict["lin2.bias"].reshape(-1, 1))
        h = np.tanh(self._state_dict["lin3.weight"] @ h + self._state_dict["lin3.bias"].reshape(-1, 1))
        out = self._state_dict["lin4.weight"] @ h + self._state_dict["lin4.bias"].reshape(-1, 1)
        out = (out.ravel() * self.output_std + self.output_mean).astype(np.float64)
        return out


# -----------------------------------------------------------------------------
# PINN MLP and training
# -----------------------------------------------------------------------------

# Physics-informed loss: enforce bicycle kinematic consistency (state derivative
# first 3 components = [dx/dt, dy/dt, d(psi)/dt] must equal [vx*cos(psi)-vy*sin(psi),
# vx*sin(psi)+vy*cos(psi), omega]). This is always true regardless of forces.
USE_PHYSICS_INFORMED_LOSS = True
PHYSICS_LOSS_WEIGHT = 1.0  # weight for L_physics relative to L_data


def _kinematic_target_from_state(x_raw: np.ndarray) -> np.ndarray:
    """Compute [dx/dt, dy/dt, d(psi)/dt] from state [px, py, psi, vx, vy, omega]. Shape (n, 3)."""
    psi = x_raw[:, 2]
    vx = x_raw[:, 3]
    vy = x_raw[:, 4]
    omega = x_raw[:, 5]
    dx_dt = vx * np.cos(psi) - vy * np.sin(psi)
    dy_dt = vx * np.sin(psi) + vy * np.cos(psi)
    dpsi_dt = omega
    return np.column_stack([dx_dt, dy_dt, dpsi_dt])


def _sample_varied_state_inputs(
    n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate varied (x, u) samples: uniform mix + nominal region + low speed + high steer
    so the PINN sees diverse regimes. Returns X of shape (n_samples, 8).
    """
    n = n_samples
    # 50% uniform over full range
    n_unif = n // 2
    vx_u = rng.uniform(0.5, 2.5, n_unif)
    vy_u = rng.uniform(-0.5, 0.5, n_unif)
    omega_u = rng.uniform(-1.0, 1.0, n_unif)
    x_u = rng.uniform(-2, 2, n_unif)
    y_u = rng.uniform(-2, 2, n_unif)
    psi_u = rng.uniform(-np.pi, np.pi, n_unif)
    pwm_u = rng.uniform(-0.1, 1.0, n_unif)
    steer_u = rng.uniform(-0.35, 0.35, n_unif)
    X_unif = np.column_stack([x_u, y_u, psi_u, vx_u, vy_u, omega_u, pwm_u, steer_u])

    # 25% nominal driving (vx ~1.5, small steer)
    n_nom = (n - n_unif) // 2
    vx_n = rng.uniform(1.2, 1.8, n_nom)
    vy_n = rng.uniform(-0.2, 0.2, n_nom)
    omega_n = rng.uniform(-0.5, 0.5, n_nom)
    x_n = rng.uniform(-1, 1, n_nom)
    y_n = rng.uniform(-1, 1, n_nom)
    psi_n = rng.uniform(-np.pi, np.pi, n_nom)
    pwm_n = rng.uniform(0.2, 0.8, n_nom)
    steer_n = rng.uniform(-0.2, 0.2, n_nom)
    X_nom = np.column_stack([x_n, y_n, psi_n, vx_n, vy_n, omega_n, pwm_n, steer_n])

    # 25% low speed + high steer / boundary
    n_bnd = n - n_unif - n_nom
    vx_b = rng.uniform(0.4, 0.9, n_bnd)
    vy_b = rng.uniform(-0.4, 0.4, n_bnd)
    omega_b = rng.uniform(-1.2, 1.2, n_bnd)
    x_b = rng.uniform(-2, 2, n_bnd)
    y_b = rng.uniform(-2, 2, n_bnd)
    psi_b = rng.uniform(-np.pi, np.pi, n_bnd)
    pwm_b = rng.uniform(-0.1, 1.0, n_bnd)
    steer_b = rng.uniform(-0.35, 0.35, n_bnd)
    X_bnd = np.column_stack([x_b, y_b, psi_b, vx_b, vy_b, omega_b, pwm_b, steer_b])

    X = np.vstack([X_unif, X_nom, X_bnd]).astype(np.float64)
    rng.shuffle(X)
    return X


class _PINNMLP(nn.Module):
    """3 hidden-layer MLP: 8 -> 64 -> 64 -> 64 -> 6 with Tanh after each hidden layer."""

    def __init__(self, in_dim: int = 8, hidden: Tuple[int, ...] = (64, 64, 64), out_dim: int = 6):
        super().__init__()
        assert len(hidden) == 3, "PINN MLP uses exactly 3 hidden layers"
        self.lin1 = nn.Linear(in_dim, hidden[0])
        self.lin2 = nn.Linear(hidden[0], hidden[1])
        self.lin3 = nn.Linear(hidden[1], hidden[2])
        self.lin4 = nn.Linear(hidden[2], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        return self.lin4(x)


def train_pinn_specialists(
    base_params: Dict[str, Any],
    n_samples: int = 20000,
    hidden_sizes: Tuple[int, int, int] = (64, 64, 64),
    epochs: int = 10000,
    lr: float = 1e-3,
    seed: int = 42,
    val_frac: float = 0.15,
    patience: int = 150,
    min_delta: float = 1e-1,
    use_scheduler: bool = True,
    lbfgs_max_steps: int = 400,
    lbfgs_patience: int = 80,
) -> Tuple[List[NeuralSpecialist], List[Tuple]]:
    """
    Train one MLP per regime (same regimes as create_ode_specialists).
    Data: (x,u) -> dxdt from Dynamic oracle. Returns (neural_specialists, configs).

    Training:
    - Phase 1 (Adam): early stopping, ReduceLROnPlateau, physics-informed loss.
    - Phase 2 (L-BFGS): continue from Adam best with same loss, ReduceLROnPlateau, and early stopping.
    - Physics-informed loss: L_data + weight * L_physics where L_physics enforces
      kinematic consistency (first 3 components of dxdt = [vx*cos(psi)-vy*sin(psi), ...]).
    - Varied sampling: uniform + nominal region + low-speed/boundary (see _sample_varied_state_inputs).

    Learning rate vs early stopping (different roles):
    - ReduceLROnPlateau (Adam only): LR *scheduler*. When val loss does not improve for
      scheduler_patience=15 epochs, it multiplies the current LR by factor=0.5 (down to min_lr=1e-6).
      It does NOT stop training; it only makes Adam take smaller steps.
    - Early stopping: *Stopping* criterion. When val loss does not improve for `patience` epochs
      (Adam) or `lbfgs_patience` steps (L-BFGS), the phase stops and the best model is kept.
      It does not change any learning rate.
    - Adam uses initial lr=1e-3 (argument `lr`). L-BFGS uses lr=1.0 as initial line-search scale;
      ReduceLROnPlateau (when use_scheduler=True) reduces it on plateau down to min_lr=1e-4.
    """
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    specialists: List[NeuralSpecialist] = []
    configs: List[Tuple] = []

    regimes = [
        (1.2, 0.95, 1.1, 0.9, "Optimal: Warm slicks, light"),
        (1.1, 1.0, 1.05, 0.95, "Good: Dry, nominal"),
        (1.0, 1.0, 1.0, 1.0, "Nominal: Baseline"),
        (0.9, 1.05, 0.95, 1.05, "Mild wet"),
        (0.7, 1.1, 0.85, 1.15, "Wet: Heavy rain"),
        (0.6, 1.15, 0.75, 1.25, "Very wet"),
        (0.5, 1.1, 0.7, 1.3, "Ice: Low friction"),
        (0.4, 1.2, 0.6, 1.4, "Extreme: Ice + load"),
    ]

    for idx, (f_scale, m_scale, s_scale, d_scale, desc) in enumerate(tqdm(regimes, desc="PINN regimes", unit="regime")):
        params = base_params.copy()
        params["Df"] = params["Df"] * f_scale
        params["Dr"] = params["Dr"] * f_scale
        params["mass"] = params["mass"] * m_scale
        params["Iz"] = params["Iz"] * m_scale * 1.1
        params["Cf"] = params["Cf"] * s_scale
        params["Cr"] = params["Cr"] * s_scale
        params["Bf"] = params["Bf"] * (0.8 + 0.4 * s_scale)
        params["Br"] = params["Br"] * (0.8 + 0.4 * s_scale)
        params["Cr0"] = params["Cr0"] * d_scale
        params["Cr2"] = params["Cr2"] * d_scale
        oracle = Dynamic(**params)

        assert USE_PHYSICS_INFORMED_LOSS, "PINN training uses physics-informed loss (kinematic residual); set USE_PHYSICS_INFORMED_LOSS=True"

        # Varied sampling: uniform + nominal region + low-speed/boundary (see _sample_varied_state_inputs)
        X = _sample_varied_state_inputs(n_samples, rng)
        dxdt_list = []
        for i in tqdm(range(n_samples), desc=f"  Data {idx}", unit="pt", leave=False):
            x_i = X[i, :6]
            u_i = X[i, 6:8]
            dxdt_list.append(oracle._diffequation(None, x_i, u_i))
        Y = np.array(dxdt_list, dtype=np.float64)

        # Train/val split (use train stats for normalisation to avoid leakage)
        n_val = max(1, int(n_samples * val_frac))
        n_train = n_samples - n_val
        perm = rng.permutation(n_samples)
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        # Physics-informed: kinematic targets [dx/dt, dy/dt, d(psi)/dt] from state (always true)
        k_train_raw = _kinematic_target_from_state(X_train[:, :6])
        k_val_raw = _kinematic_target_from_state(X_val[:, :6])

        input_mean = X_train.mean(axis=0)
        input_std = X_train.std(axis=0) + 1e-8
        output_mean = Y_train.mean(axis=0)
        output_std = Y_train.std(axis=0) + 1e-8
        X_train_n = (X_train - input_mean) / input_std
        Y_train_n = (Y_train - output_mean) / output_std
        X_val_n = (X_val - input_mean) / input_std
        Y_val_n = (Y_val - output_mean) / output_std
        # Normalized kinematic targets for physics-informed loss (first 3 dims)
        k_train_n = (k_train_raw - output_mean[:3]) / (output_std[:3] + 1e-8)
        k_val_n = (k_val_raw - output_mean[:3]) / (output_std[:3] + 1e-8)

        tx = torch.tensor(X_train_n, dtype=torch.float32, device=device)
        ty = torch.tensor(Y_train_n, dtype=torch.float32, device=device)
        tk_train = torch.tensor(k_train_n, dtype=torch.float32, device=device)
        tk_val = torch.tensor(k_val_n, dtype=torch.float32, device=device)
        tx_val = torch.tensor(X_val_n, dtype=torch.float32, device=device)
        ty_val = torch.tensor(Y_val_n, dtype=torch.float32, device=device)

        model = _PINNMLP(8, hidden_sizes, 6).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=15, min_lr=1e-6
            )
            if use_scheduler
            else None
        )

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_no_improve = 0
        final_train_loss = float("nan")
        final_val_loss = float("nan")
        epoch_stopped = 0

        def _total_loss(x_in, y_target, k_target, model_module):
            """Data loss + physics-informed kinematic loss (when USE_PHYSICS_INFORMED_LOSS)."""
            pred = model_module(x_in)
            L_data = nn.functional.mse_loss(pred, y_target)
            if not USE_PHYSICS_INFORMED_LOSS:
                return L_data
            L_physics = nn.functional.mse_loss(pred[:, :3], k_target)
            return L_data + PHYSICS_LOSS_WEIGHT * L_physics

        pbar = tqdm(range(epochs), desc=f"  Adam {idx}", unit="ep", leave=False)
        for ep in pbar:
            model.train()
            opt.zero_grad()
            loss_train = _total_loss(tx, ty, tk_train, model)
            loss_train.backward()
            opt.step()
            final_train_loss = loss_train.item()

            with torch.no_grad():
                model.eval()
                loss_val = _total_loss(tx_val, ty_val, tk_val, model).item()
            final_val_loss = loss_val

            if scheduler is not None:
                scheduler.step(loss_val)

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            pbar.set_postfix(
                train=f"{final_train_loss:.4f}",
                val=f"{final_val_loss:.4f}",
                best=f"{best_val_loss:.4f}",
                stop_in=str(max(0, patience - epochs_no_improve)),
                pinn="on" if USE_PHYSICS_INFORMED_LOSS else "off",
            )

            if epochs_no_improve >= patience:
                epoch_stopped = ep + 1
                break

        # Phase 2: L-BFGS from Adam best, same loss, ReduceLROnPlateau, and early stopping
        if best_state is not None:
            model.load_state_dict(best_state)
        best_val_loss_lbfgs = best_val_loss
        best_state_lbfgs = {k: v.cpu().clone() for k, v in model.state_dict().items()} if best_state is not None else None
        lbfgs_no_improve = 0
        opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
        scheduler_lbfgs = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_lbfgs, mode="min", factor=0.5, patience=15, min_lr=1e-4
            )
            if use_scheduler
            else None
        )
        pbar_lbfgs = tqdm(range(lbfgs_max_steps), desc=f"  L-BFGS {idx}", unit="step", leave=False)
        for step in pbar_lbfgs:
            def closure():
                opt_lbfgs.zero_grad()
                loss = _total_loss(tx, ty, tk_train, model)
                loss.backward()
                return loss
            opt_lbfgs.step(closure)
            with torch.no_grad():
                model.eval()
                loss_v = _total_loss(tx_val, ty_val, tk_val, model).item()
            if scheduler_lbfgs is not None:
                scheduler_lbfgs.step(loss_v)
            if loss_v < best_val_loss_lbfgs:
                best_val_loss_lbfgs = loss_v
                best_state_lbfgs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                lbfgs_no_improve = 0
            else:
                lbfgs_no_improve += 1
            current_lr = opt_lbfgs.param_groups[0]["lr"] if scheduler_lbfgs is not None else 1.0
            pbar_lbfgs.set_postfix(val=f"{loss_v:.4f}", best=f"{best_val_loss_lbfgs:.4f}", lr=f"{current_lr:.0e}", stop_in=str(max(0, lbfgs_patience - lbfgs_no_improve)))
            if lbfgs_no_improve >= lbfgs_patience:
                break
        if best_state_lbfgs is not None:
            model.load_state_dict(best_state_lbfgs)
            best_val_loss = best_val_loss_lbfgs

        state_dict = {k: v.detach() for k, v in model.state_dict().items()}
        spec = NeuralSpecialist(
            state_dict=state_dict,
            input_mean=input_mean, input_std=input_std,
            output_mean=output_mean, output_std=output_std,
            hidden_sizes=hidden_sizes,
        )
        specialists.append(spec)
        configs.append((f_scale, m_scale, s_scale, d_scale, desc))
        stop_msg = f" early@ep{epoch_stopped}" if epoch_stopped else ""
        print(
            f"  PINN Specialist {idx}: {desc} (train={final_train_loss:.6f}, val={final_val_loss:.6f}, best_val={best_val_loss:.6f}{stop_msg})"
        )

    return specialists, configs


# -----------------------------------------------------------------------------
# No-shift diagnostic: run and plot
# -----------------------------------------------------------------------------

def run_no_shift_diagnostic(
    regimes: List[Tuple[str, Dict[str, float]]],
    specialists: List[Any],
    friction_scales: np.ndarray,
    verbose: bool = True,
    use_dyn_noise: bool = False,
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[Dict, Dict, Dict, Dict, str]]:
    """
    For each (name, shift_dict), run A=Baseline+Shift, B=Adaptive+Shift,
    C=Baseline+NoShift, D=Adaptive+NoShift. Return data_list for plotting.
    use_dyn_noise=True contaminates the NMPC dynamics with Gaussian noise (aleatoric-only).
    """
    params_nom = ORCA(control="pwm")
    CHANGE_TIME = 2.2
    data_list: List[Tuple[Dict, Dict, Dict, Dict, str]] = []
    _rng = rng if rng is not None else np.random.default_rng()

    for name, shift_dict in tqdm(regimes, desc="No-shift diagnostic regimes", unit="regime"):
        params_shifted = apply_regime_shift(params_nom, shift_dict)
        if verbose:
            print(f"\n--- No-Shift Diagnostic: {name} ---")
            print("Running A (Baseline+Shift), B (Adaptive+Shift), C (Baseline+NoShift), D (Adaptive+NoShift)...")
        log_A = run_simulation_ablation(
            "Baseline", specialists, friction_scales,
            params_nom, params_shifted, adaptive_mode=False, no_shift=False,
            use_dyn_noise=use_dyn_noise, noise_std=noise_std, rng=_rng,
        )
        log_B = run_simulation_ablation(
            "Adaptive", specialists, friction_scales,
            params_nom, params_shifted, adaptive_mode=True, no_shift=False,
            use_dyn_noise=use_dyn_noise, noise_std=noise_std, rng=_rng,
        )
        log_C = run_simulation_ablation(
            "Baseline", specialists, friction_scales,
            params_nom, params_shifted, adaptive_mode=False, no_shift=True,
            use_dyn_noise=use_dyn_noise, noise_std=noise_std, rng=_rng,
        )
        log_D = run_simulation_ablation(
            "Adaptive", specialists, friction_scales,
            params_nom, params_shifted, adaptive_mode=True, no_shift=True,
            use_dyn_noise=use_dyn_noise, noise_std=noise_std, rng=_rng,
        )
        data_list.append((log_A, log_B, log_C, log_D, name))
        if verbose:
            m_A = calculate_metrics(log_A, CHANGE_TIME, "A (Baseline+Shift)", verbose=False)
            m_B = calculate_metrics(log_B, CHANGE_TIME, "B (Adaptive+Shift)", verbose=False)
            m_C = calculate_metrics(log_C, CHANGE_TIME, "C (Baseline+NoShift)", verbose=False)
            m_D = calculate_metrics(log_D, CHANGE_TIME, "D (Adaptive+NoShift)", verbose=False)
            geom_vx = abs(m_C["rmse_vx_post"] - m_C["rmse_vx_pre"])
            geom_vy = abs(m_C["rmse_vy_post"] - m_C["rmse_vy_pre"])
            shift_vx_b = m_A["rmse_vx_post"] - m_A["rmse_vx_pre"]
            shift_vx_a = m_B["rmse_vx_post"] - m_B["rmse_vx_pre"]
            shift_vy_b = m_A["rmse_vy_post"] - m_A["rmse_vy_pre"]
            shift_vy_a = m_B["rmse_vy_post"] - m_B["rmse_vy_pre"]
            print("  Geometry effect (no-shift): vx ~{:.4f}, vy ~{:.4f}".format(geom_vx, geom_vy))
            print("  Shift degradation (baseline): vx {:.4f}, vy {:.4f}".format(shift_vx_b, shift_vy_b))
            print("  Adaptation reduces: vx by {:.4f}, vy by {:.4f}".format(
                shift_vx_b - shift_vx_a, shift_vy_b - shift_vy_a))
    return data_list


def plot_no_shift_diagnostic(
    data_list: List[Tuple[Dict, Dict, Dict, Dict, str]],
    specialists: List[Any],
    surrogate_label: str,
    save_path: str,
) -> None:
    """
    2×3 layout: vx error, vy error, Governor weights. Four curves per vx/vy:
    Baseline+Shift, Adaptive+Shift, Baseline+NoShift, Adaptive+NoShift.
    """
    FONT_SIZE_LABEL = 14
    FONT_SIZE_TITLE = 16
    FONT_SIZE_PANEL = 16
    FONT_SIZE_ANNOTATION = 14
    FONT_SIZE_TICKS = 12
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    legend_handles = [
        Line2D([0], [0], color="red", linestyle="--", linewidth=2.5, label="Baseline + Shift"),
        Line2D([0], [0], color="blue", linestyle="-", linewidth=2.5, label="Adaptive + Shift"),
        Line2D([0], [0], color="red", linestyle=":", linewidth=2, label="Baseline + No shift"),
        Line2D([0], [0], color="blue", linestyle=":", linewidth=2, label="Adaptive + No shift"),
    ]
    for row, (log_A, log_B, log_C, log_D, title) in enumerate(data_list):
        t = np.array(log_A["t"])
        ref_vx = np.array(log_A["ref_vx"])
        err_vx_A = ref_vx - np.array(log_A["vx"])
        err_vx_B = ref_vx - np.array(log_B["vx"])
        err_vx_C = np.array(log_C["ref_vx"]) - np.array(log_C["vx"])
        err_vx_D = np.array(log_D["ref_vx"]) - np.array(log_D["vx"])
        err_vy_A = -np.array(log_A["vy"])
        err_vy_B = -np.array(log_B["vy"])
        err_vy_C = -np.array(log_C["vy"])
        err_vy_D = -np.array(log_D["vy"])
        ax = axes[row, 0]
        ax.axhline(0, color="k", linestyle=":", alpha=0.6, linewidth=1)
        ax.plot(t, err_vx_A, "r--", linewidth=2, alpha=0.8)
        ax.plot(t, err_vx_B, "b-", linewidth=2, alpha=0.9)
        ax.plot(t, err_vx_C, "r:", linewidth=1.5, alpha=0.7)
        ax.plot(t, err_vx_D, "b:", linewidth=1.5, alpha=0.7)
        ax.axvline(2.2, color="k", linestyle="-", alpha=0.5, linewidth=1.5)
        ax.set_ylabel(r"$v_x$ Error [m/s]", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FONT_SIZE_TICKS)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, panel_labels[row * 3 + 0], transform=ax.transAxes,
                fontsize=FONT_SIZE_PANEL, fontweight="bold", va="top", ha="left")
        ax.text(2.25, ax.get_ylim()[1] * 0.95, "Regime\nShift", fontsize=FONT_SIZE_ANNOTATION, color="k", va="top", ha="left")
        if row == 1:
            ax.set_xlabel("Time [s]", fontsize=FONT_SIZE_LABEL)
        ax = axes[row, 1]
        ax.axhline(0, color="k", linestyle=":", alpha=0.6, linewidth=1)
        ax.plot(t, err_vy_A, "r--", linewidth=1.5, alpha=0.8)
        ax.plot(t, err_vy_B, "b-", linewidth=2, alpha=0.9)
        ax.plot(t, err_vy_C, "r:", linewidth=1.5, alpha=0.7)
        ax.plot(t, err_vy_D, "b:", linewidth=1.5, alpha=0.7)
        ax.axvline(2.2, color="k", linestyle="-", alpha=0.5, linewidth=1.5)
        ax.set_ylabel(r"$v_y$ Error [m/s]", fontsize=FONT_SIZE_LABEL)
        ax.set_title("Lateral Velocity", fontsize=FONT_SIZE_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FONT_SIZE_TICKS)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, panel_labels[row * 3 + 1], transform=ax.transAxes,
                fontsize=FONT_SIZE_PANEL, fontweight="bold", va="top", ha="left")
        ax.text(2.25, ax.get_ylim()[1] * 0.95, "Regime\nShift", fontsize=FONT_SIZE_ANNOTATION, color="k", va="top", ha="left")
        if row == 1:
            ax.set_xlabel("Time [s]", fontsize=FONT_SIZE_LABEL)
        ax = axes[row, 2]
        w_B = np.array(log_B["w"])
        w_D = np.array(log_D["w"])
        n_spec = w_B.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_spec))
        for i in range(n_spec):
            ax.plot(t, w_B[:, i], color=colors[i], linestyle="-", alpha=0.8, linewidth=1.5)
        for i in range(n_spec):
            ax.plot(t, w_D[:, i], color=colors[i], linestyle=":", alpha=0.6, linewidth=1)
        ax.axvline(2.2, color="k", linestyle="-", alpha=0.5, linewidth=1.5)
        ax.set_ylabel("Weights", fontsize=FONT_SIZE_LABEL)
        ax.set_xlabel("Time [s]", fontsize=FONT_SIZE_LABEL)
        ax.set_title("Governor Adaptation", fontsize=FONT_SIZE_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FONT_SIZE_TICKS)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, panel_labels[row * 3 + 2], transform=ax.transAxes,
                fontsize=FONT_SIZE_PANEL, fontweight="bold", va="top", ha="left")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.suptitle(f"No-Shift Diagnostic — {surrogate_label}", fontsize=18, fontweight="bold", y=0.96)
    fig.legend(
        handles=legend_handles, labels=[h.get_label() for h in legend_handles],
        loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.94),
        bbox_transform=fig.transFigure, fontsize=FONT_SIZE_LABEL, frameon=True, fancybox=False,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"No-shift diagnostic plot saved to '{save_path}' ({surrogate_label})")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    CHANGE_TIME = 2.2
    LOG_PATH = "no_shift_diagnostic_metrics.log"
    JSON_PATH = "no_shift_diagnostic_metrics.json"

    # Regimes for the No-Shift Diagnostic (indices 1 and 6 from the ablation study)
    REGIMES: List[Tuple[str, Dict[str, float]]] = [
        ("Friction Only (Severe)", {"friction": 0.5}),
        ("All Parameters (Severe)", {"friction": 0.5, "mass": 1.2, "stiffness": 0.7, "drag": 1.4}),
    ]
    params_nom = ORCA(control="pwm")

    all_phases_json: List[Dict[str, Any]] = []
    with open(LOG_PATH, "w") as log_f:
        log_f.write(f"No-Shift Diagnostic — Metrics Log\n")
        log_f.write(f"Generated: {datetime.now().isoformat()}\n")
        log_f.write(f"Regime shift time: t = {CHANGE_TIME} s\n")
        log_f.write(f"Methods: A=Baseline+Shift, B=Adaptive+Shift, C=Baseline+NoShift, D=Adaptive+NoShift\n")

        # ----- ODE benchmark -----
        print("\n" + "=" * 70)
        print("NO-SHIFT DIAGNOSTIC — ODE Surrogate (Idealized Specialists)")
        print("=" * 70)
        specialists_ode, configs_ode = create_ode_specialists(params_nom, n_specialists=8)
        friction_scales = np.array([c[0] for c in configs_ode])
        data_ode = run_no_shift_diagnostic(REGIMES, specialists_ode, friction_scales, verbose=True)
        plot_no_shift_diagnostic(
            data_ode, specialists_ode,
            surrogate_label="ODE Surrogate",
            save_path="no_shift_diagnostic_ODE.png",
        )
        table_ode = compute_diagnostic_metrics_table(data_ode, change_time=CHANGE_TIME)
        report_ode, json_ode = format_metrics_report("ODE Surrogate", table_ode, change_time=CHANGE_TIME)
        print(report_ode)
        log_f.write(report_ode)
        log_f.write("\n")
        all_phases_json.append(json_ode)

        # ----- ODE + Gaussian noise (aleatoric-only surrogate) -----
        print("\n" + "=" * 70)
        print("NO-SHIFT DIAGNOSTIC — ODE Surrogate + Aleatoric Noise (NMPC contaminated)")
        print("=" * 70)
        specialists_ode_noise, configs_ode_noise = create_ode_specialists_with_noise(
            params_nom, n_specialists=8, noise_std=0.05, seed=42
        )
        friction_scales_ode_noise = np.array([c[0] for c in configs_ode_noise])
        rng_ode_noise = np.random.default_rng(42)
        data_ode_noise = run_no_shift_diagnostic(
            REGIMES, specialists_ode_noise, friction_scales_ode_noise, verbose=True,
            use_dyn_noise=True, noise_std=0.05, rng=rng_ode_noise,
        )
        plot_no_shift_diagnostic(
            data_ode_noise, specialists_ode_noise,
            surrogate_label="ODE Surrogate + Aleatoric Noise",
            save_path="no_shift_diagnostic_ODE_noise.png",
        )
        table_ode_noise = compute_diagnostic_metrics_table(data_ode_noise, change_time=CHANGE_TIME)
        report_ode_noise, json_ode_noise = format_metrics_report("ODE Surrogate + Aleatoric Noise", table_ode_noise, change_time=CHANGE_TIME)
        print(report_ode_noise)
        log_f.write(report_ode_noise)
        log_f.write("\n")
        all_phases_json.append(json_ode_noise)

        # ----- PINN benchmark -----
        print("\n" + "=" * 70)
        print("NO-SHIFT DIAGNOSTIC — Neural Network (PINN) Surrogate")
        print("=" * 70)
        print("Starting PINN simulation (Expect slow compilation/solve times due to symbolic graph complexity)...")
        specialists_pinn, configs_pinn = train_pinn_specialists(
            params_nom,
            n_samples=20000,
            hidden_sizes=(64, 64, 64),
            epochs=10,
            lr=1e-3,
            seed=42,
            val_frac=0.15,
            patience=150,
            use_scheduler=True,
            lbfgs_max_steps=10,
            lbfgs_patience=150,
        )
        friction_scales_pinn = np.array([c[0] for c in configs_pinn])
        data_pinn = run_no_shift_diagnostic(REGIMES, specialists_pinn, friction_scales_pinn, verbose=True)
        plot_no_shift_diagnostic(
            data_pinn, specialists_pinn,
            surrogate_label="Neural Network (PINN)",
            save_path="no_shift_diagnostic_PINN.png",
        )
        table_pinn = compute_diagnostic_metrics_table(data_pinn, change_time=CHANGE_TIME)
        report_pinn, json_pinn = format_metrics_report("Neural Network (PINN)", table_pinn, change_time=CHANGE_TIME)
        print(report_pinn)
        log_f.write(report_pinn)
        log_f.write("\n")
        all_phases_json.append(json_pinn)

    with open(JSON_PATH, "w") as jf:
        json.dump(
            {"generated": datetime.now().isoformat(), "change_time_s": CHANGE_TIME, "phases": all_phases_json},
            jf,
            indent=2,
        )
    print("\n=== No-Shift Diagnostic Complete ===")
    print("ODE figure:       no_shift_diagnostic_ODE.png")
    print("ODE+noise figure: no_shift_diagnostic_ODE_noise.png")
    print("PINN figure:      no_shift_diagnostic_PINN.png")
    print(f"Metrics log:      {LOG_PATH}")
    print(f"Metrics JSON:     {JSON_PATH}")


if __name__ == "__main__":
    main()
