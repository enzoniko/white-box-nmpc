from __future__ import annotations

"""
Friction (mu_scale) estimators for ORCA closed-loop experiments.

We implement two lightweight baselines:
  - RLS (Recursive Least Squares) on a *local linearization* of dv(mu)
  - UKF (Unscented Kalman Filter) treating mu as a latent scalar state with random-walk dynamics

These are intended as baseline estimators feeding into `phys_mu` (parameterized physics)
without requiring solver rebuilds.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np


@dataclass
class RLSConfig:
    lam: float = 0.98  # forgetting factor
    initial_P: float = 100.0
    phi_min: float = 1e-6  # skip updates when sensitivity is too small
    fd_eps: float = 1e-3   # finite difference epsilon for d dv / d mu
    clamp_min: float = 0.05
    clamp_max: float = 3.0
    max_update_steps: Optional[int] = None  # if set: stop updating after this many calls


class RLSMuEstimator:
    """
    Scalar-parameter RLS on locally linearized dynamics.

    Measurement model uses linearization:
      dv_obs ≈ dv_pred(mu0) + J(mu0) * (mu - mu0)

    Rearranged into a (scalar) regression per component:
      y_j := dv_obs_j - dv_pred_j(mu0) + J_j(mu0) * mu0  ≈  J_j(mu0) * mu
      phi := J_j(mu0)

    We run three sequential scalar RLS updates per time step (dvx,dvy,domega),
    skipping components with |phi| < phi_min.
    """

    def __init__(self, mu0: float, cfg: RLSConfig):
        self.cfg = cfg
        self.mu = float(mu0)
        self.P = float(cfg.initial_P)
        self._n_updates = 0

        # diagnostics
        self.last_denoms = []  # type: list[float]
        self.last_phis = []    # type: list[float]
        self.last_residuals = []  # type: list[float]

    def _clamp(self, mu: float) -> float:
        return float(np.clip(mu, self.cfg.clamp_min, self.cfg.clamp_max))

    def update(
        self,
        dv_obs: np.ndarray,
        dv_pred_fn: Callable[[float], np.ndarray],
    ) -> Dict[str, float]:
        """
        Update mu estimate using measured dv and a callback dv_pred_fn(mu)->dv.
        """
        if self.cfg.max_update_steps is not None and self._n_updates >= int(self.cfg.max_update_steps):
            return {"updated": 0.0, "mu": float(self.mu)}

        dv_obs = np.asarray(dv_obs, dtype=np.float64).reshape(3)
        mu0 = float(self.mu)
        eps = float(self.cfg.fd_eps)

        dv0 = np.asarray(dv_pred_fn(mu0), dtype=np.float64).reshape(3)
        dvp = np.asarray(dv_pred_fn(mu0 + eps), dtype=np.float64).reshape(3)
        dvm = np.asarray(dv_pred_fn(mu0 - eps), dtype=np.float64).reshape(3)
        J = (dvp - dvm) / (2.0 * eps)  # d dv / d mu, shape (3,)

        n_used = 0
        denoms = []
        phis = []
        residuals = []

        lam = float(self.cfg.lam)
        P = float(self.P)
        theta = float(self.mu)

        for j in range(3):
            phi = float(J[j])
            if abs(phi) < float(self.cfg.phi_min):
                continue
            y = float(dv_obs[j] - dv0[j] + phi * mu0)
            denom = lam + phi * P * phi
            K = (P * phi) / denom
            resid = y - phi * theta
            theta = theta + K * resid
            P = (1.0 / lam) * (P - K * phi * P)
            n_used += 1
            denoms.append(float(denom))
            phis.append(float(phi))
            residuals.append(float(resid))

        theta = self._clamp(theta)
        self.mu = float(theta)
        self.P = float(P)
        self._n_updates += 1

        # store rolling diagnostics (bounded)
        self.last_denoms = (self.last_denoms + denoms)[-200:]
        self.last_phis = (self.last_phis + phis)[-200:]
        self.last_residuals = (self.last_residuals + residuals)[-200:]

        out: Dict[str, float] = {
            "updated": float(1.0 if n_used > 0 else 0.0),
            "n_components_used": float(n_used),
            "mu": float(self.mu),
            "P": float(self.P),
        }
        if denoms:
            out["denom_min"] = float(np.min(denoms))
            out["denom_p95"] = float(np.percentile(denoms, 95))
        if phis:
            out["phi_abs_mean"] = float(np.mean(np.abs(phis)))
            out["phi_abs_min"] = float(np.min(np.abs(phis)))
        if residuals:
            out["resid_abs_mean"] = float(np.mean(np.abs(residuals)))
        return out


@dataclass
class UKFConfig:
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    process_noise: float = 1e-4  # Q (scalar)
    measurement_noise: float = 0.1  # R (scalar, applied to each dv component)
    initial_P: float = 0.25
    clamp_min: float = 0.05
    clamp_max: float = 3.0
    max_update_steps: Optional[int] = None


class UKFMuEstimator:
    """
    Unscented KF estimating scalar mu from 3D dv measurements.

    State: mu
    Process: mu_{k+1} = mu_k + w,  w ~ N(0, Q)
    Measurement: z_k = dv(mu_k) + v, v ~ N(0, R I_3)
    """

    def __init__(self, mu0: float, cfg: UKFConfig):
        self.cfg = cfg
        self.mu = float(mu0)
        self.P = float(cfg.initial_P)
        self._n_updates = 0

        # diagnostics
        self.last_cond_S = []  # type: list[float]
        self.last_innov_norm = []  # type: list[float]

    def _clamp(self, mu: float) -> float:
        return float(np.clip(mu, self.cfg.clamp_min, self.cfg.clamp_max))

    def update(
        self,
        dv_obs: np.ndarray,
        dv_pred_fn: Callable[[float], np.ndarray],
    ) -> Dict[str, float]:
        if self.cfg.max_update_steps is not None and self._n_updates >= int(self.cfg.max_update_steps):
            return {"updated": 0.0, "mu": float(self.mu)}

        z = np.asarray(dv_obs, dtype=np.float64).reshape(3, 1)
        x = float(self.mu)
        P = float(self.P)

        n = 1
        alpha = float(self.cfg.alpha)
        beta = float(self.cfg.beta)
        kappa = float(self.cfg.kappa)
        lam = alpha * alpha * (n + kappa) - n
        c = n + lam

        # sigma points for 1D
        sqrt_cP = np.sqrt(max(c * P, 0.0))
        X = np.array([x, x + sqrt_cP, x - sqrt_cP], dtype=np.float64)  # (2n+1,)

        Wm = np.full((2 * n + 1,), 1.0 / (2.0 * c), dtype=np.float64)
        Wc = np.full((2 * n + 1,), 1.0 / (2.0 * c), dtype=np.float64)
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - alpha * alpha + beta)

        # Predict step (random walk)
        Q = float(self.cfg.process_noise)
        x_pred = float(np.sum(Wm * X))
        P_pred = float(np.sum(Wc * (X - x_pred) ** 2) + Q)

        # Propagate sigma points through measurement function
        Zsig = np.zeros((3, 2 * n + 1), dtype=np.float64)
        for i in range(2 * n + 1):
            Zsig[:, i] = np.asarray(dv_pred_fn(float(X[i])), dtype=np.float64).reshape(3)

        z_pred = (Zsig @ Wm.reshape(-1, 1)).reshape(3, 1)

        Rv = float(self.cfg.measurement_noise)
        R = (Rv ** 2) * np.eye(3)
        # Innovation covariance
        S = np.zeros((3, 3), dtype=np.float64)
        for i in range(2 * n + 1):
            dz = (Zsig[:, i].reshape(3, 1) - z_pred)
            S += Wc[i] * (dz @ dz.T)
        S += R

        # Cross-covariance Pxz (1x3)
        Pxz = np.zeros((1, 3), dtype=np.float64)
        for i in range(2 * n + 1):
            dx = float(X[i] - x_pred)
            dz = (Zsig[:, i].reshape(3, 1) - z_pred)
            Pxz += Wc[i] * dx * dz.T

        # Kalman gain (1x3)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = Pxz @ S_inv

        innov = (z - z_pred)  # (3,1)
        x_new = float(x_pred + (K @ innov).reshape(1)[0])
        P_new = float(P_pred - (K @ S @ K.T).reshape(1)[0])

        x_new = self._clamp(x_new)
        P_new = max(P_new, 1e-12)

        self.mu = float(x_new)
        self.P = float(P_new)
        self._n_updates += 1

        # diagnostics
        try:
            condS = float(np.linalg.cond(S))
        except Exception:
            condS = float("nan")
        innov_norm = float(np.linalg.norm(innov))
        self.last_cond_S = (self.last_cond_S + [condS])[-200:]
        self.last_innov_norm = (self.last_innov_norm + [innov_norm])[-200:]

        return {
            "updated": 1.0,
            "mu": float(self.mu),
            "P": float(self.P),
            "cond_S": float(condS),
            "innov_norm": float(innov_norm),
        }


