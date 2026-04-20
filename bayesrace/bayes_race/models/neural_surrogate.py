"""
Neural Surrogate Dynamic Model (BayesRace-compatible)

This provides a drop-in `Model` for BayesRace NMPC that uses the s2gpt_pinn
specialist ensemble (or monolith) to produce the (vx, vy, omega) derivatives.

State convention (BayesRace Dynamic):
  x = [X, Y, psi, vx, vy, omega]  (6 states)
Input convention (BayesRace Dynamic, control='pwm'):
  u = [pwm, steer]

Our surrogate convention:
  state_dyn = [vx, vy, omega]
  control_dyn = [delta, throttle] = [steer, pwm]
  static = [m, Iz]

Two CasADi backends:
  - "callback": CasADi -> Python callback -> PyTorch forward + analytic Jacobian
  - "native": pure CasADi graph exported from PyTorch weights

Top-K gating:
  - Supported in callback backend (skips evaluation of non-topK specialists)
  - For native backend, Top-K is represented by passing sparse weights (still computes all specialists).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Sequence

import numpy as np
import casadi as cs

from bayes_race.models.model import Model


@dataclass
class SurrogateParams:
    static_mass: float
    static_Iz: float
    mu_fixed: float = 0.9
    top_k: Optional[int] = None
    # For Mode B experiments (weights passed as NLP parameter)
    n_specialists: Optional[int] = None


class NeuralSurrogateDynamic(Model):
    def __init__(
        self,
        ensemble,
        params: SurrogateParams,
        backend: Literal["callback", "native"] = "native",
        mode: Literal["fixed_mu", "weights_param"] = "fixed_mu",
    ):
        super().__init__()
        self.ensemble = ensemble
        self.params = params
        self.backend = backend
        self.mode = mode

        self.n_states = 6
        self.n_inputs = 2

        # Build CasADi callable for accel dynamics
        self._f_accel = None
        self._build_backend()

    def _build_backend(self):
        static = np.array([self.params.static_mass, self.params.static_Iz], dtype=np.float32)

        if self.backend == "callback":
            # Python callback route
            from s2gpt_pinn.casadi_callbacks import HSSDynamicsCallback

            if self.mode == "fixed_mu":
                self._cb = HSSDynamicsCallback(
                    name="hss_cb",
                    ensemble=self.ensemble,
                    static_params=static,
                    mu_current=self.params.mu_fixed,
                    top_k=self.params.top_k,
                )

                x = cs.SX.sym("x", 3)
                u = cs.SX.sym("u", 2)
                self._f_accel = cs.Function("f_accel_cb", [x, u], [self._cb(x, u)])
                return

            if self.mode == "weights_param":
                if self.params.n_specialists is None:
                    raise ValueError("weights_param mode requires n_specialists")
                w = cs.SX.sym("w", self.params.n_specialists)
                self._cb = HSSDynamicsCallback(
                    name="hss_cb_w",
                    ensemble=self.ensemble,
                    static_params=static,
                    use_mode_b_weights=True,
                    mode_b_weights=np.ones(self.params.n_specialists, dtype=np.float32) / self.params.n_specialists,
                    top_k=self.params.top_k,
                )
                x = cs.SX.sym("x", 3)
                u = cs.SX.sym("u", 2)
                # weights enter through parameters of callback object (not symbolic),
                # so for "weights_param" we prefer native backend. Keep this for completeness.
                self._f_accel = cs.Function("f_accel_cb_w", [x, u, w], [self._cb(x, u)])
                return

            raise ValueError(self.mode)

        if self.backend == "native":
            from s2gpt_pinn.casadi_callbacks import export_ensemble_to_casadi, CasadiExportConfig

            if self.mode == "fixed_mu":
                f, _ = export_ensemble_to_casadi(
                    self.ensemble,
                    CasadiExportConfig(mode="mode_a_rbf", rbf_width=float(getattr(self.ensemble, "rbf_width", 0.15))),
                    name="ens_native",
                )
                x = cs.SX.sym("x", 3)
                u = cs.SX.sym("u", 2)
                mu = cs.DM([self.params.mu_fixed])
                s = cs.DM(static.astype(np.float64))
                y = f(x, u, s, mu)
                self._f_accel = cs.Function("f_accel_native", [x, u], [y])
                return

            if self.mode == "weights_param":
                if self.params.n_specialists is None:
                    raise ValueError("weights_param mode requires n_specialists")
                f, _ = export_ensemble_to_casadi(
                    self.ensemble,
                    CasadiExportConfig(mode="mode_b_weights", normalize_weights=True),
                    name="ens_native_w",
                )
                x = cs.SX.sym("x", 3)
                u = cs.SX.sym("u", 2)
                w = cs.SX.sym("w", self.params.n_specialists)
                s = cs.DM(static.astype(np.float64))
                y = f(x, u, s, w)
                self._f_accel = cs.Function("f_accel_native_w", [x, u, w], [y])
                return

            raise ValueError(self.mode)

        raise ValueError(self.backend)

    def casadi(self, x, u, dxdt, dyn_params=None, step_index=None):
        """
        BayesRace expects dxdt to be filled in-place.
        step_index: optional prediction step index (for setupNLPAdaptive compatibility).
        """
        pwm = u[0]
        steer = u[1]

        psi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]

        # Kinematics in inertial frame
        dxdt[0] = vx * cs.cos(psi) - vy * cs.sin(psi)
        dxdt[1] = vx * cs.sin(psi) + vy * cs.cos(psi)
        dxdt[2] = omega

        x_dyn = cs.vertcat(vx, vy, omega)
        u_dyn = cs.vertcat(steer, pwm)  # [delta, throttle]

        if self.mode == "fixed_mu":
            y = self._f_accel(x_dyn, u_dyn)
        else:
            if dyn_params is None:
                raise ValueError("weights_param mode requires dyn_params (weights) symbol/value")
            y = self._f_accel(x_dyn, u_dyn, dyn_params)

        # Surrogate outputs are [dvx, dvy, domega]
        dxdt[3] = y[0]
        dxdt[4] = y[1]
        dxdt[5] = y[2]
        return dxdt


