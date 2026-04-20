#!/usr/bin/env python3
"""
CasADi Integration for H-SS / S²GPT-style Specialist Ensembles

This module provides TWO integration strategies:

1) Python Callback (slowest, simplest):
   - CasADi calls back into Python at each dynamics evaluation.
   - We evaluate the PyTorch ensemble and (optionally) its analytic Jacobians.

2) CasADi-Native Graph (fastest once built):
   - We export the trained MLP(s) (weights/biases) into CasADi expressions.
   - The resulting dynamics is a pure CasADi graph, enabling:
       - CasADi/Ipopt to use analytic derivatives
       - optional codegen / compilation
   - Adaptation can be done WITHOUT rebuilding the graph by:
       - updating weights as parameters (Mode B), or
       - updating mu_current as a parameter (Mode A RBF).

Notes:
- This code targets the 3-state surrogate used by this repo: [vx, vy, omega].
- Inputs are [delta, throttle], static params are [m, Iz] (locked / known).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np
import casadi as cs
import torch

from .specialist import HSSEnsemble, HSSSpecialist


@dataclass
class CasadiExportConfig:
    """Options for exporting specialists/ensemble into CasADi graphs."""
    mode: Literal["mode_a_rbf", "mode_b_weights"] = "mode_b_weights"
    # If mode_a_rbf: weights = soft RBF(mu_current, mu_centers)
    rbf_width: float = 0.15
    # If mode_b_weights: provide weights parameter vector w[K]
    normalize_weights: bool = True


def _mlp_forward_casadi(
    x: cs.SX,
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    output_scale: np.ndarray,
) -> cs.SX:
    """
    Forward pass for tanh MLP with final linear layer.

    weights: list of W (out_dim, in_dim) for each layer, including output layer
    biases: list of b (out_dim,) for each layer, including output layer
    """
    h = x
    n_layers_total = len(weights)
    for li in range(n_layers_total):
        W = cs.DM(weights[li])
        b = cs.DM(biases[li])
        h = cs.mtimes(W, h) + b
        if li < n_layers_total - 1:
            h = cs.tanh(h)
    # scale output to physical range (matches PyTorch)
    return cs.diag(cs.DM(output_scale)) @ h


def export_specialist_to_casadi(
    specialist: HSSSpecialist,
    name: str = "specialist",
) -> cs.Function:
    """
    Export a single trained HSSSpecialist to a CasADi Function:
        f([vx,vy,omega], [delta,throttle], [m,Iz]) -> [dvx,dvy,domega]
    """
    specialist.eval()

    # Extract normalization scales and output scale
    state_scale = specialist.state_scale.detach().cpu().numpy().astype(np.float64)
    control_scale = specialist.control_scale.detach().cpu().numpy().astype(np.float64)
    static_scale = specialist.static_scale.detach().cpu().numpy().astype(np.float64)
    output_scale = specialist.output_scale.detach().cpu().numpy().astype(np.float64)

    # Extract weights/biases
    W_list: List[np.ndarray] = []
    b_list: List[np.ndarray] = []
    for layer in specialist.layers:
        W_list.append(layer.weight.detach().cpu().numpy().astype(np.float64))
        b_list.append(layer.bias.detach().cpu().numpy().astype(np.float64))
    W_list.append(specialist.output_layer.weight.detach().cpu().numpy().astype(np.float64))
    b_list.append(specialist.output_layer.bias.detach().cpu().numpy().astype(np.float64))

    x = cs.SX.sym("x", 3)
    u = cs.SX.sym("u", 2)
    s = cs.SX.sym("s", 2)

    # Normalize inputs exactly like PyTorch
    x_norm = cs.diag(1.0 / cs.DM(state_scale)) @ x
    u_norm = cs.diag(1.0 / cs.DM(control_scale)) @ u
    s_norm = cs.diag(1.0 / cs.DM(static_scale)) @ s
    inp = cs.vertcat(x_norm, u_norm, s_norm)  # 7x1

    y = _mlp_forward_casadi(inp, W_list, b_list, output_scale)
    return cs.Function(name, [x, u, s], [y], ["x", "u", "s"], ["y"])


def export_ensemble_to_casadi(
    ensemble: HSSEnsemble,
    export_cfg: CasadiExportConfig,
    name: str = "hss_ensemble",
) -> Tuple[cs.Function, dict]:
    """
    Export an ensemble to a CasADi Function.

    Returns (f, meta) where:
      - f: CasADi Function with signature depending on mode:
        mode_b_weights: f(x,u,s,w) -> y
        mode_a_rbf:     f(x,u,s,mu) -> y
      - meta: info dict (n_specialists, etc.)
    """
    K = ensemble.n_specialists
    specialists = list(ensemble.specialists)

    # Export each specialist into a CasADi expression (not a nested Function call for speed)
    # We'll inline each specialist's MLP into one big CasADi graph.
    exported = [export_specialist_to_casadi(sp, name=f"{name}_sp{i}") for i, sp in enumerate(specialists)]

    x = cs.SX.sym("x", 3)
    u = cs.SX.sym("u", 2)
    s = cs.SX.sym("s", 2)

    y_list = [fi(x, u, s) for fi in exported]  # each is (3,1)

    if export_cfg.mode == "mode_b_weights":
        w_sym = cs.SX.sym("w", K)
        w_eff = w_sym
        if export_cfg.normalize_weights:
            w_pos = cs.fmax(w_sym, 0)
            w_eff = w_pos / (cs.sum1(w_pos) + 1e-12)
        y = cs.SX.zeros(3, 1)
        for i in range(K):
            y += w_eff[i] * y_list[i]
        f = cs.Function(name, [x, u, s, w_sym], [y], ["x", "u", "s", "w"], ["y"])
        return f, {"mode": "mode_b_weights", "K": K}

    if export_cfg.mode == "mode_a_rbf":
        mu = cs.SX.sym("mu", 1)
        mu_centers = cs.DM(ensemble.mu_centers.detach().cpu().numpy().astype(np.float64))
        distances = cs.power(mu_centers - mu, 2)
        weights = cs.exp(-distances / (2.0 * (export_cfg.rbf_width ** 2)))
        weights = weights / (cs.sum1(weights) + 1e-12)
        y = cs.SX.zeros(3, 1)
        for i in range(K):
            y += weights[i] * y_list[i]
        f = cs.Function(name, [x, u, s, mu], [y], ["x", "u", "s", "mu"], ["y"])
        return f, {"mode": "mode_a_rbf", "K": K}

    raise ValueError(f"Unsupported export mode: {export_cfg.mode}")


class HSSDynamicsCallback(cs.Callback):
    """
    Python callback dynamics for CasADi:
        y = f(x,u)  where x in R^3, u in R^2
    Static params + mu are captured in the object.
    """

    def __init__(
        self,
        name: str,
        ensemble: HSSEnsemble,
        static_params: np.ndarray,
        mu_current: Optional[float] = None,
        use_mode_b_weights: bool = False,
        mode_b_weights: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        opts=None,
    ):
        cs.Callback.__init__(self)
        self._name = name
        self.ensemble = ensemble
        self.static_params = np.asarray(static_params, dtype=np.float32).reshape(2)
        self.mu_current = float(mu_current) if mu_current is not None else None
        self.use_mode_b_weights = bool(use_mode_b_weights)
        self.mode_b_weights = None if mode_b_weights is None else np.asarray(mode_b_weights, dtype=np.float32)
        self.top_k = None if top_k is None else int(top_k)
        self._jac_cb = None  # keep Jacobian callback alive (CasADi holds a weak ref in some paths)
        self.construct(name, opts or {})

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ["x", "u"][i]

    def get_name_out(self, i):
        return ["y"][i]

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(3, 1)
        return cs.Sparsity.dense(2, 1)

    def get_sparsity_out(self, i):
        return cs.Sparsity.dense(3, 1)

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        # Construct once and keep a strong reference to prevent GC during solver evaluation.
        if self._jac_cb is None:
            self._jac_cb = HSSJacobianCallback(
                name=f"jac_{self._name}",
                ensemble=self.ensemble,
                static_params=self.static_params,
                mu_current=self.mu_current,
                use_mode_b_weights=self.use_mode_b_weights,
                mode_b_weights=self.mode_b_weights,
                top_k=self.top_k,
                opts=opts,
            )
        return self._jac_cb

    def eval(self, arg):
        x = np.array(arg[0]).reshape(-1).astype(np.float32)
        u = np.array(arg[1]).reshape(-1).astype(np.float32)

        # Mode B: fixed weights provided by caller
        if self.use_mode_b_weights and self.mode_b_weights is not None:
            self.ensemble.set_weights(torch.from_numpy(self.mode_b_weights))
            mu = None
        else:
            mu = self.mu_current
        # Note: `predict_numpy` doesn't expose top_k; go through torch forward to enable Top-K gating.
        device = next(self.ensemble.parameters()).device
        x_t = torch.from_numpy(x).float().to(device).unsqueeze(0)
        u_t = torch.from_numpy(u).float().to(device).unsqueeze(0)
        s_t = torch.from_numpy(self.static_params).float().to(device).unsqueeze(0)
        with torch.no_grad():
            y_t = self.ensemble(x_t, u_t, s_t, mu_current=mu, top_k=self.top_k)
        y = y_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return [cs.DM(y).reshape((3, 1))]


class HSSJacobianCallback(cs.Callback):
    """
    Jacobian callback for HSSDynamicsCallback:
      returns J = d f / d [x;u] as a (3 x 5) dense matrix.
    """

    def __init__(
        self,
        name: str,
        ensemble: HSSEnsemble,
        static_params: np.ndarray,
        mu_current: Optional[float] = None,
        use_mode_b_weights: bool = False,
        mode_b_weights: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        opts=None,
    ):
        cs.Callback.__init__(self)
        self._name = name
        self.ensemble = ensemble
        self.static_params = np.asarray(static_params, dtype=np.float32).reshape(2)
        self.mu_current = float(mu_current) if mu_current is not None else None
        self.use_mode_b_weights = bool(use_mode_b_weights)
        self.mode_b_weights = None if mode_b_weights is None else np.asarray(mode_b_weights, dtype=np.float32)
        self.top_k = None if top_k is None else int(top_k)
        self.construct(name, opts or {})

    def get_n_in(self):
        # CasADi expects the Jacobian callback signature to include the output(s)
        # of the original callback as additional inputs: [x, u, y].
        return 3

    def get_n_out(self):
        # Return separate Jacobians for each input:
        #   jac_y_x: (3x3) and jac_y_u: (3x2)
        return 2

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(3, 1)  # x
        if i == 1:
            return cs.Sparsity.dense(2, 1)  # u
        return cs.Sparsity.dense(3, 1)      # y (unused)

    def get_sparsity_out(self, i):
        if i == 0:
            return cs.Sparsity.dense(3, 3)  # df/dx
        return cs.Sparsity.dense(3, 2)      # df/du

    def eval(self, arg):
        x = np.array(arg[0]).reshape(-1).astype(np.float32)
        u = np.array(arg[1]).reshape(-1).astype(np.float32)
        # arg[2] is y (callback output), unused

        x_t = torch.from_numpy(x)
        u_t = torch.from_numpy(u)
        s_t = torch.from_numpy(self.static_params)

        if self.use_mode_b_weights and self.mode_b_weights is not None:
            self.ensemble.set_weights(torch.from_numpy(self.mode_b_weights))
            mu = None
        else:
            mu = self.mu_current
        jac_x, jac_u = self.ensemble.jacobian_analytic(x_t, u_t, s_t, mu_current=mu, top_k=self.top_k)
        Jx = jac_x.detach().cpu().numpy().astype(np.float64)  # (3,3)
        Ju = jac_u.detach().cpu().numpy().astype(np.float64)  # (3,2)
        return [cs.DM(Jx), cs.DM(Ju)]


# -----------------------------------------------------------------------------
# Backwards-compatible names used by older MPC code in this repo
# -----------------------------------------------------------------------------

class S2GPTDynamics:
    """
    Compatibility wrapper expected by older `nmpc.py`.
    Provides a CasADi-callable object `dynamics(x, u) -> dxdt`.
    """

    def __init__(self, ensemble: HSSEnsemble, current_mu: float, static_params: np.ndarray, top_k: Optional[int] = None):
        self._cb = HSSDynamicsCallback(
            name="s2gpt_dyn_cb",
            ensemble=ensemble,
            static_params=static_params,
            mu_current=current_mu,
            top_k=top_k,
        )

    def __call__(self, x, u):
        return self._cb(x, u)


class S2GPTDynamicsFull(S2GPTDynamics):
    """
    Compatibility alias. Historically this distinguished Jacobian handling,
    but our callback already exposes Jacobians via `has_jacobian`.
    """
    pass


