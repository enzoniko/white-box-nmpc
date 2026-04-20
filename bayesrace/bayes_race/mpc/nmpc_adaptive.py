"""
Adaptive NMPC Setup for BayesRace with extra dynamic parameters.

This is a minimal extension of `bayes_race.mpc.nmpc.setupNLP` that adds an extra
parameter vector `dyn` into the NLP parameters `p` and forwards it to the model's
CasADi dynamics call:

  dxdt = model.casadi(x_k, u_k, dxdtc, dyn, k)

where k is the prediction step index (0..horizon-1); models may use it e.g. for
per-step additive noise (aleatoric).

This allows:
  - neural weights (Mode B) as dynamic parameters WITHOUT rebuilding the NLP
  - or other runtime parameters if desired
"""

from __future__ import annotations

import numpy as np
import casadi as cs


class setupNLPAdaptive:
    def __init__(self, horizon, Ts, Q, P, R, params, model, track, dyn_dim: int, track_cons: bool = False):
        self.horizon = horizon
        self.params = params
        self.model = model
        self.track = track
        self.track_cons = track_cons
        self.dyn_dim = int(dyn_dim)

        n_states = model.n_states
        n_inputs = model.n_inputs
        xref_size = 2

        x0 = cs.SX.sym("x0", n_states, 1)
        xref = cs.SX.sym("xref", xref_size, horizon + 1)
        uprev = cs.SX.sym("uprev", 2, 1)
        x = cs.SX.sym("x", n_states, horizon + 1)
        u = cs.SX.sym("u", n_inputs, horizon)
        dxdtc = cs.SX.sym("dxdt", n_states, 1)
        dyn = cs.SX.sym("dyn", self.dyn_dim, 1)

        cost_tracking = 0
        cost_actuation = 0
        cost_violation = 0

        cost_tracking += (x[:xref_size, -1] - xref[:xref_size, -1]).T @ P @ (x[:xref_size, -1] - xref[:xref_size, -1])
        constraints = x[:, 0] - x0

        for k in range(horizon):
            dxdt = model.casadi(x[:, k], u[:, k], dxdtc, dyn, k)
            constraints = cs.vertcat(constraints, x[:, k + 1] - x[:, k] - Ts * dxdt)

        for k in range(horizon):
            if k == 0:
                deltaU = u[:, k] - uprev
            else:
                deltaU = u[:, k] - u[:, k - 1]

            cost_tracking += (x[:xref_size, k + 1] - xref[:xref_size, k + 1]).T @ Q @ (x[:xref_size, k + 1] - xref[:xref_size, k + 1])
            cost_actuation += deltaU.T @ R @ deltaU

            constraints = cs.vertcat(constraints, u[:, k] - params["max_inputs"])
            constraints = cs.vertcat(constraints, -u[:, k] + params["min_inputs"])
            constraints = cs.vertcat(constraints, deltaU[1] - params["max_rates"][1] * Ts)
            constraints = cs.vertcat(constraints, -deltaU[1] + params["min_rates"][1] * Ts)

        cost = cost_tracking + cost_actuation + cost_violation

        xvars = cs.vertcat(cs.reshape(x, -1, 1), cs.reshape(u, -1, 1))
        pvars = cs.vertcat(cs.reshape(x0, -1, 1), cs.reshape(xref, -1, 1), cs.reshape(uprev, -1, 1), cs.reshape(dyn, -1, 1))

        nlp = {"x": xvars, "p": pvars, "f": cost, "g": constraints}
        options = {
            "expand": True,
            "print_time": False,
            "ipopt": {"print_level": 0, "max_iter": 100},
        }
        self.problem = cs.nlpsol("nmpc_adaptive", "ipopt", nlp, options)

    def solve(self, x0, xref, uprev, dyn):
        horizon = self.horizon
        dyn = np.asarray(dyn).reshape(-1, 1)

        arg = {}
        # Match BayesRace's `setupNLP.solve` packing: xref is provided as (2, N+1),
        # but BayesRace flattens xref.T (column-major time ordering).
        arg["p"] = np.concatenate(
            [
                x0.reshape(-1, 1),
                xref.T.reshape(-1, 1),
                uprev.reshape(-1, 1),
                dyn.reshape(-1, 1),
            ]
        )

        # Initial guess (zeros) is okay for benchmarking; users can warm start if desired.
        idx_x0 = self.problem.index_in("x0")
        n_dec = int(self.problem.size1_in(idx_x0))
        arg["x0"] = np.zeros((n_dec, 1))

        # Bounds match BayesRace NMPC:
        # - decision variables unbounded
        # - first n_states*(horizon+1) constraints are equality (dynamics + init)
        # - remaining constraints are inequalities with upper bound 0 and lower -inf
        n_states = self.model.n_states
        n_inputs = self.model.n_inputs
        arg["lbx"] = -np.inf * np.ones((n_states * (horizon + 1) + n_inputs * horizon,))
        arg["ubx"] = np.inf * np.ones((n_states * (horizon + 1) + n_inputs * horizon,))

        idx_g = self.problem.index_out("g")
        n_g = int(self.problem.size1_out(idx_g))
        n_eq = n_states * (horizon + 1)
        n_ineq = n_g - n_eq
        arg["lbg"] = np.concatenate([np.zeros((n_eq, 1)), -np.inf * np.ones((n_ineq, 1))])
        arg["ubg"] = np.concatenate([np.zeros((n_eq, 1)), np.zeros((n_ineq, 1))])

        sol = self.problem(**arg)
        wopt = np.array(sol["x"]).reshape(-1, 1)

        # Extract optimal u sequence
        n_states = self.model.n_states
        n_inputs = self.model.n_inputs
        offset_u = n_states * (horizon + 1)
        uopt = wopt[offset_u : offset_u + n_inputs * horizon].reshape(n_inputs, horizon, order="F")
        return uopt, float(sol["f"]), wopt


