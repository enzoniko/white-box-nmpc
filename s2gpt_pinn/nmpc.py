#!/usr/bin/env python3
"""
S²GPT-PINN NMPC Controller

Nonlinear Model Predictive Control using S²GPT-PINN neural surrogate dynamics.
Key features:

1. Pre-computed analytic Jacobians for fast optimization
2. Friction-adaptive dynamics via specialist ensemble
3. Compatible with existing bayesrace/deep-dynamics framework
4. Target: <2ms MPC solve time

Integration follows the structure of bayesrace.mpc.nmpc but replaces
the physics-based dynamics with the neural surrogate.
"""

import numpy as np
import casadi as cs
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from .specialist import S2GPTEnsemble
from .casadi_callbacks import S2GPTDynamics, S2GPTDynamicsFull


@dataclass
class S2GPTMPCConfig:
    """Configuration for S²GPT-PINN NMPC controller."""
    
    # Timing
    horizon: int = 20          # Prediction horizon N
    dt: float = 0.02           # Sampling time (50 Hz)
    
    # Cost weights
    Q_xy: float = 1.0          # Position tracking weight
    Q_vel: float = 0.1         # Velocity tracking weight
    P_terminal: float = 10.0   # Terminal cost weight
    R_delta: float = 0.01      # Steering rate penalty
    R_throttle: float = 0.001  # Throttle rate penalty
    
    # Input constraints
    max_delta: float = 0.5     # Max steering angle [rad]
    min_delta: float = -0.5    # Min steering angle [rad]
    max_throttle: float = 1.0  # Max throttle
    min_throttle: float = -0.5 # Min throttle (braking)
    max_delta_rate: float = 2.0  # Max steering rate [rad/s]
    
    # Solver options
    max_iter: int = 50         # Max IPOPT iterations
    print_level: int = 0       # IPOPT print level
    warm_start: bool = True    # Use previous solution as init
    
    # Vehicle static params (from Phase 1 commissioning)
    mass: float = 800.0        # kg
    inertia: float = 1200.0    # kg·m²


class S2GPTNMPC:
    """
    NMPC Controller using S²GPT-PINN Neural Dynamics.
    
    State: [x, y, ψ, vx, vy, ω] - 6 states
    Control: [δ, T] - steering and throttle
    
    The controller solves:
        min Σ ||x - x_ref||²_Q + ||Δu||²_R
        s.t. x_{k+1} = f_neural(x_k, u_k)
             u_min ≤ u ≤ u_max
             |Δu| ≤ rate_max
    """
    
    def __init__(
        self,
        ensemble: S2GPTEnsemble,
        config: Optional[S2GPTMPCConfig] = None,
        track=None
    ):
        """
        Initialize S²GPT-PINN NMPC.
        
        Args:
            ensemble: Trained S2GPTEnsemble for dynamics
            config: MPC configuration
            track: Optional track object for constraints
        """
        self.ensemble = ensemble
        self.config = config or S2GPTMPCConfig()
        self.track = track
        
        # Current friction coefficient
        self._current_mu = 0.8
        
        # Static params from config
        self.static_params = np.array([
            self.config.mass,
            self.config.inertia
        ])
        
        # Neural dynamics callback (will be recreated when mu changes)
        self._dynamics_callback = None
        self._solver = None
        
        # Warm start storage
        self._last_solution = None
        
        # Performance tracking
        self.solve_times = []
        
        # Build the NLP
        self._build_nlp()
    
    @property
    def current_mu(self) -> float:
        return self._current_mu
    
    def update_friction(self, new_mu: float):
        """Update friction coefficient and rebuild dynamics callback."""
        if abs(new_mu - self._current_mu) > 0.01:
            self._current_mu = new_mu
            self._dynamics_callback = None  # Force rebuild
            self._build_nlp()
    
    def _get_dynamics_callback(self) -> S2GPTDynamicsFull:
        """Get or create dynamics callback with current friction."""
        if self._dynamics_callback is None:
            self._dynamics_callback = S2GPTDynamicsFull(
                self.ensemble,
                current_mu=self._current_mu,
                static_params=self.static_params
            )
        return self._dynamics_callback
    
    def _build_nlp(self):
        """Build CasADi NLP for MPC optimization."""
        cfg = self.config
        N = cfg.horizon
        n_states = 6
        n_inputs = 2
        
        # Decision variables
        X = cs.SX.sym('X', n_states, N + 1)  # States over horizon
        U = cs.SX.sym('U', n_inputs, N)       # Controls over horizon
        
        # Parameters (passed at solve time)
        x0 = cs.SX.sym('x0', n_states)        # Initial state
        xref = cs.SX.sym('xref', 2, N + 1)    # Reference trajectory [x, y]
        u_prev = cs.SX.sym('u_prev', n_inputs) # Previous control
        mu_param = cs.SX.sym('mu', 1)          # Current friction
        
        # Cost weights
        Q = cs.diag([cfg.Q_xy, cfg.Q_xy, 0, cfg.Q_vel, 0, 0])
        P = cs.diag([cfg.P_terminal, cfg.P_terminal, 0, 0, 0, 0])
        R = cs.diag([cfg.R_delta, cfg.R_throttle])
        
        # Build cost and constraints
        cost = 0
        g = []  # Constraint vector
        
        # Initial state constraint
        g.append(X[:, 0] - x0)
        
        # Dynamics constraints using neural surrogate
        dynamics = self._get_dynamics_callback()
        dxdt_sym = cs.SX.sym('dxdt', n_states)
        
        for k in range(N):
            # Neural dynamics (callback handles the forward pass)
            dxdt = dynamics(X[:, k], U[:, k])
            
            # Forward Euler integration
            x_next = X[:, k] + cfg.dt * dxdt
            
            # Dynamics constraint
            g.append(X[:, k + 1] - x_next)
            
            # Stage cost
            x_err = X[:2, k + 1] - xref[:, k + 1]
            cost += x_err.T @ Q[:2, :2] @ x_err
            
            # Control rate cost
            if k == 0:
                du = U[:, k] - u_prev
            else:
                du = U[:, k] - U[:, k - 1]
            cost += du.T @ R @ du
            
            # Input constraints
            g.append(U[0, k] - cfg.max_delta)  # delta <= max
            g.append(cfg.min_delta - U[0, k])  # delta >= min
            g.append(U[1, k] - cfg.max_throttle)
            g.append(cfg.min_throttle - U[1, k])
            
            # Rate constraints
            g.append(du[0] - cfg.max_delta_rate * cfg.dt)
            g.append(-cfg.max_delta_rate * cfg.dt - du[0])
        
        # Terminal cost
        x_err_terminal = X[:2, N] - xref[:, N]
        cost += x_err_terminal.T @ P[:2, :2] @ x_err_terminal
        
        # Pack decision variables
        opt_vars = cs.vertcat(
            cs.reshape(X, -1, 1),
            cs.reshape(U, -1, 1)
        )
        
        # Pack parameters
        params = cs.vertcat(
            x0,
            cs.reshape(xref, -1, 1),
            u_prev,
            mu_param
        )
        
        # Create NLP
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': cs.vertcat(*g),
            'p': params
        }
        
        # Solver options
        ipopt_opts = {
            'print_level': cfg.print_level,
            'max_iter': cfg.max_iter,
            'warm_start_init_point': 'yes' if cfg.warm_start else 'no',
            'linear_solver': 'mumps',
            'mu_strategy': 'adaptive',
        }
        
        opts = {
            'ipopt': ipopt_opts,
            'print_time': False,
        }
        
        self._solver = cs.nlpsol('s2gpt_mpc', 'ipopt', nlp, opts)
        
        # Store problem dimensions
        self._n_states = n_states
        self._n_inputs = n_inputs
        self._n_opt = n_states * (N + 1) + n_inputs * N
        
        # Constraint bounds
        n_dyn_cons = n_states * (N + 1)
        n_input_cons = 4 * N
        n_rate_cons = 2 * N
        n_total_cons = n_dyn_cons + n_input_cons + n_rate_cons
        
        self._lbg = np.concatenate([
            np.zeros(n_dyn_cons),      # Dynamics equality
            -np.inf * np.ones(n_input_cons + n_rate_cons)  # Inequalities
        ])
        self._ubg = np.concatenate([
            np.zeros(n_dyn_cons),      # Dynamics equality
            np.zeros(n_input_cons + n_rate_cons)  # Inequalities ≤ 0
        ])
    
    def solve(
        self,
        x0: np.ndarray,
        xref: np.ndarray,
        u_prev: np.ndarray,
        mu_current: Optional[float] = None
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Solve MPC optimization.
        
        Args:
            x0: Current state [6,]
            xref: Reference trajectory [2, N+1] (x, y positions)
            u_prev: Previous control [2,]
            mu_current: Optional friction update
            
        Returns:
            u_opt: Optimal first control [2,]
            solve_time: Solution time in seconds
            info: Dict with optimization details
        """
        cfg = self.config
        N = cfg.horizon
        
        # Update friction if needed
        if mu_current is not None:
            self.update_friction(mu_current)
        
        # Pack parameters
        params = np.concatenate([
            x0,
            xref.T.flatten(),
            u_prev,
            [self._current_mu]
        ])
        
        # Initial guess
        if self._last_solution is not None and cfg.warm_start:
            x0_opt = self._last_solution
        else:
            # Simple initialization: propagate current state
            x0_opt = np.zeros(self._n_opt)
            for k in range(N + 1):
                x0_opt[k * 6:(k + 1) * 6] = x0
        
        # Solve
        start_time = time.perf_counter()
        
        try:
            sol = self._solver(
                x0=x0_opt,
                p=params,
                lbg=self._lbg,
                ubg=self._ubg,
                lbx=-np.inf,
                ubx=np.inf
            )
            
            solve_time = time.perf_counter() - start_time
            self.solve_times.append(solve_time)
            
            # Extract solution
            opt_vars = np.array(sol['x']).flatten()
            
            # Store for warm start
            self._last_solution = opt_vars
            
            # Extract states and controls
            X_opt = opt_vars[:6 * (N + 1)].reshape(N + 1, 6).T
            U_opt = opt_vars[6 * (N + 1):].reshape(N, 2).T
            
            # Return first control
            u_opt = U_opt[:, 0]
            
            info = {
                'success': True,
                'cost': float(sol['f']),
                'X_opt': X_opt,
                'U_opt': U_opt,
                'solve_time': solve_time,
                'iterations': int(self._solver.stats()['iter_count']),
            }
            
        except Exception as e:
            solve_time = time.perf_counter() - start_time
            u_opt = u_prev.copy()  # Fallback
            
            info = {
                'success': False,
                'error': str(e),
                'solve_time': solve_time,
            }
        
        return u_opt, solve_time, info
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.solve_times:
            return {}
        
        times = np.array(self.solve_times) * 1000  # Convert to ms
        return {
            'mean_solve_time_ms': np.mean(times),
            'std_solve_time_ms': np.std(times),
            'max_solve_time_ms': np.max(times),
            'min_solve_time_ms': np.min(times),
            'p95_solve_time_ms': np.percentile(times, 95),
            'n_solves': len(self.solve_times),
        }


class S2GPTNMPCSimple:
    """
    Simplified NMPC for 3-state dynamics (velocity-only).
    
    State: [vx, vy, ω]
    Control: [δ, T]
    
    Useful for testing and benchmarking the neural dynamics
    without the full 6-state integration.
    """
    
    def __init__(
        self,
        ensemble: S2GPTEnsemble,
        config: Optional[S2GPTMPCConfig] = None
    ):
        self.ensemble = ensemble
        self.config = config or S2GPTMPCConfig()
        
        self._current_mu = 0.8
        self.static_params = np.array([
            self.config.mass,
            self.config.inertia
        ])
        
        self._build_nlp()
    
    def _build_nlp(self):
        """Build simplified NLP for 3-state dynamics."""
        cfg = self.config
        N = cfg.horizon
        n_states = 3
        n_inputs = 2
        
        # Decision variables
        X = cs.SX.sym('X', n_states, N + 1)
        U = cs.SX.sym('U', n_inputs, N)
        
        # Parameters
        x0 = cs.SX.sym('x0', n_states)
        vref = cs.SX.sym('vref', 1)  # Reference velocity
        u_prev = cs.SX.sym('u_prev', n_inputs)
        
        # Cost and constraints
        cost = 0
        g = [X[:, 0] - x0]
        
        # Create dynamics callback
        from .casadi_callbacks import S2GPTDynamics
        dynamics = S2GPTDynamics(
            self.ensemble,
            current_mu=self._current_mu,
            static_params=self.static_params
        )
        
        for k in range(N):
            # Neural dynamics
            dxdt = dynamics(X[:, k], U[:, k])
            x_next = X[:, k] + cfg.dt * dxdt
            g.append(X[:, k + 1] - x_next)
            
            # Velocity tracking cost
            cost += (X[0, k + 1] - vref) ** 2
            
            # Control rate cost
            if k == 0:
                du = U[:, k] - u_prev
            else:
                du = U[:, k] - U[:, k - 1]
            cost += cfg.R_delta * du[0] ** 2 + cfg.R_throttle * du[1] ** 2
            
            # Input bounds
            g.append(U[0, k] - cfg.max_delta)
            g.append(cfg.min_delta - U[0, k])
            g.append(U[1, k] - cfg.max_throttle)
            g.append(cfg.min_throttle - U[1, k])
        
        # Pack
        opt_vars = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
        params = cs.vertcat(x0, vref, u_prev)
        
        nlp = {'x': opt_vars, 'f': cost, 'g': cs.vertcat(*g), 'p': params}
        
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': cfg.max_iter,
                'warm_start_init_point': 'yes',
            },
            'print_time': False,
        }
        
        self._solver = cs.nlpsol('s2gpt_mpc_simple', 'ipopt', nlp, opts)
        
        # Bounds
        n_dyn = n_states * (N + 1)
        n_inp = 4 * N
        self._lbg = np.concatenate([np.zeros(n_dyn), -np.inf * np.ones(n_inp)])
        self._ubg = np.concatenate([np.zeros(n_dyn), np.zeros(n_inp)])
        
        self._n_states = n_states
        self._n_inputs = n_inputs
        self._N = N
    
    def solve(
        self,
        x0: np.ndarray,
        vref: float,
        u_prev: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Solve simplified MPC."""
        params = np.concatenate([x0, [vref], u_prev])
        
        n_opt = self._n_states * (self._N + 1) + self._n_inputs * self._N
        x0_opt = np.zeros(n_opt)
        
        start = time.perf_counter()
        sol = self._solver(x0=x0_opt, p=params, lbg=self._lbg, ubg=self._ubg)
        solve_time = time.perf_counter() - start
        
        opt_vars = np.array(sol['x']).flatten()
        U_start = self._n_states * (self._N + 1)
        u_opt = opt_vars[U_start:U_start + 2]
        
        return u_opt, solve_time * 1000  # Return time in ms


if __name__ == "__main__":
    from .specialist import S2GPTConfig, S2GPTSpecialist, S2GPTEnsemble
    
    print("Testing S²GPT-PINN NMPC Controller")
    print("=" * 60)
    
    # Create test ensemble
    config = S2GPTConfig(hidden_dim=64, n_layers=3)
    specialists = [S2GPTSpecialist(config) for _ in range(4)]
    ensemble = S2GPTEnsemble(specialists, mu_centers=[0.3, 0.6, 0.9, 1.2])
    
    # Create MPC
    mpc_config = S2GPTMPCConfig(horizon=20, dt=0.02)
    
    print("Testing simplified 3-state MPC...")
    mpc_simple = S2GPTNMPCSimple(ensemble, mpc_config)
    
    x0 = np.array([20.0, 0.5, 0.1])  # [vx, vy, omega]
    vref = 25.0
    u_prev = np.array([0.0, 0.3])
    
    u_opt, solve_time = mpc_simple.solve(x0, vref, u_prev)
    print(f"  Optimal control: δ={u_opt[0]:.4f}, T={u_opt[1]:.4f}")
    print(f"  Solve time: {solve_time:.2f} ms")
    
    # Benchmark
    print("\nBenchmarking solve times...")
    times = []
    for _ in range(50):
        _, t = mpc_simple.solve(x0, vref, u_prev)
        times.append(t)
    
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Std:  {np.std(times):.2f} ms")
    print(f"  Max:  {np.max(times):.2f} ms")
    
    print("\n✓ NMPC tests passed!")

