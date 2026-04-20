#!/usr/bin/env python3
"""
generate_vtc_tracking_figure.py

Generates the "Transient Response & Adaptation" figure for the VTC paper.
Simulates a friction regime shift (Nominal -> Low Friction) and visualizes
the Neural Ensemble's adaptation via linear regression on finite differences.

BEST SCENARIO VERSION (CORRECTED):
- Uses exact physics (Dynamic models) instead of neural networks
- Implements 8+ specialists with different friction parameters
- Uses Governor module with simplex-constrained linear regression
- Implements forward Euler discretization as per paper

CORRECTIONS APPLIED:
1. Fixed Planner: Custom 'ConstantSpeedFixed' enforces v=1.5m/s (ignoring track's optimal profile)
2. Faster Adaptation: Governor window reduced from 40 -> 10 steps (0.8s -> 0.2s latency)
3. Fixed Logging: Logs actual target speed (1.5 m/s) instead of x-position

Scenario:
  - Duration: 5.0s
  - Event: At t=2.2s, friction drops by 40% (Asphalt -> Ice/Wet).
  - Controller: NMPC using a Weighted Ensemble Surrogate.
  - Estimator: Governor module solving simplex-constrained linear regression.

Usage:
  python3 generate_vtc_tracking_figure.py
"""

import time
import numpy as np
import casadi as cs
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque
from scipy.optimize import minimize

# --- Import BayesRace Modules ---
from bayes_race.params import ORCA
from bayes_race.models import Dynamic
from bayes_race.models.model import Model
from bayes_race.tracks import ETHZ
from bayes_race.mpc.nmpc_adaptive import setupNLPAdaptive


# --- Slalom Planner (Straight Path) ---
def SlalomReferenceStraight(x0, v0, N, Ts, freq=0.8, amp=0.5, heading=0.0, phase_offset=0.0):
    """
    Generates a TRUE slalom reference: sine wave on a straight line.
    This ensures the slalom frequency is independent of track geometry.
    
    Args:
        x0: Current position (2,) [x, y]
        v0: Target speed (m/s)
        N: Horizon length
        Ts: Time step
        freq: Frequency of slalom (Hz) - oscillations per second
        amp: Amplitude of slalom (meters)
        heading: Direction of straight path (radians)
        phase_offset: Current phase (radians) for continuity
    
    Returns:
        xref: Reference trajectory (2, N+1)
        phase_offset: Updated phase for next call
    """
    xref = np.zeros([2, N+1])
    xref[:2, 0] = x0[:2]  # Start at current position
    
    # Initialize phase
    if phase_offset == 0.0:
        continuous_phase = 0.0  # Start from zero for first call
    else:
        continuous_phase = phase_offset
    
    # Direction vectors (forward and lateral)
    forward_x = np.cos(heading)
    forward_y = np.sin(heading)
    lateral_x = -np.sin(heading)  # Perpendicular (left)
    lateral_y = np.cos(heading)
    
    # Compute reference for point 0 (current time step)
    offset = amp * np.sin(continuous_phase)
    xref[0, 0] = x0[0] + lateral_x * offset
    xref[1, 0] = x0[1] + lateral_y * offset
    
    for idh in range(1, N+1):
        # Advance phase based on TIME (not distance) - this is the key fix!
        continuous_phase += 2 * np.pi * freq * Ts
        
        # Compute lateral offset from sine wave
        offset = amp * np.sin(continuous_phase)
        
        # Advance along straight path
        distance_forward = v0 * Ts * idh
        
        # Position = start + forward_distance + lateral_offset
        xref[0, idh] = x0[0] + forward_x * distance_forward + lateral_x * offset
        xref[1, idh] = x0[1] + forward_y * distance_forward + lateral_y * offset
    
    return xref, continuous_phase


# --- Custom Planner to Enforce Constant Speed ---
def ConstantSpeedFixed(x0, v0, track, N, Ts, projidx):
    """
    Generates a reference trajectory at EXACTLY v0, ignoring the track's 
    internal speed profile.
    
    This fixes the issue where ConstantSpeed uses track.spline_v.calc(dist)
    which returns optimal racing speeds instead of respecting v0.
    """
    # Project x0 onto raceline
    raceline = track.raceline
    # Use a small window for projection to be fast
    xy, idx = track.project_fast(x=x0[0], y=x0[1], raceline=raceline[:, projidx:projidx+20])
    projidx = idx + projidx

    xref = np.zeros([2, N+1])

    # Get current arc length 's' at the projection point
    # The spline.s array contains arc lengths for each point on the raceline
    if hasattr(track, 'spline') and hasattr(track.spline, 's'):
        # Ensure projidx is within bounds
        if projidx >= len(track.spline.s):
            projidx = projidx % len(track.spline.s)
        current_s = track.spline.s[projidx]
        track_length = track.spline.s[-1]
    else:
        # Fallback: estimate from raceline distance
        dist0 = np.sum(np.linalg.norm(np.diff(raceline[:, :projidx+1], axis=1), 2, axis=0))
        current_s = dist0
        track_length = np.sum(np.linalg.norm(np.diff(raceline, axis=1), 2, axis=0))
    
    # Compute reference for point 0 (current time step) - project onto centerline
    if hasattr(track, 'spline') and hasattr(track.spline, 'calc_position'):
        pos = track.spline.calc_position(current_s)
        xref[0, 0] = pos[0]
        xref[1, 0] = pos[1]
    else:
        # Fallback: interpolate from raceline
        dists = np.cumsum(np.linalg.norm(np.diff(raceline, axis=1), 2, axis=0))
        closest_idx = np.argmin(np.abs(dists - current_s))
        xref[0, 0] = raceline[0, closest_idx]
        xref[1, 0] = raceline[1, closest_idx]
    
    for idh in range(1, N+1):
        # Advance distance by constant velocity v0
        current_s += v0 * Ts
        # Wrap around track length
        current_s = current_s % track_length
        
        # Get (x,y) from spline at this distance
        if hasattr(track, 'spline') and hasattr(track.spline, 'calc_position'):
            pos = track.spline.calc_position(current_s)
            xref[0, idh] = pos[0]
            xref[1, idh] = pos[1]
        else:
            # Fallback: interpolate from raceline
            # Find closest point on raceline
            dists = np.cumsum(np.linalg.norm(np.diff(raceline, axis=1), 2, axis=0))
            closest_idx = np.argmin(np.abs(dists - current_s))
            xref[0, idh] = raceline[0, closest_idx]
            xref[1, idh] = raceline[1, closest_idx]
        
    return xref, projidx


class GovernorEMA:
    """
    Governor module with Exponential Moving Average (EMA) smoothing.
    Reduces weight oscillation and provides more stable adaptation.
    
    Implements the approach from the paper with EMA smoothing:
    min_w ||x_k - x_{k-1} - dt * sum_i w_i * Psi_i(x_{k-1}, u_{k-1})||^2
    subject to sum_i w_i = 1, w_i >= 0
    
    Then applies EMA: w_new = (1-alpha) * w_old + alpha * w_optimized
    """
    
    def __init__(self, n_specialists, window_size=10, alpha=0.2, regularization=1e-3):
        """
        Args:
            n_specialists: Number of specialists
            window_size: Size of sliding window H_k
            alpha: EMA smoothing factor (0.0 = frozen, 1.0 = no smoothing)
                   Lower alpha = more smoothing (0.2 means keep 80% old, add 20% new)
            regularization: Regularization parameter for numerical stability
        """
        self.n_specialists = n_specialists
        self.window_size = window_size
        self.alpha = alpha
        self.regularization = regularization
        
        # Sliding window: each element is h_k = (x_k, x_{k-1}, u_{k-1})
        self.history = deque(maxlen=window_size)
        
        # Current weights (with EMA smoothing)
        self.weights = np.ones(n_specialists) / n_specialists
    
    def add_measurement(self, x_k, x_k_minus_1, u_k_minus_1):
        """Add a measurement to the sliding window."""
        self.history.append((x_k.copy(), x_k_minus_1.copy(), u_k_minus_1.copy()))
    
    def update_weights(self, specialists, dt):
        """Update weights with EMA smoothing."""
        if len(self.history) < 3:  # Need at least 3 measurements
            return self.weights
        
        # Build regression problem
        n_obs = len(self.history)
        n_dims = 3
        Phi = np.zeros((n_obs * n_dims, self.n_specialists))
        y = np.zeros(n_obs * n_dims)
        
        for i, (x_k, x_k_minus_1, u_k_minus_1) in enumerate(self.history):
            dx_meas = (x_k[3:6] - x_k_minus_1[3:6]) / dt
            y[i * n_dims:(i + 1) * n_dims] = dx_meas
            for j, specialist in enumerate(specialists):
                dx_pred = specialist._diffequation(None, x_k_minus_1, u_k_minus_1)
                Phi[i * n_dims:(i + 1) * n_dims, j] = dx_pred[3:6]
        
        def objective(w):
            return np.sum((Phi @ w - y) ** 2) + self.regularization * np.sum(w ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(self.n_specialists)]
        
        try:
            result = minimize(objective, self.weights, method='SLSQP', 
                             bounds=bounds, constraints=constraints,
                             options={'maxiter': 1000, 'ftol': 1e-9})
            if result.success:
                raw_weights = result.x
                raw_weights = np.maximum(raw_weights, 0.0)
                raw_weights = raw_weights / (np.sum(raw_weights) + 1e-10)
                # Apply EMA smoothing
                self.weights = (1 - self.alpha) * self.weights + self.alpha * raw_weights
        except:
            pass
        
        return self.weights
    
    def get_weights(self):
        """Get current weights."""
        return self.weights.copy()


class Governor:
    """
    Governor module that estimates mixing weights by solving a simplex-constrained
    linear regression problem over a sliding window of recent measurements.
    
    Implements the approach from the paper:
    min_w ||x_k - x_{k-1} - dt * sum_i w_i * Psi_i(x_{k-1}, u_{k-1})||^2
    subject to sum_i w_i = 1, w_i >= 0
    """
    
    def __init__(self, n_specialists, window_size=10, regularization=1e-4):
        """
        Initialize Governor.
        
        Args:
            n_specialists: Number of specialists
            window_size: Size of sliding window H_k (reduced to 10 for faster adaptation)
            regularization: Regularization parameter for numerical stability
        """
        self.n_specialists = n_specialists
        self.window_size = window_size
        self.regularization = regularization
        
        # Sliding window: each element is h_k = (x_k, x_{k-1}, u_{k-1})
        self.history = deque(maxlen=window_size)
        
        # Current weights
        self.weights = np.ones(n_specialists) / n_specialists
    
    def add_measurement(self, x_k, x_k_minus_1, u_k_minus_1):
        """
        Add a measurement to the sliding window.
        
        Args:
            x_k: Current state (6,)
            x_k_minus_1: Previous state (6,)
            u_k_minus_1: Previous control input (2,)
        """
        self.history.append((x_k.copy(), x_k_minus_1.copy(), u_k_minus_1.copy()))
    
    def update_weights(self, specialists, dt):
        """
        Update weights by solving the simplex-constrained linear regression.
        For GovernorEMA, applies EMA smoothing after optimization.
        
        Args:
            specialists: List of Dynamic model specialists
            dt: Time step
        
        Returns:
            Updated weights (n_specialists,)
        """
        if len(self.history) < 2:
            # Not enough data, return uniform weights
            return self.weights
        
        # Build the regression problem
        # For each h_k in H_k, we have:
        # x_k - x_{k-1} = dt * sum_i w_i * Psi_i(x_{k-1}, u_{k-1})
        
        # We only use the dynamic states [vx, vy, omega] (indices 3,4,5)
        n_obs = len(self.history)
        n_dims = 3  # vx, vy, omega
        
        # Build design matrix Phi and target vector y
        Phi = np.zeros((n_obs * n_dims, self.n_specialists))
        y = np.zeros(n_obs * n_dims)
        
        for i, (x_k, x_k_minus_1, u_k_minus_1) in enumerate(self.history):
            # Target: measured state change (finite difference)
            dx_meas = (x_k[3:6] - x_k_minus_1[3:6]) / dt  # Acceleration
            
            # Get predictions from each specialist
            for j, specialist in enumerate(specialists):
                # Compute specialist prediction: Psi_j(x_{k-1}, u_{k-1})
                dx_pred = specialist._diffequation(None, x_k_minus_1, u_k_minus_1)
                # Only dynamic states
                Phi[i * n_dims:(i + 1) * n_dims, j] = dx_pred[3:6]
            
            # Target vector
            y[i * n_dims:(i + 1) * n_dims] = dx_meas
        
        # Solve: min_w ||Phi @ w - y||^2 subject to sum(w) = 1, w >= 0
        # Using scipy.optimize.minimize with constraints
        
        def objective(w):
            """Objective function: ||Phi @ w - y||^2 + regularization * ||w||^2"""
            residual = Phi @ w - y
            return np.sum(residual ** 2) + self.regularization * np.sum(w ** 2)
        
        # Constraints: sum(w) = 1, w >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0) for _ in range(self.n_specialists)]
        
        # Initial guess: current weights or uniform
        w0 = self.weights.copy()
        
        # Solve optimization
        try:
            result = minimize(
                objective,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                raw_weights = result.x
                # Ensure non-negative and normalized
                raw_weights = np.maximum(raw_weights, 0.0)
                raw_weights = raw_weights / (np.sum(raw_weights) + 1e-10)
                self.weights = raw_weights
            else:
                # If optimization fails, keep previous weights
                pass
        except Exception as e:
            # Keep previous weights on error
            pass
        
        return self.weights
    
    def get_weights(self):
        """Get current weights."""
        return self.weights.copy()


        """
        Add a measurement to the sliding window.
        
        Args:
            x_k: Current state (6,)
            x_k_minus_1: Previous state (6,)
            u_k_minus_1: Previous control input (2,)
        """
        self.history.append((x_k.copy(), x_k_minus_1.copy(), u_k_minus_1.copy()))
    
    def update_weights(self, specialists, dt):
        """
        Update weights by solving the simplex-constrained linear regression.
        
        Args:
            specialists: List of Dynamic model specialists
            dt: Time step
        
        Returns:
            Updated weights (n_specialists,)
        """
        if len(self.history) < 2:
            # Not enough data, return uniform weights
            return self.weights
        
        # Build the regression problem
        # For each h_k in H_k, we have:
        # x_k - x_{k-1} = dt * sum_i w_i * Psi_i(x_{k-1}, u_{k-1})
        
        # We only use the dynamic states [vx, vy, omega] (indices 3,4,5)
        n_obs = len(self.history)
        n_dims = 3  # vx, vy, omega
        
        # Build design matrix Phi and target vector y
        Phi = np.zeros((n_obs * n_dims, self.n_specialists))
        y = np.zeros(n_obs * n_dims)
        
        for i, (x_k, x_k_minus_1, u_k_minus_1) in enumerate(self.history):
            # Target: measured state change (finite difference)
            dx_meas = (x_k[3:6] - x_k_minus_1[3:6]) / dt  # Acceleration
            
            # Get predictions from each specialist
            for j, specialist in enumerate(specialists):
                # Compute specialist prediction: Psi_j(x_{k-1}, u_{k-1})
                dx_pred = specialist._diffequation(None, x_k_minus_1, u_k_minus_1)
                # Only dynamic states
                Phi[i * n_dims:(i + 1) * n_dims, j] = dx_pred[3:6]
            
            # Target vector
            y[i * n_dims:(i + 1) * n_dims] = dx_meas
        
        # Solve: min_w ||Phi @ w - y||^2 subject to sum(w) = 1, w >= 0
        # Using scipy.optimize.minimize with constraints
        
        def objective(w):
            """Objective function: ||Phi @ w - y||^2 + regularization * ||w||^2"""
            residual = Phi @ w - y
            return np.sum(residual ** 2) + self.regularization * np.sum(w ** 2)
        
        # Constraints: sum(w) = 1, w >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0) for _ in range(self.n_specialists)]
        
        # Initial guess: current weights or uniform
        w0 = self.weights.copy()
        
        # Solve optimization
        try:
            result = minimize(
                objective,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                self.weights = result.x
                # Ensure non-negative and normalized (safety check)
                self.weights = np.maximum(self.weights, 0.0)
                self.weights = self.weights / (np.sum(self.weights) + 1e-10)
            else:
                # If optimization fails, keep previous weights
                print(f"Warning: Governor optimization failed, keeping previous weights")
        except Exception as e:
            print(f"Warning: Governor optimization error: {e}, keeping previous weights")
        
        return self.weights
    
    def get_weights(self):
        """Get current weights."""
        return self.weights.copy()


class AdaptiveModel(Model):
    """
    Adaptive Model that combines multiple Dynamic specialists using weighted ensemble.
    Uses exact physics (Dynamic models) instead of neural networks.
    """
    
    def __init__(self, specialists):
        """
        Initialize AdaptiveModel.
        
        Args:
            specialists: List of Dynamic model specialists
        """
        super().__init__()
        self.n_states = 6
        self.n_inputs = 2
        self.specialists = specialists
        self.n_specialists = len(specialists)
        
        # Build CasADi function for the weighted ensemble
        self._build_casadi()
    
    def _build_casadi(self):
        """Build CasADi symbolic function for weighted ensemble."""
        x = cs.SX.sym('x', 6)
        u = cs.SX.sym('u', 2)
        w = cs.SX.sym('w', self.n_specialists)  # Weights for specialists
        
        # Get derivatives from all specialists
        dx_specialists = []
        for specialist in self.specialists:
            dx = specialist.casadi(x, u, cs.SX.zeros(6))
            dx_specialists.append(dx)
        
        # Weighted sum for dynamic states [vx, vy, omega] (indices 3,4,5)
        # Kinematics [x, y, psi] (indices 0,1,2) are shared (use first specialist)
        accel_combined = cs.SX.zeros(3)
        for i, dx in enumerate(dx_specialists):
            accel_combined += w[i] * dx[3:6]
        
        # Combine: kinematics from first specialist, dynamics from weighted ensemble
        dxdt_eq = cs.vertcat(
            dx_specialists[0][0:3],      # Kinematics (shared)
            accel_combined               # Dynamics (weighted)
        )
        
        self._f_comb = cs.Function('f_comb', [x, u, w], [dxdt_eq])
    
    def casadi(self, x, u, dxdt, dyn_params=None, step_index=None):
        """
        CasADi interface compatible with setupNLPAdaptive.
        
        Args:
            x: State (6,) - CasADi SX or numpy
            u: Control (2,) - CasADi SX or numpy
            dxdt: Output derivative (6,) - CasADi SX
            dyn_params: Dynamic parameters (weights) (n_specialists,) - CasADi SX or numpy
            step_index: Optional step index (for per-step noise; unused here).
        
        Returns:
            dxdt: State derivative
        """
        # Handle both symbolic (during build) and numeric (during solve) cases
        if dyn_params is None:
            # Default: uniform weights (for symbolic, we'll use a placeholder)
            # During build, dyn_params should always be provided
            w = cs.SX.ones(self.n_specialists) / self.n_specialists
        else:
            # Check if it's a CasADi symbolic type
            if isinstance(dyn_params, (cs.SX, cs.MX, cs.DM)):
                w = dyn_params
                # Normalize symbolic weights to ensure sum = 1 and non-negative
                # For symbolic, we use a soft normalization
                w = cs.fmax(w, 0.0)  # Non-negative
                w_sum = cs.sum1(w) + 1e-10
                w = w / w_sum
            else:
                # Numeric case (during solve)
                w = np.asarray(dyn_params).flatten()
                # Ensure valid weights
                w = np.maximum(w, 0.0)
                w = w / (np.sum(w) + 1e-10)
        
        res = self._f_comb(x, u, w)
        dxdt[0] = res[0]
        dxdt[1] = res[1]
        dxdt[2] = res[2]
        dxdt[3] = res[3]
        dxdt[4] = res[4]
        dxdt[5] = res[5]
        return dxdt


def create_specialists(base_params, n_specialists=8, friction_range=(0.3, 1.1)):
    """
    Create multiple specialists with different friction parameters.
    
    Args:
        base_params: Base ORCA parameters
        n_specialists: Number of specialists to create
        friction_range: (min_scale, max_scale) for friction coefficients
    
    Returns:
        List of Dynamic models with different friction parameters
    """
    specialists = []
    friction_scales = np.linspace(friction_range[0], friction_range[1], n_specialists)
    
    for i, scale in enumerate(friction_scales):
        # Create a copy of base params
        params = base_params.copy()
        
        # Scale friction coefficients Df and Dr
        params['Df'] = base_params['Df'] * scale
        params['Dr'] = base_params['Dr'] * scale
        
        # Create Dynamic model
        specialist = Dynamic(
            lf=params['lf'],
            lr=params['lr'],
            mass=params['mass'],
            Iz=params['Iz'],
            Cf=params['Cf'],
            Cr=params['Cr'],
            Bf=params['Bf'],
            Br=params['Br'],
            Df=params['Df'],
            Dr=params['Dr'],
            Cm1=params['Cm1'],
            Cm2=params['Cm2'],
            Cr0=params['Cr0'],
            Cr2=params['Cr2']
        )
        specialists.append(specialist)
        print(f"  Specialist {i}: friction scale = {scale:.3f} (Df={params['Df']:.4f}, Dr={params['Dr']:.4f})")
    
    return specialists, friction_scales


# --- Simulation Engine ---

def run_simulation(label, specialists, friction_scales, adaptive_mode=True, use_slalom=False):
    """
    Run a single simulation pass.
    
    Args:
        label: Label for logging (e.g., "Adaptive" or "Baseline")
        specialists: List of Dynamic model specialists
        friction_scales: Array of friction scales for each specialist
        adaptive_mode: If True, use Governor for weight updates. If False, use fixed weights.
        use_slalom: If True, use slalom reference. If False, use constant speed centerline.
    
    Returns:
        Dictionary with logged data
    """
    # Configuration
    T_SIM = 5.0
    DT = 0.02
    N_STEPS = int(T_SIM / DT)
    
    if use_slalom:
        CHANGE_TIME = 1.0  # Early change to catch the slalom
        V_REF = 2.1  # Higher speed for limit handling
        HORIZON = 25  # Extended horizon for better anticipation on low friction
    else:
        CHANGE_TIME = 2.2
        V_REF = 1.5  # Moderate speed
        HORIZON = 15  # Standard horizon
    
    CHANGE_STEP = int(CHANGE_TIME / DT)
    
    # Costs - Higher tracking weight for slalom to force precision
    if use_slalom:
        Q = np.diag([30.0, 30.0])  # Higher weight for aggressive tracking
    else:
        Q = np.diag([10.0, 10.0])
    P = np.diag([0.0, 0.0])    # Terminal
    R = np.diag([0.1, 1.0])    # Control [acc, steer]
    
    # Parameters (ORCA 1:43 scale)
    params_nom = ORCA(control='pwm')
    
    # Define Regime Shift (Asphalt -> Low Friction)
    if use_slalom:
        FRICTION_FACTOR = 0.5  # More severe drop (50% = Ice)
    else:
        FRICTION_FACTOR = 0.6  # 40% drop
    params_slip = ORCA(control='pwm')
    params_slip['Df'] *= FRICTION_FACTOR
    params_slip['Dr'] *= FRICTION_FACTOR
    
    # Track
    track = ETHZ(reference='optimal', longer=True)
    
    # Setup Models
    model_plant = Dynamic(**params_nom)
    model_surrogate = AdaptiveModel(specialists)
    
    # NMPC Controller
    nlp = setupNLPAdaptive(
        HORIZON, DT, Q, P, R,
        params_nom, model_surrogate, track,
        dyn_dim=len(specialists),
        track_cons=False
    )
    
    # Initialize State
    x0 = np.zeros(6)
    x0[0], x0[1] = track.x_init, track.y_init
    x0[2] = track.psi_init
    x0[3] = V_REF  # Start with target speed
    
    u_prev = np.zeros((2, 1))
    projidx = 0
    phase_offset = 0.0  # Initialize phase offset for slalom continuity
    
    # Store initial heading for straight slalom (fixed direction)
    slalom_heading = x0[2] if use_slalom else None
    
    # Set Initial Weights
    # Find index closest to 1.0 (Asphalt/Nominal)
    idx_nom = np.argmin(np.abs(friction_scales - 1.0))
    
    if adaptive_mode:
        if use_slalom:
            # For slalom, start with slight bias toward nominal but allow adaptation
            weights = np.zeros(len(specialists))
            weights[idx_nom] = 0.5  # Bias towards nominal start
            weights += 0.5 / len(specialists)  # Distribute remainder
            weights = weights / np.sum(weights)  # Normalize
            # Use EMA Governor with strong smoothing (alpha=0.2)
            governor = GovernorEMA(
                n_specialists=len(specialists),
                window_size=10,  # Smaller window for faster response
                alpha=0.2,  # Strong smoothing (keep 80% old, add 20% new)
                regularization=1e-3
            )
        else:
            # Adaptive mode: Start uniform, let Governor update
            weights = np.ones(len(specialists)) / len(specialists)
            governor = Governor(
                n_specialists=len(specialists),
                window_size=10,
                regularization=1e-4
            )
    else:
        # Baseline mode: Fixed to "Asphalt" specialist
        weights = np.zeros(len(specialists))
        weights[idx_nom] = 1.0
        governor = None
    
    # Logging
    log = {
        't': [],
        'vx': [],
        'vy': [],
        'psi': [],
        'w': [],
        'ref_vx': [],
        'x': [],
        'y': [],
        'ref_x': [],
        'ref_y': []
    }
    
    scenario_type = "Slalom @ 2.1m/s" if use_slalom else "Constant Speed @ 1.5m/s"
    print(f"\nRunning {label} Simulation ({scenario_type})...")
    if use_slalom:
        print(f"  Slalom: freq=0.8Hz, amp=0.5m, heading={np.degrees(slalom_heading):.1f}°")
    print(f"  Initial weights: {weights}")
    if not adaptive_mode:
        print(f"  Fixed to specialist {idx_nom} (μ={friction_scales[idx_nom]:.2f})")
    
    for k in range(N_STEPS):
        t_curr = k * DT
        
        # Regime Shift
        if k == CHANGE_STEP:
            print(f"  [Event] t={t_curr:.2f}s: Friction dropped (Plant Df/Dr *= {FRICTION_FACTOR})")
            model_plant = Dynamic(**params_slip)
        
        # Planner (Slalom or Constant Speed)
        if use_slalom:
            # Use straight slalom with time-based frequency (Hz)
            xref, phase_offset = SlalomReferenceStraight(
                x0=x0[:2], v0=V_REF, N=HORIZON, Ts=DT,
                freq=0.8,  # Hz (oscillations per second)
                amp=0.5,   # meters
                heading=slalom_heading,  # Fixed heading for straight path
                phase_offset=phase_offset
            )
        else:
            xref, projidx = ConstantSpeedFixed(
                x0=x0[:2], v0=V_REF, track=track, N=HORIZON, Ts=DT, projidx=projidx
            )
        
        # Control (NMPC)
        try:
            u_opt, fval, x_pred = nlp.solve(x0, xref[:2, :], u_prev, dyn=weights.reshape(-1, 1))
            u_apply = u_opt[:, 0].reshape(2, 1)
        except RuntimeError:
            if k % 50 == 0:  # Only print occasionally
                print(f"  Solver failed at k={k} (t={t_curr:.2f}s)")
            u_apply = np.array([[-0.5], [0]])
        
        # Plant Step
        x_traj, _ = model_plant.sim_continuous(x0, u_apply, [0, DT])
        x_next = x_traj[:, -1]
        
        # Adaptation (only if adaptive mode)
        if adaptive_mode and governor is not None:
            governor.add_measurement(x_next, x0, u_apply.flatten())
            if k > 2:  # Need at least 2-3 previous measurements
                weights = governor.update_weights(specialists, DT)
        
        # Logging
        log['t'].append(t_curr)
        log['vx'].append(x0[3])
        log['vy'].append(x0[4])
        log['psi'].append(x0[2])
        log['w'].append(weights.copy())
        log['ref_vx'].append(V_REF)
        log['x'].append(x0[0])
        log['y'].append(x0[1])
        # Log reference at current step (horizon point 0) for tracking comparison
        log['ref_x'].append(xref[0, 0])
        log['ref_y'].append(xref[1, 0])
        
        # Update State
        x0 = x_next
        u_prev = u_apply
    
    return log


def calculate_metrics(log, change_time, label):
    """
    Calculate RMSE metrics for vx, vy, and psi.
    Split into Pre-Event and Post-Event periods.
    
    Args:
        log: Dictionary with logged data
        change_time: Time when friction change occurs
        label: Label for printing
    
    Returns:
        Dictionary with metrics
    """
    t = np.array(log['t'])
    vx = np.array(log['vx'])
    vy = np.array(log['vy'])
    psi = np.array(log['psi'])
    
    # Masks
    mask_pre = t <= change_time
    mask_post = t > change_time
    
    # Reference values
    ref_vx = 1.5
    ref_vy = 0.0
    # For psi, we use the initial value as reference (lane keeping)
    ref_psi = psi[0] if len(psi) > 0 else 0.0
    
    # Calculate RMSE
    metrics = {}
    
    # Pre-Event
    metrics['rmse_vx_pre'] = np.sqrt(np.mean((vx[mask_pre] - ref_vx) ** 2))
    metrics['rmse_vy_pre'] = np.sqrt(np.mean((vy[mask_pre] - ref_vy) ** 2))
    metrics['rmse_psi_pre'] = np.sqrt(np.mean((psi[mask_pre] - ref_psi) ** 2))
    
    # Post-Event
    metrics['rmse_vx_post'] = np.sqrt(np.mean((vx[mask_post] - ref_vx) ** 2))
    metrics['rmse_vy_post'] = np.sqrt(np.mean((vy[mask_post] - ref_vy) ** 2))
    metrics['rmse_psi_post'] = np.sqrt(np.mean((psi[mask_post] - ref_psi) ** 2))
    
    # Print metrics
    print(f"\n--- {label} Metrics ---")
    print(f"Pre-Event  (t <= {change_time:.1f}s):")
    print(f"  RMSE vx:  {metrics['rmse_vx_pre']:.4f} m/s")
    print(f"  RMSE vy:  {metrics['rmse_vy_pre']:.4f} m/s")
    print(f"  RMSE psi: {metrics['rmse_psi_pre']:.4f} rad")
    print(f"Post-Event (t > {change_time:.1f}s):")
    print(f"  RMSE vx:  {metrics['rmse_vx_post']:.4f} m/s")
    print(f"  RMSE vy:  {metrics['rmse_vy_post']:.4f} m/s")
    print(f"  RMSE psi: {metrics['rmse_psi_post']:.4f} rad")
    
    return metrics


# --- Main Simulation ---

def calculate_path_rmse(log, t_split):
    """
    Calculate path tracking RMSE (Euclidean distance from reference path).
    
    Args:
        log: Dictionary with logged data
        t_split: Time to split pre/post event
    
    Returns:
        Dictionary with pre and post event RMSE
    """
    t = np.array(log['t'])
    x = np.array(log['x'])
    y = np.array(log['y'])
    rx = np.array(log['ref_x'])
    ry = np.array(log['ref_y'])
    
    # Euclidean distance from reference
    dist = np.sqrt((x - rx)**2 + (y - ry)**2)
    
    mask_pre = t <= t_split
    mask_post = t > t_split
    
    return {
        'pre': np.mean(dist[mask_pre]),
        'post': np.mean(dist[mask_post]),
        'all': np.mean(dist)
    }


def main():
    # Setup Specialists (shifted lower to cover ice well)
    print("Creating specialists with exact physics...")
    params_nom = ORCA(control='pwm')
    specialists, friction_scales = create_specialists(
        params_nom,
        n_specialists=8,
        friction_range=(0.3, 1.1)  # Range shifted lower to handle 0.5 drop well
    )
    
    # Run limit handling scenario (Slalom)
    print("\n" + "="*60)
    print("LIMIT HANDLING SCENARIO: Slalom on Ice")
    print("="*60)
    CHANGE_TIME_SLALOM = 1.0  # Early change
    
    # Run 1: Adaptive (Slalom)
    log_adapt_slalom = run_simulation(
        "Adaptive", specialists, friction_scales, 
        adaptive_mode=True, use_slalom=True
    )
    
    # Run 2: Baseline (Slalom, Fixed Asphalt)
    log_base_slalom = run_simulation(
        "Baseline", specialists, friction_scales, 
        adaptive_mode=False, use_slalom=True
    )
    
    # Calculate path tracking metrics
    path_rmse_adapt = calculate_path_rmse(log_adapt_slalom, CHANGE_TIME_SLALOM)
    path_rmse_base = calculate_path_rmse(log_base_slalom, CHANGE_TIME_SLALOM)
    
    print(f"\n--- RESULTS (Slalom High Speed) ---")
    print(f"Adaptive Path RMSE:")
    print(f"  Pre-Event:  {path_rmse_adapt['pre']:.4f} m")
    print(f"  Post-Event: {path_rmse_adapt['post']:.4f} m")
    print(f"  Overall:    {path_rmse_adapt['all']:.4f} m")
    print(f"Baseline Path RMSE:")
    print(f"  Pre-Event:  {path_rmse_base['pre']:.4f} m")
    print(f"  Post-Event: {path_rmse_base['post']:.4f} m")
    print(f"  Overall:    {path_rmse_base['all']:.4f} m")
    
    if path_rmse_base['post'] > 1e-6:  # Avoid division by zero
        improvement = ((path_rmse_base['post'] - path_rmse_adapt['post']) / path_rmse_base['post']) * 100
        print(f"\nPost-Event Improvement: {improvement:.1f}%")
    else:
        print(f"\nPost-Event Improvement: N/A (baseline RMSE too small)")
    
    # Also run the original constant speed scenario for comparison
    print("\n" + "="*60)
    print("BASELINE SCENARIO: Constant Speed")
    print("="*60)
    CHANGE_TIME = 2.2
    
    # Run 1: Adaptive
    log_adapt = run_simulation("Adaptive", specialists, friction_scales, adaptive_mode=True, use_slalom=False)
    
    # Run 2: Baseline (Fixed Asphalt)
    log_base = run_simulation("Baseline", specialists, friction_scales, adaptive_mode=False, use_slalom=False)
    
    # Calculate Metrics for constant speed scenario
    metrics_adapt = calculate_metrics(log_adapt, CHANGE_TIME, "ADAPTIVE (Constant Speed)")
    metrics_base = calculate_metrics(log_base, CHANGE_TIME, "BASELINE (Constant Speed)")
    
    # Summary comparison
    print(f"\n=== SUMMARY COMPARISON (Constant Speed) ===")
    print(f"Post-Event RMSE vx:")
    print(f"  Adaptive: {metrics_adapt['rmse_vx_post']:.4f} m/s")
    print(f"  Baseline: {metrics_base['rmse_vx_post']:.4f} m/s")
    print(f"  Improvement: {(1 - metrics_adapt['rmse_vx_post']/metrics_base['rmse_vx_post'])*100:.1f}%")
    
    # --- Plotting: Limit Handling Scenario ---
    t_slalom = np.array(log_adapt_slalom['t'])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Trajectory Comparison (XY)
    ax = axes[0, 0]
    ax.plot(log_base_slalom['ref_x'], log_base_slalom['ref_y'], 'k--', 
            label='Reference (Slalom)', alpha=0.5, linewidth=1.5)
    ax.plot(log_base_slalom['x'], log_base_slalom['y'], 'r', 
            label='Baseline', linewidth=2, alpha=0.8)
    ax.plot(log_adapt_slalom['x'], log_adapt_slalom['y'], 'b', 
            label='Adaptive', linewidth=2, alpha=0.9)
    ax.set_title('Trajectory Comparison (Slalom on Ice)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Panel 2: Tracking Error over Time
    ax = axes[0, 1]
    err_ad = np.sqrt((np.array(log_adapt_slalom['x']) - np.array(log_adapt_slalom['ref_x']))**2 + 
                     (np.array(log_adapt_slalom['y']) - np.array(log_adapt_slalom['ref_y']))**2)
    err_ba = np.sqrt((np.array(log_base_slalom['x']) - np.array(log_base_slalom['ref_x']))**2 + 
                     (np.array(log_base_slalom['y']) - np.array(log_base_slalom['ref_y']))**2)
    
    ax.plot(t_slalom, err_ba, 'r', label='Baseline Error', linewidth=2, alpha=0.8)
    ax.plot(t_slalom, err_ad, 'b', label='Adaptive Error', linewidth=2, alpha=0.9)
    ax.axvline(CHANGE_TIME_SLALOM, color='k', linestyle='--', alpha=0.7, linewidth=1.5)
    if len(t_slalom) > 0:
        ax.text(CHANGE_TIME_SLALOM + 0.05, ax.get_ylim()[1]*0.9, 'Event', color='k', fontweight='bold')
    ax.set_title('Tracking Error (Euclidean Distance)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [m]')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Velocity
    ax = axes[1, 0]
    ax.plot(t_slalom, log_adapt_slalom['vx'], 'b', label='Adaptive vx', linewidth=2, alpha=0.9)
    ax.plot(t_slalom, log_base_slalom['vx'], 'r--', label='Baseline vx', linewidth=2, alpha=0.8)
    ax.axhline(2.1, color='k', linestyle=':', label='Ref (2.1 m/s)', alpha=0.6)
    ax.axvline(CHANGE_TIME_SLALOM, color='k', linestyle='--', alpha=0.7)
    ax.set_title('Speed Maintenance')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'$v_x$ [m/s]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Adaptive Weights
    ax = axes[1, 1]
    w_slalom = np.array(log_adapt_slalom['w'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(specialists)))
    for i in range(len(specialists)):
        label = f'Spec {i} (μ={friction_scales[i]:.2f})' if i in [0, 2, 5, 7] else None
        ax.plot(t_slalom, w_slalom[:, i], color=colors[i], label=label, 
                alpha=0.8, linewidth=1.5)
    ax.set_title('Governor Weight Adaptation')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Weights')
    ax.legend(loc='center right', fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.axvline(CHANGE_TIME_SLALOM, color='k', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vtc_limit_handling_result.png', dpi=300)
    print("\nFigure saved to 'vtc_limit_handling_result.png'")
    
    # --- Plotting: Original Constant Speed Comparison ---
    t = np.array(log_adapt['t'])
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Panel 1: Longitudinal Velocity Comparison
    ax = axes2[0]  # Fixed: use axes2 instead of axes
    ax.plot(t, log_adapt['ref_vx'], 'k:', label='Target (1.5 m/s)', alpha=0.6, linewidth=1.5)
    ax.plot(t, log_base['vx'], 'r--', linewidth=2, label='Baseline (Fixed Asphalt)', alpha=0.8)
    ax.plot(t, log_adapt['vx'], 'b-', linewidth=2, label='Adaptive (HYDRA)', alpha=0.9)
    ax.set_ylabel(r'$v_x$ [m/s]')
    ax.set_title(r'Regime Shift: Asphalt ($\mu \approx 1.0$) $\rightarrow$ Ice ($\mu \approx 0.6$)')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(CHANGE_TIME, color='k', linestyle='-', alpha=0.5, linewidth=1.5)
    ax.text(CHANGE_TIME + 0.05, 1.4, "Event", color='k', fontsize=9, fontweight='bold')
    
    # Panel 2: Lateral Velocity Comparison (Stability)
    ax = axes2[1]  # Fixed: use axes2 instead of axes
    ax.plot(t, [0]*len(t), 'k:', alpha=0.6, linewidth=1)
    ax.plot(t, log_base['vy'], 'r--', linewidth=1.5, label='Baseline', alpha=0.8)
    ax.plot(t, log_adapt['vy'], 'g-', linewidth=2, label='Adaptive', alpha=0.9)
    ax.set_ylabel(r'$v_y$ [m/s]')
    ax.set_title('Lateral Stability')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(CHANGE_TIME, color='k', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Panel 3: Adaptive Weights
    ax = axes2[2]  # Fixed: use axes2 instead of axes
    w = np.array(log_adapt['w'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(specialists)))
    # Plot all specialists, but only label key ones to reduce clutter
    for i in range(len(specialists)):
        label = f'Spec {i} ($\mu={friction_scales[i]:.2f}$)' if i in [0, 2, 5, 7] else None
        ax.plot(t, w[:, i], color=colors[i], label=label, alpha=0.8, linewidth=1.5)
    
    ax.set_ylabel('Weights')
    ax.set_xlabel('Time [s]')
    ax.set_title('Governor Adaptation')
    ax.legend(loc='center right', fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.axvline(CHANGE_TIME, color='k', linestyle='-', alpha=0.5, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig('vtc_comparison_result.png', dpi=300)
    print("\nFigure saved to 'vtc_comparison_result.png'")
    
    # Also save the detailed figure
    fig2 = plt.figure(figsize=(14, 10))
    
    # Panel 1: Longitudinal Velocity
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(t, log_adapt['ref_vx'], 'k:', label='Target', alpha=0.6)
    ax1.plot(t, log_base['vx'], 'r--', linewidth=2, label='Baseline', alpha=0.7)
    ax1.plot(t, log_adapt['vx'], 'b-', linewidth=2, label='Adaptive')
    ax1.set_ylabel(r'$v_x$ [m/s]')
    ax1.set_title('Longitudinal Velocity')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(CHANGE_TIME, color='k', linestyle='-', alpha=0.5)
    
    # Panel 2: Lateral Velocity
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(t, [0]*len(t), 'k:', alpha=0.6)
    ax2.plot(t, log_base['vy'], 'r--', linewidth=1.5, label='Baseline')
    ax2.plot(t, log_adapt['vy'], 'g-', linewidth=2, label='Adaptive')
    ax2.set_ylabel(r'$v_y$ [m/s]')
    ax2.set_title('Lateral Velocity')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(CHANGE_TIME, color='k', linestyle='-', alpha=0.5)
    
    # Panel 3: Yaw Angle
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(t, log_base['psi'], 'r--', linewidth=1.5, label='Baseline', alpha=0.7)
    ax3.plot(t, log_adapt['psi'], 'b-', linewidth=2, label='Adaptive')
    ax3.set_ylabel(r'$\psi$ [rad]')
    ax3.set_title('Yaw Angle')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(CHANGE_TIME, color='k', linestyle='-', alpha=0.5)
    
    # Panel 4: Weights (all specialists)
    ax4 = plt.subplot(3, 2, 4)
    for i in range(len(specialists)):
        ax4.plot(t, w[:, i], label=f'Spec {i} (μ={friction_scales[i]:.2f})',
                 color=colors[i], linewidth=1.5, alpha=0.8)
    ax4.set_ylabel('Ensemble Weights')
    ax4.set_xlabel('Time [s]')
    ax4.set_title('Governor Weight Adaptation')
    ax4.legend(loc='center right', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(CHANGE_TIME, color='k', linestyle='-', alpha=0.5)
    
    # Panel 5: Metrics Comparison (Bar chart)
    ax5 = plt.subplot(3, 2, 5)
    categories = ['vx Pre', 'vx Post', 'vy Pre', 'vy Post']
    adapt_vals = [metrics_adapt['rmse_vx_pre'], metrics_adapt['rmse_vx_post'],
                  metrics_adapt['rmse_vy_pre'], metrics_adapt['rmse_vy_post']]
    base_vals = [metrics_base['rmse_vx_pre'], metrics_base['rmse_vx_post'],
                 metrics_base['rmse_vy_pre'], metrics_base['rmse_vy_post']]
    
    x = np.arange(len(categories))
    width = 0.35
    ax5.bar(x - width/2, adapt_vals, width, label='Adaptive', color='b', alpha=0.7)
    ax5.bar(x + width/2, base_vals, width, label='Baseline', color='r', alpha=0.7)
    ax5.set_ylabel('RMSE')
    ax5.set_title('RMSE Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Weight Entropy
    ax6 = plt.subplot(3, 2, 6)
    entropy = -np.sum(w * np.log(w + 1e-10), axis=1)
    ax6.plot(t, entropy, 'purple', linewidth=2)
    ax6.set_ylabel('Weight Entropy')
    ax6.set_xlabel('Time [s]')
    ax6.set_title('Ensemble Diversity (Lower = More Specialized)')
    ax6.grid(True, alpha=0.3)
    ax6.axvline(CHANGE_TIME, color='k', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('vtc_tracking_figure_best_scenario.png', dpi=300)
    print("Detailed figure saved to 'vtc_tracking_figure_best_scenario.png'")
    
    print(f"\n=== Simulation Complete ===")
    print(f"Number of specialists: {len(specialists)}")
    print(f"Final adaptive weights (slalom): {w_slalom[-1]}")
    print(f"Dominant specialist at end (slalom): {np.argmax(w_slalom[-1])} (μ={friction_scales[np.argmax(w_slalom[-1])]:.2f})")


if __name__ == '__main__':
    main()
