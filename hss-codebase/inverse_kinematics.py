#!/usr/bin/env python3
"""
Inverse Kinematics and Data Generation for H-SS

Implements:
1. Pacejka Inversion: Analytically compute steering angles that hit tire saturation
2. Chirp Excitation: Superimpose frequency-swept signals to excite transient dynamics
3. Fused Trajectory Generation: Combine saturation targeting + chirp for optimal data

Key Functions:
- generate_fused_trajectory(): Main data generator with domain randomization
- sample_candidates_dynamic(): Generate candidate dynamic parameter sets
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Import from hydra_pob for physics consistency
sys.path.insert(0, str(Path(__file__).parent.parent))
from hydra_pob import VehicleParams, DynamicBicycleOracle


@dataclass
class TrajectoryConfig:
    """Configuration for fused trajectory generation."""
    # Velocity ranges
    vx_min: float = 5.0
    vx_max: float = 40.0
    vy_min: float = -3.0
    vy_max: float = 3.0
    omega_min: float = -2.0
    omega_max: float = 2.0
    
    # Saturation targeting (80-95% of peak force)
    saturation_min: float = 0.80
    saturation_max: float = 0.95
    
    # Chirp excitation parameters
    chirp_amplitude: float = 0.08  # rad
    chirp_freq_min: float = 0.5   # Hz
    chirp_freq_max: float = 5.0   # Hz
    
    # Throttle range
    throttle_min: float = -0.5
    throttle_max: float = 0.8
    
    # Domain randomization ranges (mass/inertia)
    mass_min: float = 600.0
    mass_max: float = 1200.0
    inertia_min: float = 800.0
    inertia_max: float = 2000.0


def pacejka_peak_slip(B: float, C: float, E: float) -> float:
    """
    Compute the slip angle at which Pacejka force is maximized.
    
    For the Magic Formula: F = D * sin(C * atan(B*α - E*(B*α - atan(B*α))))
    The peak occurs approximately at α_peak ≈ atan(π/(2*B*C))
    
    This is a simplified approximation; exact solution requires numerical methods.
    """
    # Approximate peak slip angle
    # For typical B=10-14, C=1.3-1.7, this gives α_peak ≈ 0.1-0.2 rad
    if B * C > 0.1:
        return np.arctan(np.pi / (2 * B * C))
    return 0.15  # Fallback


def pacejka_force(alpha: np.ndarray, B: float, C: float, D: float, E: float,
                  normal_load: float) -> np.ndarray:
    """
    Evaluate Pacejka Magic Formula.
    
    F = D * Fz * sin(C * atan(B*α - E*(B*α - atan(B*α))))
    """
    inner = B * alpha - E * (B * alpha - np.arctan(B * alpha))
    return D * normal_load * np.sin(C * np.arctan(inner))


def inverse_pacejka_slip(
    target_force_fraction: float,
    B: float, C: float, D: float, E: float,
    normal_load: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """
    Invert Pacejka to find slip angle for target force (as fraction of peak).
    
    Uses Newton-Raphson iteration to solve:
        F(α) = target_fraction * F_peak
    
    Args:
        target_force_fraction: Desired force as fraction of peak (0.8-0.95)
        B, C, D, E: Pacejka coefficients
        normal_load: Tire normal force (N)
        
    Returns:
        Slip angle (rad) that produces the target force
    """
    # Find peak force and slip
    alpha_peak = pacejka_peak_slip(B, C, E)
    F_peak = np.abs(pacejka_force(np.array([alpha_peak]), B, C, D, E, normal_load)[0])
    
    F_target = target_force_fraction * F_peak
    
    # Newton-Raphson on monotonic region (0, alpha_peak)
    alpha = alpha_peak * 0.5  # Initial guess in middle of monotonic region
    
    for _ in range(max_iter):
        F = pacejka_force(np.array([alpha]), B, C, D, E, normal_load)[0]
        err = F - F_target
        
        if np.abs(err) < tol:
            break
        
        # Numerical derivative
        dalpha = 1e-6
        F_plus = pacejka_force(np.array([alpha + dalpha]), B, C, D, E, normal_load)[0]
        dF_dalpha = (F_plus - F) / dalpha
        
        if np.abs(dF_dalpha) > 1e-10:
            alpha = alpha - err / dF_dalpha
            alpha = np.clip(alpha, 0.01, alpha_peak * 0.99)
    
    return float(alpha)


def compute_saturation_steering(
    vx: np.ndarray, vy: np.ndarray, omega: np.ndarray,
    params: VehicleParams,
    target_saturation: np.ndarray,
) -> np.ndarray:
    """
    Compute steering angles that target specific tire saturation levels.
    
    Uses inverse kinematics to find δ such that front slip angle α_f
    corresponds to target_saturation * α_peak.
    
    δ = α_f_target + atan2(vy + lf*ω, vx)
    
    Args:
        vx, vy, omega: Vehicle state arrays [n_samples]
        params: Vehicle parameters (for tire coefficients)
        target_saturation: Target saturation fraction [n_samples] in (0.8, 0.95)
        
    Returns:
        Steering angles δ [n_samples]
    """
    n = len(vx)
    delta = np.zeros(n)
    
    # Compute kinematic term
    kinematic_term = np.arctan2(vy + params.lf * omega, np.maximum(vx, 0.1))
    
    # For each sample, compute target slip angle via inverse Pacejka
    for i in range(n):
        # Front tire normal load
        normal_load_f = params.m * 9.81 * params.lr / (params.lf + params.lr)
        
        # Target slip angle for this saturation level
        alpha_f_target = inverse_pacejka_slip(
            target_saturation[i],
            params.pacejka_B_f, params.pacejka_C_f,
            params.pacejka_D_f, params.pacejka_E_f,
            normal_load_f,
        )
        
        # Randomly choose positive or negative slip
        sign = np.random.choice([-1.0, 1.0])
        alpha_f_target *= sign
        
        # Inverse kinematics: δ = α_f + kinematic_term
        delta[i] = alpha_f_target + kinematic_term[i]
    
    return delta


def generate_chirp_excitation(
    n_samples: int,
    config: TrajectoryConfig,
    pseudo_time: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate chirp excitation signal for I_z dynamics excitation.
    
    chirp(t) = A * sin(2π * f(t) * t)
    
    where f(t) sweeps from freq_min to freq_max.
    
    Args:
        n_samples: Number of samples
        config: Configuration with chirp parameters
        pseudo_time: Optional pseudo-time array (for trajectory coherence)
        
    Returns:
        Chirp signal [n_samples]
    """
    if pseudo_time is None:
        pseudo_time = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # Sample frequencies for each point
    freq = np.random.uniform(config.chirp_freq_min, config.chirp_freq_max, n_samples)
    
    # Generate chirp signal
    chirp = config.chirp_amplitude * np.sin(2 * np.pi * freq * pseudo_time)
    
    return chirp


def generate_fused_trajectory(
    dynamic_params: VehicleParams,
    n_samples: int = 2048,
    config: Optional[TrajectoryConfig] = None,
    device: torch.device = None,
    return_numpy: bool = False,
) -> Tuple:
    """
    Generate fused trajectory with domain-randomized mass/inertia.
    
    FUSION STRATEGY:
    1. Base steering from Pacejka inversion (80-95% saturation)
    2. Chirp excitation superimposed (A * sin(2πft))
    3. Mass/Inertia domain randomized per sample
    
    This produces training data that:
    - Covers tire saturation regions (informative for μ estimation)
    - Excites transient dynamics (informative for I_z estimation)
    - Teaches physics relationship a ∝ F/m via mass randomization
    
    Args:
        dynamic_params: VehicleParams with fixed DYNAMIC params (μ, Cd, B, C, E)
        n_samples: Number of samples to generate
        config: TrajectoryConfig (uses defaults if None)
        device: Torch device (uses CUDA if available)
        return_numpy: If True, return numpy arrays instead of tensors
        
    Returns:
        states: [n_samples, 3] (vx, vy, omega)
        controls: [n_samples, 2] (delta, throttle)
        static_params: [n_samples, 2] (mass, Iz) - DOMAIN RANDOMIZED
        accelerations: [n_samples, 3] (dvx, dvy, domega)
    """
    if config is None:
        config = TrajectoryConfig()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sample state space
    vx = np.random.uniform(config.vx_min, config.vx_max, n_samples)
    vy = np.random.uniform(config.vy_min, config.vy_max, n_samples)
    omega = np.random.uniform(config.omega_min, config.omega_max, n_samples)
    
    # Sample saturation targets
    saturation = np.random.uniform(config.saturation_min, config.saturation_max, n_samples)
    
    # DOMAIN RANDOMIZATION: Sample mass and inertia for EACH sample
    mass = np.random.uniform(config.mass_min, config.mass_max, n_samples)
    inertia = np.random.uniform(config.inertia_min, config.inertia_max, n_samples)
    
    # Generate steering with saturation targeting
    delta_saturation = compute_saturation_steering(vx, vy, omega, dynamic_params, saturation)
    
    # Add chirp excitation
    chirp = generate_chirp_excitation(n_samples, config)
    delta = delta_saturation + chirp
    
    # Sample throttle
    throttle = np.random.uniform(config.throttle_min, config.throttle_max, n_samples)
    
    # Compute accelerations with per-sample mass/inertia
    # This is the KEY to domain randomization
    accelerations = compute_accelerations_vectorized(
        vx, vy, omega, delta, throttle,
        mass, inertia, dynamic_params
    )
    
    # Stack arrays
    states = np.stack([vx, vy, omega], axis=1)
    controls = np.stack([delta, throttle], axis=1)
    static_params = np.stack([mass, inertia], axis=1)
    
    if return_numpy:
        return states, controls, static_params, accelerations
    
    # Convert to torch
    states_t = torch.from_numpy(states).float().to(device)
    controls_t = torch.from_numpy(controls).float().to(device)
    static_t = torch.from_numpy(static_params).float().to(device)
    accels_t = torch.from_numpy(accelerations).float().to(device)
    
    return states_t, controls_t, static_t, accels_t


def compute_accelerations_vectorized(
    vx: np.ndarray, vy: np.ndarray, omega: np.ndarray,
    delta: np.ndarray, throttle: np.ndarray,
    mass: np.ndarray, inertia: np.ndarray,
    params: VehicleParams,
) -> np.ndarray:
    """
    Compute accelerations with per-sample mass and inertia.
    
    This is the vectorized physics engine that enables domain randomization.
    Uses the same equations as DynamicBicycleOracle but with per-sample m, Iz.
    
    Args:
        vx, vy, omega: Vehicle state [n_samples]
        delta, throttle: Control inputs [n_samples]
        mass, inertia: Per-sample static params [n_samples]
        params: VehicleParams with dynamic coefficients (Pacejka, aero)
        
    Returns:
        accelerations: [n_samples, 3] (dvx, dvy, domega)
    """
    p = params
    vx = np.maximum(vx, 0.1)  # Avoid division by zero
    
    # Slip angles
    alpha_f = delta - np.arctan2(vy + p.lf * omega, vx)
    alpha_r = np.arctan2(p.lr * omega - vy, vx)
    
    # Normal loads with per-sample mass
    normal_load_f = mass * 9.81 * p.lr / (p.lf + p.lr)
    normal_load_r = mass * 9.81 * p.lf / (p.lf + p.lr)
    
    # Pacejka lateral forces
    Ffy = pacejka_force(alpha_f, p.pacejka_B_f, p.pacejka_C_f, p.pacejka_D_f, 
                        p.pacejka_E_f, normal_load_f)
    Fry = pacejka_force(alpha_r, p.pacejka_B_r, p.pacejka_C_r, p.pacejka_D_r,
                        p.pacejka_E_r, normal_load_r)
    
    # Drivetrain force
    Frx = (p.cm1 * throttle) - (p.cm2 * vx) - p.cr0 - (p.cd * vx**2)
    
    # Accelerations with per-sample mass and inertia
    dvx = (Frx - Ffy * np.sin(delta) + mass * vy * omega) / mass
    dvy = (Fry + Ffy * np.cos(delta) - mass * vx * omega) / mass
    domega = (Ffy * p.lf * np.cos(delta) - Fry * p.lr) / inertia
    
    return np.stack([dvx, dvy, domega], axis=1)


def sample_candidates_dynamic(
    n_candidates: int = 50,
    seed: Optional[int] = None,
) -> List[VehicleParams]:
    """
    Generate candidate dynamic parameter sets for S²GPT library building.
    
    Samples the DYNAMIC parameter space:
    - Friction coefficients (μ / D)
    - Pacejka shape coefficients (B, C, E)
    - Aero drag (Cd)
    - Drivetrain characteristics
    
    Static parameters (m, Iz) are kept at nominal values since they are
    handled via input conditioning rather than library diversity.
    
    Args:
        n_candidates: Number of candidate parameter sets
        seed: Random seed for reproducibility
        
    Returns:
        List of VehicleParams with varied dynamic parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    candidates = []
    
    # Parameter ranges (physically plausible for road vehicles)
    ranges = {
        # Friction (D) - most important for dynamics
        "pacejka_D_f": (0.3, 1.2),  # Ice (0.3) to Dry (1.2)
        "pacejka_D_r": (0.3, 1.2),
        # Pacejka shape
        "pacejka_B_f": (8.0, 16.0),
        "pacejka_B_r": (8.0, 16.0),
        "pacejka_C_f": (1.0, 2.0),
        "pacejka_C_r": (1.0, 2.0),
        "pacejka_E_f": (0.0, 1.0),
        "pacejka_E_r": (0.0, 1.0),
        # Aero and drivetrain
        "cd": (0.2, 0.8),
        "cm1": (1500.0, 2500.0),
    }
    
    for _ in range(n_candidates):
        kwargs = {}
        for param, (lo, hi) in ranges.items():
            kwargs[param] = float(np.random.uniform(lo, hi))
        
        # Use nominal static params (these are handled via conditioning)
        base = VehicleParams(m=800.0, Iz=1200.0)  # Nominal values
        candidates.append(base.clone_with(**kwargs))
    
    return candidates


def generate_sparse_grid(
    candidates: List[VehicleParams],
    n_samples_per_candidate: int = 128,
    config: Optional[TrajectoryConfig] = None,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate dense evaluation grid for S²GPT error computation.
    
    Creates a fixed grid of (state, control) pairs evaluated across
    multiple candidate parameter configurations.
    
    Args:
        candidates: List of VehicleParams to evaluate
        n_samples_per_candidate: Samples per candidate
        config: Trajectory configuration
        device: Torch device
        
    Returns:
        states: [total, 3]
        controls: [total, 2]  
        static_params: [total, 2] (fixed nominal for grid)
        candidate_indices: [total] indicating which candidate each sample belongs to
    """
    if config is None:
        config = TrajectoryConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_states = []
    all_controls = []
    all_static = []
    all_indices = []
    
    # Use fixed nominal static params for grid (since we're evaluating candidates)
    nominal_m = (config.mass_min + config.mass_max) / 2
    nominal_Iz = (config.inertia_min + config.inertia_max) / 2
    
    for idx, params in enumerate(candidates):
        # Generate samples for this candidate
        vx = np.random.uniform(config.vx_min, config.vx_max, n_samples_per_candidate)
        vy = np.random.uniform(config.vy_min, config.vy_max, n_samples_per_candidate)
        omega = np.random.uniform(config.omega_min, config.omega_max, n_samples_per_candidate)
        
        # Saturation-targeted steering
        saturation = np.random.uniform(config.saturation_min, config.saturation_max, 
                                       n_samples_per_candidate)
        delta = compute_saturation_steering(vx, vy, omega, params, saturation)
        chirp = generate_chirp_excitation(n_samples_per_candidate, config)
        delta = delta + chirp
        
        throttle = np.random.uniform(config.throttle_min, config.throttle_max, 
                                     n_samples_per_candidate)
        
        states = np.stack([vx, vy, omega], axis=1)
        controls = np.stack([delta, throttle], axis=1)
        static = np.full((n_samples_per_candidate, 2), [nominal_m, nominal_Iz])
        
        all_states.append(states)
        all_controls.append(controls)
        all_static.append(static)
        all_indices.append(np.full(n_samples_per_candidate, idx))
    
    states_t = torch.from_numpy(np.vstack(all_states)).float().to(device)
    controls_t = torch.from_numpy(np.vstack(all_controls)).float().to(device)
    static_t = torch.from_numpy(np.vstack(all_static)).float().to(device)
    indices_t = torch.from_numpy(np.concatenate(all_indices)).long().to(device)
    
    return states_t, controls_t, static_t, indices_t


def generate_test_trajectory(
    params: VehicleParams,
    n_steps: int = 500,
    dt: float = 0.02,
    locked_mass: float = 1000.0,
    locked_inertia: float = 1500.0,
    scenario: str = "steady_state_cornering",
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a test trajectory for validation with locked static params.
    
    Scenarios:
    - "steady_state_cornering": Constant speed turning
    - "slalom": S-curve maneuver
    - "dry_to_wet": Friction transition (for online adaptation testing)
    
    Args:
        params: VehicleParams for dynamics (will be modified for dry_to_wet)
        n_steps: Number of timesteps
        dt: Time step (s)
        locked_mass: Fixed mass (from Phase 1)
        locked_inertia: Fixed inertia (from Phase 1)
        scenario: Test scenario type
        device: Torch device
        
    Returns:
        states: [n_steps, 3]
        controls: [n_steps, 2]
        static_params: [n_steps, 2] (locked values)
        accelerations: [n_steps, 3]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create oracle with specified mass/inertia
    test_params = params.clone_with(m=locked_mass, Iz=locked_inertia)
    oracle = DynamicBicycleOracle(test_params, device=device)
    
    # Initialize state
    vx0 = 20.0  # m/s
    state = torch.tensor([[vx0, 0.0, 0.0]], device=device)
    
    all_states = []
    all_controls = []
    all_accels = []
    
    for step in range(n_steps):
        t = step * dt
        
        if scenario == "steady_state_cornering":
            # Constant steering, mild throttle
            delta = 0.1 * np.sin(0.5 * t)  # Slow oscillation
            throttle = 0.3
            
        elif scenario == "slalom":
            # S-curve with varying frequency
            freq = 0.5 + 0.5 * t / (n_steps * dt)  # Increasing frequency
            delta = 0.2 * np.sin(2 * np.pi * freq * t)
            throttle = 0.4 - 0.1 * np.abs(np.sin(2 * np.pi * freq * t))
            
        elif scenario == "dry_to_wet":
            # Transition scenario - simulate friction change at t=halfway
            if t < (n_steps * dt / 2):
                # Dry road
                oracle.params = params.clone_with(m=locked_mass, Iz=locked_inertia,
                                                   pacejka_D_f=1.0, pacejka_D_r=1.0)
            else:
                # Wet road
                oracle.params = params.clone_with(m=locked_mass, Iz=locked_inertia,
                                                   pacejka_D_f=0.5, pacejka_D_r=0.5)
            delta = 0.15 * np.sin(1.0 * t)
            throttle = 0.35
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        control = torch.tensor([[delta, throttle]], device=device)
        accel = oracle.accelerations(state, control)
        
        all_states.append(state.clone())
        all_controls.append(control.clone())
        all_accels.append(accel.clone())
        
        # RK4 integration
        state = oracle.rk4_step(state, control, dt=dt)
        
        # Clamp vx to prevent instability
        state[0, 0] = torch.clamp(state[0, 0], min=5.0, max=50.0)
    
    states_t = torch.cat(all_states, dim=0)
    controls_t = torch.cat(all_controls, dim=0)
    accels_t = torch.cat(all_accels, dim=0)
    
    # Static params are locked (construct properly)
    static_t = torch.zeros((n_steps, 2), device=device, dtype=torch.float32)
    static_t[:, 0] = locked_mass
    static_t[:, 1] = locked_inertia
    
    return states_t, controls_t, static_t, accels_t


# Convenience function for quick trajectory generation
def generate_training_batch(
    dynamic_params: VehicleParams,
    n_samples: int = 2048,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper for generate_fused_trajectory.
    
    Returns:
        states: [n_samples, 3]
        controls: [n_samples, 2]
        static_params: [n_samples, 2] (domain randomized)
        accelerations: [n_samples, 3]
    """
    return generate_fused_trajectory(
        dynamic_params=dynamic_params,
        n_samples=n_samples,
        device=device,
        return_numpy=False,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing inverse_kinematics.py...")
    
    # Test fused trajectory generation
    params = VehicleParams(m=800.0, Iz=1200.0, pacejka_D_f=0.9, pacejka_D_r=0.9)
    states, controls, static_params, accels = generate_fused_trajectory(
        params, n_samples=512, return_numpy=True
    )
    
    print(f"States shape: {states.shape}")
    print(f"Controls shape: {controls.shape}")
    print(f"Static params shape: {static_params.shape}")
    print(f"Accelerations shape: {accels.shape}")
    
    print(f"\nMass range: [{static_params[:, 0].min():.1f}, {static_params[:, 0].max():.1f}]")
    print(f"Inertia range: [{static_params[:, 1].min():.1f}, {static_params[:, 1].max():.1f}]")
    print(f"Steering range: [{controls[:, 0].min():.3f}, {controls[:, 0].max():.3f}]")
    
    # Test candidate generation
    candidates = sample_candidates_dynamic(n_candidates=10)
    print(f"\nGenerated {len(candidates)} dynamic candidates")
    print(f"Friction range: [{min(c.pacejka_D_f for c in candidates):.2f}, "
          f"{max(c.pacejka_D_f for c in candidates):.2f}]")
    
    print("\n✓ inverse_kinematics.py tests passed!")
