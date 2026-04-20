#!/usr/bin/env python3
"""
H-SS Training Pipeline

Implements the complete training workflow for H-SS (HYDRA Self-Supervised) specialists:
1. Greedy specialist selection (without GEIM/EIM sparsification)
2. Domain randomization for mass/inertia during training
3. Physics-informed loss functions with early stopping
4. H-SS compliant data generation (saturation targeting + chirp excitation)

This module implements the H-SS algorithm for building specialist libraries
that can adapt online through linear regression weight updates.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Try relative import first (when imported as module)
    from .specialist import HSSConfig, HSSSpecialist, HSSEnsemble, verify_jacobian
except ImportError:
    # Fall back to absolute import (when run as script)
    from specialist import HSSConfig, HSSSpecialist, HSSEnsemble, verify_jacobian


@dataclass
class VehicleParams:
    """Vehicle parameters for dynamics computation."""
    # Static parameters
    m: float = 800.0           # mass (kg)
    Iz: float = 1200.0         # yaw inertia (kg·m²)
    lf: float = 1.5            # CG to front axle (m)
    lr: float = 1.0            # CG to rear axle (m)
    
    # Pacejka tire parameters
    pacejka_B_f: float = 12.0
    pacejka_C_f: float = 1.5
    pacejka_D_f: float = 1.0   # Front friction coefficient
    pacejka_E_f: float = 0.5
    pacejka_B_r: float = 12.0
    pacejka_C_r: float = 1.5
    pacejka_D_r: float = 1.0   # Rear friction coefficient
    pacejka_E_r: float = 0.5
    
    # Drivetrain
    cm1: float = 2000.0
    cm2: float = 10.0
    cr0: float = 200.0
    cd: float = 0.5
    
    def clone_with(self, **kwargs) -> "VehicleParams":
        """Create copy with modified parameters."""
        new_params = VehicleParams(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        for k, v in kwargs.items():
            if hasattr(new_params, k):
                setattr(new_params, k, v)
        return new_params


@dataclass
class TrainingConfig:
    """Configuration for S²GPT-PINN training."""
    # Specialist architecture
    hidden_dim: int = 64
    n_layers: int = 3

    # Training
    n_samples: int = 4096
    epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3

    # Early stopping
    early_stopping_patience: int = 50  # Stop if no improvement for this many epochs
    early_stopping_min_delta: float = 1e-6  # Minimum improvement to count as progress
    validation_split: float = 0.1  # Fraction of data for validation

    # Data generation
    vx_range: Tuple[float, float] = (5.0, 40.0)
    vy_range: Tuple[float, float] = (-3.0, 3.0)
    omega_range: Tuple[float, float] = (-2.0, 2.0)
    throttle_range: Tuple[float, float] = (-0.5, 0.8)

    # Saturation targeting (H-SS style data generation)
    saturation_min: float = 0.80  # Target 80-95% of peak tire force
    saturation_max: float = 0.95

    # Chirp excitation parameters
    chirp_amplitude: float = 0.08  # rad
    chirp_freq_min: float = 0.5    # Hz
    chirp_freq_max: float = 5.0    # Hz

    # Domain randomization
    mass_range: Tuple[float, float] = (600.0, 1200.0)
    inertia_range: Tuple[float, float] = (800.0, 2000.0)

    # Library building
    n_candidates: int = 50
    max_basis: int = 8
    tol: float = 1e-4

    # Seeds
    seed: int = 42


def compute_pacejka_force(
    alpha: torch.Tensor,
    B: float, C: float, D: float, E: float,
    normal_load: torch.Tensor
) -> torch.Tensor:
    """Compute Pacejka Magic Formula lateral force."""
    inner = B * alpha - E * (B * alpha - torch.atan(B * alpha))
    return D * normal_load * torch.sin(C * torch.atan(inner))


def pacejka_peak_slip(B: float, C: float, E: float) -> float:
    """
    Compute the slip angle at which Pacejka force is maximized.

    For the Magic Formula: F = D * Fz * sin(C * atan(B*α - E*(B*α - atan(B*α))))
    The peak occurs approximately at α_peak ≈ atan(π/(2*B*C))

    This is a simplified approximation; exact solution requires numerical methods.
    """
    # Approximate peak slip angle
    # For typical B=10-14, C=1.3-1.7, this gives α_peak ≈ 0.1-0.2 rad
    if B * C > 0.1:
        return np.arctan(np.pi / (2 * B * C))
    return 0.15  # Fallback


def pacejka_force_numpy(
    alpha: np.ndarray,
    B: float, C: float, D: float, E: float,
    normal_load: float
) -> np.ndarray:
    """Compute Pacejka Magic Formula (numpy version for inversion)."""
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
    F_peak = np.abs(pacejka_force_numpy(np.array([alpha_peak]), B, C, D, E, normal_load)[0])

    F_target = target_force_fraction * F_peak

    # Newton-Raphson on monotonic region (0, alpha_peak)
    alpha = alpha_peak * 0.5  # Initial guess in middle of monotonic region

    for _ in range(max_iter):
        F = pacejka_force_numpy(np.array([alpha]), B, C, D, E, normal_load)[0]
        err = F - F_target

        if np.abs(err) < tol:
            break

        # Numerical derivative
        dalpha = 1e-6
        F_plus = pacejka_force_numpy(np.array([alpha + dalpha]), B, C, D, E, normal_load)[0]
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
        # Front tire normal load (use nominal mass for steering computation)
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
    config: TrainingConfig,
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


def compute_physics_accelerations(
    state: torch.Tensor,
    control: torch.Tensor,
    static_params: torch.Tensor,
    dynamic_params: VehicleParams
) -> torch.Tensor:
    """
    Compute physics-based accelerations with per-sample static parameters.
    
    This enables domain randomization: each sample can have different m, Iz
    while sharing the same dynamic parameters (friction, tire coefficients).
    
    Args:
        state: [batch, 3] (vx, vy, omega)
        control: [batch, 2] (delta, throttle)
        static_params: [batch, 2] (mass, Iz)
        dynamic_params: VehicleParams with tire/aero coefficients
        
    Returns:
        accelerations: [batch, 3] (dvx, dvy, domega)
    """
    p = dynamic_params
    
    vx = torch.clamp(state[:, 0], min=0.1)
    vy = state[:, 1]
    omega = state[:, 2]
    delta = control[:, 0]
    throttle = control[:, 1]
    m = static_params[:, 0]
    iz = static_params[:, 1]
    
    # Slip angles
    alpha_f = delta - torch.atan2(vy + p.lf * omega, vx)
    alpha_r = torch.atan2(p.lr * omega - vy, vx)
    
    # Normal loads (per-sample mass)
    normal_load_f = m * 9.81 * p.lr / (p.lf + p.lr)
    normal_load_r = m * 9.81 * p.lf / (p.lf + p.lr)
    
    # Pacejka lateral forces
    Ffy = compute_pacejka_force(alpha_f, p.pacejka_B_f, p.pacejka_C_f, 
                                 p.pacejka_D_f, p.pacejka_E_f, normal_load_f)
    Fry = compute_pacejka_force(alpha_r, p.pacejka_B_r, p.pacejka_C_r,
                                 p.pacejka_D_r, p.pacejka_E_r, normal_load_r)
    
    # Drivetrain force
    Frx = (p.cm1 * throttle) - (p.cm2 * vx) - p.cr0 - (p.cd * vx**2)
    
    # Accelerations with per-sample mass/inertia
    dvx = (Frx - Ffy * torch.sin(delta) + m * vy * omega) / m
    dvy = (Fry + Ffy * torch.cos(delta) - m * vx * omega) / m
    domega = (Ffy * p.lf * torch.cos(delta) - Fry * p.lr) / iz
    
    return torch.stack([dvx, dvy, domega], dim=1)


def generate_training_data(
    dynamic_params: VehicleParams,
    config: TrainingConfig,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training data with H-SS style sophisticated sampling.

    FUSION STRATEGY (from hss-codebase):
    1. Base steering from Pacejka inversion (80-95% saturation)
    2. Chirp excitation superimposed (A * sin(2πft))
    3. Mass/Inertia domain randomized per sample

    This produces training data that:
    - Covers tire saturation regions (informative for μ estimation)
    - Excites transient dynamics (informative for I_z estimation)
    - Teaches physics relationship a ∝ F/m via mass randomization

    Returns:
        state: [n_samples, 3]
        control: [n_samples, 2]
        static_params: [n_samples, 2] (domain randomized)
        accelerations: [n_samples, 3]
    """
    n = config.n_samples

    # Sample state space
    vx = np.random.uniform(*config.vx_range, n)
    vy = np.random.uniform(*config.vy_range, n)
    omega = np.random.uniform(*config.omega_range, n)

    # Sample saturation targets (80-95% of peak tire force)
    saturation = np.random.uniform(config.saturation_min, config.saturation_max, n)

    # DOMAIN RANDOMIZATION: Sample mass and inertia for EACH sample
    mass = np.random.uniform(*config.mass_range, n)
    inertia = np.random.uniform(*config.inertia_range, n)

    # Generate steering with SATURATION TARGETING
    delta_saturation = compute_saturation_steering(vx, vy, omega, dynamic_params, saturation)

    # Add CHIRP EXCITATION for transient dynamics
    chirp = generate_chirp_excitation(n, config)
    delta = delta_saturation + chirp

    # Sample throttle randomly
    throttle = np.random.uniform(*config.throttle_range, n)

    # Stack arrays
    states = np.stack([vx, vy, omega], axis=1)
    controls = np.stack([delta, throttle], axis=1)
    static_params_np = np.stack([mass, inertia], axis=1)

    # Convert to tensors
    state = torch.from_numpy(states).float().to(device)
    control = torch.from_numpy(controls).float().to(device)
    static_params = torch.from_numpy(static_params_np).float().to(device)

    # Compute physics accelerations with per-sample mass/inertia
    accelerations = compute_physics_accelerations(
        state, control, static_params, dynamic_params
    )

    return state, control, static_params, accelerations


def train_specialist(
    specialist: HSSSpecialist,
    dynamic_params: VehicleParams,
    config: TrainingConfig,
    device: torch.device,
    progress: bool = False
) -> Tuple[float, int]:
    """
    Train a single specialist with domain-randomized static parameters and early stopping.

    The specialist learns to predict accelerations given (state, control, static_params)
    where static_params vary per sample to teach the network a ∝ F/m.

    Uses early stopping to train until the model literally cannot improve anymore.

    Returns:
        Tuple of (final validation loss, epochs trained)
    """
    specialist.train()
    specialist.to(device)

    optimizer = optim.Adam(specialist.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_weights = specialist.state_dict().copy()
    patience_counter = 0
    epochs_without_improvement = 0

    # Generate fixed validation data (once, not regenerated each epoch)
    val_config = TrainingConfig(**{k: getattr(config, k) for k in TrainingConfig.__dataclass_fields__})
    val_config.n_samples = max(256, int(config.n_samples * config.validation_split))

    val_state, val_control, val_static, val_target = generate_training_data(
        dynamic_params, val_config, device
    )

    final_loss = float('inf')
    actual_epochs_trained = 0

    epoch_iter = tqdm(range(config.epochs), desc="Training", disable=not progress, leave=False)
    for epoch in epoch_iter:
        actual_epochs_trained = epoch + 1

        # Regenerate training data each epoch for domain randomization
        train_state, train_control, train_static, train_target = generate_training_data(
            dynamic_params, config, device
        )

        # Training phase
        specialist.train()
        n_train_samples = train_state.shape[0]
        train_permutation = torch.randperm(n_train_samples, device=device)
        train_loss = 0.0
        n_train_batches = 0

        for idx in train_permutation.split(config.batch_size):
            optimizer.zero_grad()

            pred = specialist(train_state[idx], train_control[idx], train_static[idx])
            loss = criterion(pred, train_target[idx])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / n_train_batches

        # Validation phase
        specialist.eval()
        with torch.no_grad():
            n_val_samples = val_state.shape[0]
            val_permutation = torch.randperm(n_val_samples, device=device)
            val_loss = 0.0
            n_val_batches = 0

            for idx in val_permutation.split(config.batch_size):
                pred = specialist(val_state[idx], val_control[idx], val_static[idx])
                loss = criterion(pred, val_target[idx])
                val_loss += loss.item()
                n_val_batches += 1

            avg_val_loss = val_loss / n_val_batches

        # Early stopping check
        if avg_val_loss < best_val_loss - config.early_stopping_min_delta:
            # Significant improvement
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_weights = specialist.state_dict().copy()
            patience_counter = 0
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            patience_counter += 1

        final_loss = avg_val_loss

        epoch_iter.set_postfix({
            'train': f"{avg_train_loss:.6f}",
            'val': f"{avg_val_loss:.6f}",
            'best': f"{best_val_loss:.6f}",
            'patience': patience_counter
        })

        # Check early stopping condition
        if patience_counter >= config.early_stopping_patience:
            if progress:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
                print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
            break

    # Restore best weights
    specialist.load_state_dict(best_weights)
    specialist.eval()

    return final_loss, actual_epochs_trained


def sample_candidate_params(
    n_candidates: int,
    seed: int = 42
) -> List[VehicleParams]:
    """
    Generate candidate dynamic parameter sets for library building.
    
    Samples the DYNAMIC parameter space (friction, tire stiffness, aero).
    Static parameters (m, Iz) are handled via domain randomization during training.
    """
    np.random.seed(seed)
    
    candidates = []
    base = VehicleParams()
    
    # Parameter ranges
    ranges = {
        "pacejka_D_f": (0.3, 1.2),  # Friction coefficient
        "pacejka_D_r": (0.3, 1.2),
        "pacejka_B_f": (8.0, 16.0),
        "pacejka_B_r": (8.0, 16.0),
        "pacejka_C_f": (1.0, 2.0),
        "pacejka_C_r": (1.0, 2.0),
        "cd": (0.2, 0.8),
        "cm1": (1500.0, 2500.0),
    }
    
    for _ in range(n_candidates):
        kwargs = {}
        for param, (lo, hi) in ranges.items():
            kwargs[param] = float(np.random.uniform(lo, hi))
        candidates.append(base.clone_with(**kwargs))
    
    return candidates


class HSSLibraryBuilder:
    """
    Greedy library builder for S²GPT-PINN specialists.
    
    Implements the S²GPT algorithm:
    1. Initialize with first candidate
    2. Greedy loop:
       - Compute reconstruction error for all candidates
       - Select worst-represented candidate
       - Train new specialist with domain randomization
       - Add to library
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device
    ):
        self.config = config
        self.device = device
        
        self.specialists: List[HSSSpecialist] = []
        self.trained_params: List[VehicleParams] = []
        self.mu_centers: List[float] = []
        
        # Error tracking
        self.max_errors: List[float] = []
        
        # Dense grid for error computation
        self._dense_grid = None
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    
    def _get_dense_grid(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate fixed dense grid for error evaluation."""
        if self._dense_grid is None:
            n = 1024
            cfg = self.config
            
            vx = torch.empty(n, device=self.device).uniform_(*cfg.vx_range)
            vy = torch.empty(n, device=self.device).uniform_(*cfg.vy_range)
            omega = torch.empty(n, device=self.device).uniform_(*cfg.omega_range)
            delta = torch.empty(n, device=self.device).uniform_(*cfg.delta_range)
            throttle = torch.empty(n, device=self.device).uniform_(*cfg.throttle_range)
            
            # Use nominal static params for grid
            mass = torch.full((n,), 800.0, device=self.device)
            inertia = torch.full((n,), 1200.0, device=self.device)
            
            self._dense_grid = (
                torch.stack([vx, vy, omega], dim=1),
                torch.stack([delta, throttle], dim=1),
                torch.stack([mass, inertia], dim=1)
            )
        
        return self._dense_grid
    
    def _compute_error(self, dynamic_params: VehicleParams) -> float:
        """Compute reconstruction error for a candidate parameter set."""
        if not self.specialists:
            return float("inf")
        
        state, control, static = self._get_dense_grid()
        
        # Compute target accelerations
        target = compute_physics_accelerations(
            state, control, static, dynamic_params
        )
        
        # Evaluate all specialists
        with torch.no_grad():
            preds = []
            for specialist in self.specialists:
                pred = specialist(state, control, static)
                preds.append(pred)
            
            # Stack: [N, 3, K]
            H = torch.stack(preds, dim=2)
            N = state.shape[0]
            H_flat = H.view(N * 3, len(self.specialists))
            target_flat = target.view(N * 3)
            
            try:
                sol = torch.linalg.lstsq(H_flat, target_flat).solution
                recon = H_flat @ sol
                err = torch.mean((recon - target_flat) ** 2).item()
            except RuntimeError:
                err = float("inf")
        
        return err
    
    def build_library(
        self,
        candidates: Optional[List[VehicleParams]] = None,
        progress: bool = True
    ) -> HSSEnsemble:
        """
        Build specialist library using greedy selection.
        
        Returns:
            HSSEnsemble with trained specialists
        """
        cfg = self.config
        
        if candidates is None:
            print(f"Generating {cfg.n_candidates} dynamic candidates...")
            candidates = sample_candidate_params(cfg.n_candidates, cfg.seed)
        
        print(f"\nStarting S²GPT-PINN library building...")
        print(f"  Candidates: {len(candidates)}")
        print(f"  Max specialists: {cfg.max_basis}")
        
        # Initialize with first candidate
        first_params = candidates[0]
        print(f"\nStep 1: Initialize with μ={first_params.pacejka_D_f:.2f}")
        
        spec_config = HSSConfig(
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers
        )
        first_specialist = HSSSpecialist(spec_config)
        
        loss, epochs_trained = train_specialist(
            first_specialist, first_params, cfg, self.device, progress=progress
        )
        print(f"  Training loss: {loss:.6f} (trained for {epochs_trained} epochs)")
        
        # Verify Jacobians
        errors = verify_jacobian(first_specialist)
        print(f"  Jacobian error: {errors['jacobian_x_error']:.2e}")
        
        self.specialists.append(first_specialist)
        self.trained_params.append(first_params)
        self.mu_centers.append(first_params.pacejka_D_f)
        self.max_errors.append(float("inf"))
        
        # Greedy loop
        pbar = tqdm(
            range(1, cfg.max_basis),
            desc="S²GPT Library",
            disable=not progress
        )
        
        for step in pbar:
            # Compute error for all candidates
            errors = []
            trained_ids = set(id(p) for p in self.trained_params)
            
            for params in candidates:
                if id(params) in trained_ids:
                    errors.append(0.0)
                else:
                    errors.append(self._compute_error(params))
            
            # Greedy selection
            worst_idx = int(np.argmax(errors))
            worst_err = errors[worst_idx]
            worst_params = candidates[worst_idx]
            
            self.max_errors.append(worst_err)
            pbar.set_postfix({
                "err": f"{worst_err:.4f}",
                "μ": f"{worst_params.pacejka_D_f:.2f}"
            })
            
            # Check convergence
            if worst_err < cfg.tol:
                print(f"\nConverged at step {step} with error {worst_err:.6f}")
                break
            
            # Train new specialist
            new_specialist = HSSSpecialist(spec_config)
            loss, epochs_trained = train_specialist(
                new_specialist, worst_params, cfg, self.device, progress=False
            )
            
            self.specialists.append(new_specialist)
            self.trained_params.append(worst_params)
            self.mu_centers.append(worst_params.pacejka_D_f)
        
        # Create ensemble
        ensemble = HSSEnsemble(
            self.specialists,
            mu_centers=self.mu_centers,
            rbf_width=0.15
        )
        
        print(f"\n✓ Library built with {len(self.specialists)} specialists")
        print(f"  Friction centers: {[f'{mu:.2f}' for mu in self.mu_centers]}")
        
        return ensemble
    
    def save_library(self, output_dir: str):
        """Save trained library to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save specialists
        for i, specialist in enumerate(self.specialists):
            torch.save(
                specialist.state_dict(),
                output_path / f"specialist_{i:03d}.pt"
            )
        
        # Save manifest
        manifest = {
            "n_specialists": len(self.specialists),
            "mu_centers": self.mu_centers,
            "config": {
                "hidden_dim": self.config.hidden_dim,
                "n_layers": self.config.n_layers,
            },
            "trained_params": [
                {
                    "pacejka_D_f": p.pacejka_D_f,
                    "pacejka_D_r": p.pacejka_D_r,
                    "pacejka_B_f": p.pacejka_B_f,
                    "pacejka_B_r": p.pacejka_B_r,
                }
                for p in self.trained_params
            ],
            "max_errors": self.max_errors,
        }
        
        with open(output_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Saved library to {output_path}")


def load_library(model_dir: str, device: torch.device) -> HSSEnsemble:
    """Load trained S²GPT-PINN library from disk."""
    model_path = Path(model_dir)
    
    # Load manifest
    with open(model_path / "manifest.json", 'r') as f:
        manifest = json.load(f)
    
    # Create specialists
    config = HSSConfig(
        hidden_dim=manifest["config"]["hidden_dim"],
        n_layers=manifest["config"]["n_layers"]
    )

    specialists = []
    for i in range(manifest["n_specialists"]):
        specialist = HSSSpecialist(config)
        specialist.load_state_dict(
            torch.load(model_path / f"specialist_{i:03d}.pt", map_location=device)
        )
        specialist.to(device)
        specialist.eval()
        specialists.append(specialist)

    # Create ensemble
    ensemble = HSSEnsemble(
        specialists,
        mu_centers=manifest["mu_centers"],
        rbf_width=0.15
    )
    
    print(f"Loaded {len(specialists)} specialists from {model_path}")
    return ensemble


def main():
    parser = argparse.ArgumentParser(description="S²GPT-PINN Training Pipeline")
    parser.add_argument("--test_mode", action="store_true", help="Quick test")
    parser.add_argument("--max_basis", type=int, default=8)
    parser.add_argument("--n_candidates", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="./s2gpt_models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    if args.test_mode:
        print("\n⚡ TEST MODE")
        config = TrainingConfig(
            hidden_dim=32,
            n_layers=2,
            n_samples=512,
            epochs=50,
            batch_size=128,
            n_candidates=5,
            max_basis=3,
            seed=args.seed
        )
    else:
        config = TrainingConfig(
            hidden_dim=64,
            n_layers=3,
            n_samples=args.n_samples if hasattr(args, 'n_samples') else 4096,
            epochs=args.epochs,
            batch_size=256,
            n_candidates=args.n_candidates,
            max_basis=args.max_basis,
            seed=args.seed
        )
    
    print(f"\nConfiguration:")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    
    # Build library
    builder = HSSLibraryBuilder(config, device)
    ensemble = builder.build_library(progress=True)
    
    # Save
    builder.save_library(args.output_dir)
    
    # Test ensemble
    print("\n" + "=" * 60)
    print("Testing ensemble...")
    
    state = torch.tensor([[20.0, 0.5, 0.1]], device=device)
    control = torch.tensor([[0.1, 0.3]], device=device)
    static = torch.tensor([[800.0, 1200.0]], device=device)
    
    for mu in [0.4, 0.7, 1.0]:
        output = ensemble(state, control, static, mu_current=mu)
        print(f"  μ={mu:.1f}: {output.squeeze().tolist()}")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

