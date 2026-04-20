#!/usr/bin/env python3
"""
H-SS Greedy Specialist Selection

This module implements the greedy basis construction algorithm for H-SS (HYDRA Self-Supervised):

ALGORITHM: Greedy Specialist Selection
─────────────────────────────────────────────────────────────────────────────
Input: Candidate parameter set P = {μ₁, ..., μ_M}, tolerance ε, max specialists K_max
Output: Library of specialists {Ψ₁, ..., Ψ_K}

1. Initialize: Library = {}, Sparse grid X^m = {}
2. For k = 1, 2, ... until convergence or k = K_max:
   a. For each candidate μ ∈ P:
      - Compute reconstruction error e(μ) = ||f(·;μ) - Σᵢ wᵢ·Ψᵢ(·)||
   b. Select worst: μ* = argmax_μ e(μ)
   c. If e(μ*) < ε: STOP (converged)
   d. Train specialist Ψ_k on data from oracle with params μ*
   e. Add Ψ_k to library
   f. Update sparse grid X^m via GEIM (optional)
   g. Remove μ* from P
3. Return Library
─────────────────────────────────────────────────────────────────────────────

The key insight: We train specialists where the current library is WORST,
ensuring maximum coverage with minimum specialists.
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

# Local imports - handle both relative and absolute imports
try:
    from .specialist import HSSConfig, HSSSpecialist, HSSEnsemble
except ImportError:
    from specialist import HSSConfig, HSSSpecialist, HSSEnsemble
from calibration import VehicleParamsConfig, generate_specialist_param_sets


@dataclass
class GreedySelectionConfig:
    """Configuration for greedy specialist selection."""
    # Candidate generation
    n_candidates: int = 20          # Number of candidate parameter sets
    friction_range: Tuple[float, float] = (0.3, 1.2)
    
    # Stopping criteria
    max_specialists: int = 10       # Maximum number of specialists
    tolerance: float = 0.5          # RMSE tolerance for convergence
    
    # Training config
    n_train_samples: int = 4000     # Training samples per specialist
    n_test_samples: int = 500       # Test samples for error evaluation
    epochs: int = 300               # Training epochs per specialist
    batch_size: int = 128
    lr: float = 1e-3

    # Early stopping
    early_stopping_patience: int = 30  # Stop if no improvement for this many epochs
    early_stopping_min_delta: float = 1e-6  # Minimum improvement to count as progress
    validation_split: float = 0.1  # Fraction of data for validation
    
    # Network config
    hidden_dim: int = 64
    n_layers: int = 3
    
    # Misc
    device: str = "cpu"
    seed: int = 42
    verbose: bool = True


class PhysicsOracle:
    """
    Physics-based oracle (ground truth generator).
    
    Generates training data for specialists and computes errors.
    """
    
    def __init__(self, params: VehicleParamsConfig, device: torch.device):
        self.params = params
        self.device = device
    
    def generate_data(
        self,
        n_samples: int,
        state_ranges: Optional[Dict] = None,
        control_ranges: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate (state, control, static, acceleration) tuples.
        
        Returns:
            states: [N, 3] - vx, vy, omega
            controls: [N, 2] - delta, throttle
            static: [N, 2] - m, Iz
            accels: [N, 3] - dvx, dvy, domega
        """
        if state_ranges is None:
            state_ranges = {
                'vx': (5.0, 35.0),
                'vy': (-2.0, 2.0),
                'omega': (-1.0, 1.0)
            }
        if control_ranges is None:
            control_ranges = {
                'delta': (-0.5, 0.5),
                'throttle': (-0.5, 0.8)
            }
        
        p = self.params
        
        # Sample states
        vx = np.random.uniform(*state_ranges['vx'], n_samples)
        vy = np.random.uniform(*state_ranges['vy'], n_samples)
        omega = np.random.uniform(*state_ranges['omega'], n_samples)
        
        # Sample controls
        delta = np.random.uniform(*control_ranges['delta'], n_samples)
        throttle = np.random.uniform(*control_ranges['throttle'], n_samples)
        
        # Static params (can randomize mass/inertia for domain randomization)
        mass = np.ones(n_samples) * p.m
        Iz = np.ones(n_samples) * p.Iz
        
        # Compute accelerations
        accels = []
        for i in range(n_samples):
            state = np.array([vx[i], vy[i], omega[i]])
            control = np.array([delta[i], throttle[i]])
            accel = self._compute_acceleration(state, control, mass[i], Iz[i])
            accels.append(accel)
        accels = np.array(accels)
        
        # Convert to tensors
        states = torch.tensor(np.column_stack([vx, vy, omega]), dtype=torch.float32, device=self.device)
        controls = torch.tensor(np.column_stack([delta, throttle]), dtype=torch.float32, device=self.device)
        static = torch.tensor(np.column_stack([mass, Iz]), dtype=torch.float32, device=self.device)
        accels = torch.tensor(accels, dtype=torch.float32, device=self.device)
        
        return states, controls, static, accels
    
    def _compute_acceleration(
        self,
        state: np.ndarray,
        control: np.ndarray,
        m: float,
        Iz: float
    ) -> np.ndarray:
        """Compute acceleration using Pacejka tire model."""
        p = self.params
        
        vx = max(state[0], 0.1)
        vy = state[1]
        omega = state[2]
        delta = control[0]
        throttle = control[1]
        
        # Slip angles
        alpha_f = delta - np.arctan2(vy + p.lf * omega, vx)
        alpha_r = np.arctan2(p.lr * omega - vy, vx)
        
        # Normal loads
        Fz_f = m * 9.81 * p.lr / (p.lf + p.lr)
        Fz_r = m * 9.81 * p.lf / (p.lf + p.lr)
        
        # Pacejka lateral forces
        def pacejka(alpha, B, C, D, E, Fz):
            inner = B * alpha - E * (B * alpha - np.arctan(B * alpha))
            return D * Fz * np.sin(C * np.arctan(inner))
        
        Ffy = pacejka(alpha_f, p.pacejka_B_f, p.pacejka_C_f, p.pacejka_D_f, p.pacejka_E_f, Fz_f)
        Fry = pacejka(alpha_r, p.pacejka_B_r, p.pacejka_C_r, p.pacejka_D_r, p.pacejka_E_r, Fz_r)
        
        # Drivetrain
        Frx = p.cm1 * throttle - p.cm2 * vx - p.cr0 - p.cd * vx**2
        
        # Accelerations
        dvx = (Frx - Ffy * np.sin(delta) + m * vy * omega) / m
        dvy = (Fry + Ffy * np.cos(delta) - m * vx * omega) / m
        domega = (Ffy * p.lf * np.cos(delta) - Fry * p.lr) / Iz
        
        return np.array([dvx, dvy, domega])


def train_specialist_supervised(
    specialist: HSSSpecialist,
    states: torch.Tensor,
    controls: torch.Tensor,
    static: torch.Tensor,
    targets: torch.Tensor,
    config: GreedySelectionConfig,
    verbose: bool = False,
) -> Tuple[float, List[float], int]:
    """
    Train a specialist with supervised learning on physics oracle data and early stopping.

    Uses early stopping to train until the model literally cannot improve anymore.

    Returns:
        final_val_loss: Final validation loss
        loss_history: Loss per epoch (validation loss)
        epochs_trained: Number of epochs actually trained
    """
    device = torch.device(config.device)
    specialist.to(device)

    # Split data into train/validation
    n_samples = states.shape[0]
    n_val = max(32, int(n_samples * config.validation_split))
    n_train = n_samples - n_val

    # Split indices
    indices = torch.randperm(n_samples, device=device)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Split data
    train_states = states[train_indices]
    train_controls = controls[train_indices]
    train_static = static[train_indices]
    train_targets = targets[train_indices]

    val_states = states[val_indices]
    val_controls = controls[val_indices]
    val_static = static[val_indices]
    val_targets = targets[val_indices]

    # Training setup
    specialist.train()
    optimizer = torch.optim.Adam(specialist.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30
    )
    criterion = nn.MSELoss()

    n_train_samples = train_states.shape[0]
    n_train_batches = (n_train_samples + config.batch_size - 1) // config.batch_size
    n_val_samples = val_states.shape[0]

    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_weights = specialist.state_dict().copy()
    patience_counter = 0
    epochs_without_improvement = 0

    loss_history = []
    iterator = range(config.epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training", leave=False)

    actual_epochs_trained = 0

    for epoch in iterator:
        actual_epochs_trained = epoch + 1

        # Training phase
        specialist.train()
        train_loss = 0.0

        # Shuffle training data
        train_perm = torch.randperm(n_train_samples, device=device)
        train_states_shuffled = train_states[train_perm]
        train_controls_shuffled = train_controls[train_perm]
        train_static_shuffled = train_static[train_perm]
        train_targets_shuffled = train_targets[train_perm]

        for batch_idx in range(n_train_batches):
            start = batch_idx * config.batch_size
            end = min(start + config.batch_size, n_train_samples)

            batch_states = train_states_shuffled[start:end]
            batch_controls = train_controls_shuffled[start:end]
            batch_static = train_static_shuffled[start:end]
            batch_targets = train_targets_shuffled[start:end]

            optimizer.zero_grad()
            pred = specialist(batch_states, batch_controls, batch_static)
            loss = criterion(pred, batch_targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(specialist.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / n_train_batches

        # Validation phase
        specialist.eval()
        with torch.no_grad():
            val_pred = specialist(val_states, val_controls, val_static)
            val_loss = criterion(val_pred, val_targets).item()

        loss_history.append(val_loss)
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss - config.early_stopping_min_delta:
            # Significant improvement
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = specialist.state_dict().copy()
            patience_counter = 0
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            patience_counter += 1

        if verbose and epoch % 20 == 0:
            tqdm.write(f"  Epoch {epoch}: train={avg_train_loss:.6f}, val={val_loss:.6f}, best={best_val_loss:.6f}, patience={patience_counter}")

        # Check early stopping condition
        if patience_counter >= config.early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
                print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
            break

    # Restore best weights
    specialist.load_state_dict(best_weights)
    specialist.eval()

    return best_val_loss, loss_history, actual_epochs_trained


def compute_reconstruction_error(
    ensemble: HSSEnsemble,
    oracle: PhysicsOracle,
    n_samples: int,
    device: torch.device,
) -> float:
    """
    Compute RMSE between ensemble prediction and oracle ground truth.
    
    Uses Mode B style calibration: find optimal weights via least squares,
    then compute residual error.
    """
    # Generate test data
    states, controls, static, targets = oracle.generate_data(n_samples)
    
    # Get predictions from all specialists
    n_specialists = len(ensemble.specialists)
    specialist_preds = []
    
    with torch.no_grad():
        for specialist in ensemble.specialists:
            pred = specialist(states, controls, static)
            specialist_preds.append(pred.cpu().numpy())
    
    # Stack: [N, K, 3]
    Phi = np.stack(specialist_preds, axis=1)
    y = targets.cpu().numpy()  # [N, 3]
    
    # Solve for optimal weights per output dimension
    # w* = argmin ||Φw - y||²
    weights = []
    for d in range(3):
        Phi_d = Phi[:, :, d]  # [N, K]
        y_d = y[:, d]         # [N,]
        
        # Ridge regression
        A = Phi_d.T @ Phi_d + 1e-4 * np.eye(n_specialists)
        b = Phi_d.T @ y_d
        w_d = np.linalg.solve(A, b)
        weights.append(w_d)
    
    # Average weights
    w = np.mean(weights, axis=0)
    w = np.maximum(w, 0)
    w = w / (w.sum() + 1e-10)
    
    # Compute prediction with optimal weights
    y_pred = np.einsum('nkd,k->nd', Phi, w)
    
    # RMSE
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    return rmse


class GreedySpecialistSelector:
    """
    Greedy selection of optimal specialist basis.

    Implements the H-SS greedy algorithm for building a minimal library
    that covers the parameter space with specified accuracy.
    """
    
    def __init__(self, config: GreedySelectionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Results
        self.specialists: List[HSSSpecialist] = []
        self.specialist_params: List[VehicleParamsConfig] = []
        self.selection_history: List[Dict] = []
        self.error_history: List[float] = []
    
    def _generate_candidates(self) -> List[VehicleParamsConfig]:
        """Generate candidate parameter sets."""
        return generate_specialist_param_sets(
            n_specialists=self.config.n_candidates,
            friction_range=self.config.friction_range,
            vary_pacejka_bc=True,
            vary_drivetrain=False,
        )
    
    def _create_ensemble(self) -> Optional[HSSEnsemble]:
        """Create ensemble from current specialists."""
        if not self.specialists:
            return None
        
        mu_centers = [p.pacejka_D_f for p in self.specialist_params]
        return HSSEnsemble(
            [s for s in self.specialists],  # Copy list
            mu_centers=mu_centers
        ).to(self.device)
    
    def build_library(self) -> HSSEnsemble:
        """
        Build specialist library using greedy selection.
        
        The algorithm:
        1. Generate candidate parameter sets
        2. Iteratively:
           a. Find candidate with highest reconstruction error
           b. Train specialist on that candidate
           c. Add to library
           d. Stop if error < tolerance or max specialists reached
        """
        config = self.config
        
        if config.verbose:
            print("=" * 70)
            print("GREEDY SPECIALIST SELECTION")
            print("=" * 70)
            print(f"\nConfiguration:")
            print(f"  Candidates: {config.n_candidates}")
            print(f"  Max specialists: {config.max_specialists}")
            print(f"  Tolerance: {config.tolerance}")
            print(f"  Hidden dim: {config.hidden_dim}")
            print(f"  Layers: {config.n_layers}")
        
        # Generate candidates
        candidates = self._generate_candidates()
        remaining = list(range(len(candidates)))
        
        if config.verbose:
            print(f"\n  Generated {len(candidates)} candidate parameter sets")
            print(f"  Friction range: {[f'{c.pacejka_D_f:.2f}' for c in candidates[:5]]}...")
        
        # Greedy selection loop
        iteration = 0
        
        while iteration < config.max_specialists and remaining:
            iteration += 1
            
            if config.verbose:
                print(f"\n--- Iteration {iteration}/{config.max_specialists} ---")
            
            # Compute error for each remaining candidate
            errors = []
            
            for idx in remaining:
                candidate = candidates[idx]
                oracle = PhysicsOracle(candidate, self.device)
                
                if self.specialists:
                    # Create temporary ensemble and compute error
                    ensemble = self._create_ensemble()
                    error = compute_reconstruction_error(
                        ensemble, oracle,
                        n_samples=config.n_test_samples,
                        device=self.device
                    )
                else:
                    # No specialists yet, error is infinite
                    error = float('inf')
                
                errors.append((idx, error))
                
                if config.verbose and len(remaining) <= 10:
                    print(f"    Candidate μ={candidate.pacejka_D_f:.2f}: error={error:.4f}")
            
            # Select worst candidate (highest error)
            worst_idx, worst_error = max(errors, key=lambda x: x[1])
            worst_candidate = candidates[worst_idx]
            
            if config.verbose:
                print(f"\n  Selected: μ={worst_candidate.pacejka_D_f:.2f} (error={worst_error:.4f})")
            
            # Check convergence
            if worst_error < config.tolerance and len(self.specialists) > 0:
                if config.verbose:
                    print(f"\n  ✓ Converged! Error {worst_error:.4f} < tolerance {config.tolerance}")
                break
            
            # Train specialist on worst candidate
            if config.verbose:
                print(f"  Training specialist {len(self.specialists) + 1}...")
            
            oracle = PhysicsOracle(worst_candidate, self.device)
            states, controls, static, targets = oracle.generate_data(config.n_train_samples)

            spec_config = HSSConfig(
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers
            )
            specialist = HSSSpecialist(spec_config)
            
            final_loss, loss_hist, epochs_trained = train_specialist_supervised(
                specialist, states, controls, static, targets,
                config, verbose=config.verbose
            )
            
            if config.verbose:
                print(f"    Final training loss: {final_loss:.6f}")
            
            # Add to library
            self.specialists.append(specialist)
            self.specialist_params.append(worst_candidate)
            
            # Record history
            self.selection_history.append({
                'iteration': iteration,
                'selected_mu': worst_candidate.pacejka_D_f,
                'error_before': worst_error,
                'training_loss': final_loss,
            })
            
            # Compute new error with updated library
            ensemble = self._create_ensemble()
            new_error = 0.0
            for idx in remaining:
                oracle = PhysicsOracle(candidates[idx], self.device)
                err = compute_reconstruction_error(
                    ensemble, oracle,
                    n_samples=config.n_test_samples // 2,
                    device=self.device
                )
                new_error = max(new_error, err)
            
            self.error_history.append(new_error)
            
            if config.verbose:
                print(f"    Max error after addition: {new_error:.4f}")
            
            # Remove selected candidate
            remaining.remove(worst_idx)
        
        # Final summary
        if config.verbose:
            print("\n" + "=" * 70)
            print("GREEDY SELECTION COMPLETE")
            print("=" * 70)
            print(f"\n  Specialists selected: {len(self.specialists)}")
            print(f"  Friction values: {[f'{p.pacejka_D_f:.2f}' for p in self.specialist_params]}")
            print(f"  Final max error: {self.error_history[-1] if self.error_history else 'N/A'}")
        
        return self._create_ensemble()
    
    def get_selection_summary(self) -> Dict:
        """Get summary of selection process."""
        return {
            'n_specialists': len(self.specialists),
            'specialist_params': [
                {
                    'mu_f': p.pacejka_D_f,
                    'mu_r': p.pacejka_D_r,
                    'B_f': p.pacejka_B_f,
                    'C_f': p.pacejka_C_f,
                } for p in self.specialist_params
            ],
            'selection_history': self.selection_history,
            'error_history': self.error_history,
            'converged': (
                len(self.error_history) > 0 and 
                self.error_history[-1] < self.config.tolerance
            ),
        }


def run_greedy_selection_demo():
    """Demonstrate greedy specialist selection."""
    print("\n" + "=" * 70)
    print("S²GPT-PINN GREEDY SPECIALIST SELECTION DEMO")
    print("=" * 70)
    
    config = GreedySelectionConfig(
        n_candidates=15,
        max_specialists=6,
        tolerance=0.3,
        n_train_samples=3000,
        n_test_samples=300,
        epochs=200,
        hidden_dim=64,
        n_layers=3,
        verbose=True,
    )
    
    selector = GreedySpecialistSelector(config)
    ensemble = selector.build_library()
    
    # Print summary
    summary = selector.get_selection_summary()
    
    print("\n" + "=" * 70)
    print("SELECTION SUMMARY")
    print("=" * 70)
    print(f"\nOptimal number of specialists: {summary['n_specialists']}")
    print(f"Converged: {summary['converged']}")
    print(f"\nSelected specialists:")
    for i, p in enumerate(summary['specialist_params']):
        print(f"  {i+1}. μ={p['mu_f']:.2f}, B={p['B_f']:.1f}, C={p['C_f']:.2f}")
    
    print(f"\nError history: {[f'{e:.4f}' for e in summary['error_history']]}")
    
    # Test final ensemble
    print("\n--- Testing Final Ensemble ---")
    test_frictions = [0.4, 0.7, 1.0]
    
    for mu in test_frictions:
        test_params = VehicleParamsConfig(pacejka_D_f=mu, pacejka_D_r=mu)
        oracle = PhysicsOracle(test_params, torch.device(config.device))
        error = compute_reconstruction_error(
            ensemble, oracle,
            n_samples=500,
            device=torch.device(config.device)
        )
        print(f"  μ={mu:.1f}: RMSE={error:.4f}")
    
    return ensemble, selector


if __name__ == "__main__":
    ensemble, selector = run_greedy_selection_demo()

