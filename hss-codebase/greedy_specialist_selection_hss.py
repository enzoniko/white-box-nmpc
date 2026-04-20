#!/usr/bin/env python3
"""
Greedy Specialist Selection for H-SS (HYDRA Self-Supervised)

Implements S²GPT-PINN Algorithm 2 for offline library building:
1. Generate candidate dynamic parameter sets (friction, aero, stiffness)
2. Initialize sparse collocation set X^m (starts empty)
3. Greedy loop:
   - Train candidates with mass/inertia domain randomization
   - Compute error on sparse grid (magic points)
   - Select worst candidate
   - Update sparse grid via GEIM (basis) and EIM (residuals)
   - Add specialist to library

Also provides online validation by simulating the estimator.

Usage:
    python greedy_specialist_selection_hss.py --test_mode  # Quick test
    python greedy_specialist_selection_hss.py --max_basis 10 --n_candidates 50
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from hydra_pob import (
    VehicleParams, DynamicBicycleOracle,
    geim, eim, GEIMResult, EIMResult, SparseCollocationSet,
    DEFAULT_DEVICE,
)

from h_ss_specialist import HSSSpecialist, HSSLibrary, train_specialist
from inverse_kinematics import (
    generate_fused_trajectory, sample_candidates_dynamic,
    generate_test_trajectory, TrajectoryConfig,
)
from virtual_online_simulator import VirtualOnlineSimulator


@dataclass
class S2GPTConfig:
    """Configuration for S²GPT-PINN algorithm."""
    n_candidates: int = 50       # Number of candidate dynamic parameter sets
    max_basis: int = 8           # Maximum library size
    tol: float = 1e-4            # Convergence tolerance
    dense_grid_size: int = 1024  # |X^{N_h}|: dense evaluation grid size
    
    # Training configuration
    n_samples: int = 4096        # Training samples per specialist
    epochs: int = 300            # Training epochs
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 64
    n_layers: int = 3
    
    # Seed
    seed: int = 42


class S2GPTLibraryBuilder:
    """
    S²GPT-PINN Library Builder for H-SS.
    
    Implements Algorithm 2 from the S²GPT paper with:
    - GEIM for basis function sparsification (Algorithm 3)
    - EIM for residual sparsification (Algorithm 4)
    - Domain randomization for mass/inertia during training
    - Conditioned specialists that accept (state, control, static_params)
    """
    
    def __init__(
        self,
        config: S2GPTConfig,
        device: torch.device = DEFAULT_DEVICE,
    ):
        self.config = config
        self.device = device
        
        # Library state
        self.specialists: List[HSSSpecialist] = []
        self.trained_params: List[VehicleParams] = []
        
        # S²GPT state
        self.W: List[torch.Tensor] = []  # Orthogonalized basis evaluations
        self.sparse_set: SparseCollocationSet = SparseCollocationSet([], [])
        self.geim_betas: List[torch.Tensor] = []
        self.residuals: List[torch.Tensor] = []
        
        # Error tracking for convergence plot
        self.max_errors: List[float] = []
        self.sparse_sizes: List[int] = []
        
        # Dense grid (fixed)
        self._dense_grid: Optional[Tuple[torch.Tensor, ...]] = None
        
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    
    def _get_dense_grid(
        self, 
        ref_params: Optional[VehicleParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate fixed dense grid X^{N_h} for evaluations.
        
        Returns:
            states: [N_h, 3]
            controls: [N_h, 2]
            static_params: [N_h, 2] (nominal values for grid)
        """
        if self._dense_grid is None:
            # Use nominal params for grid generation
            nominal = ref_params or VehicleParams()
            
            # Generate fused trajectory with domain randomization
            states, controls, static, _ = generate_fused_trajectory(
                nominal,
                n_samples=self.config.dense_grid_size,
                device=self.device,
                return_numpy=False,
            )
            self._dense_grid = (states, controls, static)
        
        return self._dense_grid
    
    def _evaluate_specialist(
        self,
        specialist: HSSSpecialist,
        states: torch.Tensor,
        controls: torch.Tensor,
        static_params: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate specialist on given inputs."""
        specialist.eval()
        with torch.no_grad():
            return specialist(states, controls, static_params)
    
    def _compute_oracle_target(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        static_params: torch.Tensor,
        dynamic_params: VehicleParams,
    ) -> torch.Tensor:
        """
        Compute physics accelerations with per-sample static params.
        
        This enables domain randomization - each sample can have different m, Iz.
        """
        n = states.shape[0]
        p = dynamic_params
        
        vx = torch.clamp(states[:, 0], min=0.1)
        vy = states[:, 1]
        omega = states[:, 2]
        delta = controls[:, 0]
        throttle = controls[:, 1]
        
        m = static_params[:, 0]
        iz = static_params[:, 1]
        
        # Slip angles
        alpha_f = delta - torch.atan2(vy + p.lf * omega, vx)
        alpha_r = torch.atan2(p.lr * omega - vy, vx)
        
        # Normal loads (per-sample mass!)
        normal_load_f = m * 9.81 * p.lr / (p.lf + p.lr)
        normal_load_r = m * 9.81 * p.lf / (p.lf + p.lr)
        
        # Pacejka lateral forces
        inner_f = p.pacejka_B_f * alpha_f - p.pacejka_E_f * (
            p.pacejka_B_f * alpha_f - torch.atan(p.pacejka_B_f * alpha_f)
        )
        Ffy = p.pacejka_D_f * normal_load_f * torch.sin(p.pacejka_C_f * torch.atan(inner_f))
        
        inner_r = p.pacejka_B_r * alpha_r - p.pacejka_E_r * (
            p.pacejka_B_r * alpha_r - torch.atan(p.pacejka_B_r * alpha_r)
        )
        Fry = p.pacejka_D_r * normal_load_r * torch.sin(p.pacejka_C_r * torch.atan(inner_r))
        
        # Drivetrain
        Frx = (p.cm1 * throttle) - (p.cm2 * vx) - p.cr0 - (p.cd * vx**2)
        
        # Accelerations (per-sample m, iz!)
        dvx = (Frx - Ffy * torch.sin(delta) + m * vy * omega) / m
        dvy = (Fry + Ffy * torch.cos(delta) - m * vx * omega) / m
        domega = (Ffy * p.lf * torch.cos(delta) - Fry * p.lr) / iz
        
        return torch.stack([dvx, dvy, domega], dim=1)
    
    def _train_specialist(
        self,
        dynamic_params: VehicleParams,
        progress: bool = False,
    ) -> HSSSpecialist:
        """
        Train a conditioned specialist with domain randomization.
        
        The specialist is trained on data where:
        - Dynamic params (μ, Cd, B, C, E) are FIXED to dynamic_params
        - Static params (m, Iz) are RANDOMIZED per sample
        
        This teaches the network a ∝ F/m physics relationship.
        """
        model = HSSSpecialist(
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
        ).to(self.device)
        
        # Generate training data with domain randomization
        states, controls, static_params, accels = generate_fused_trajectory(
            dynamic_params,
            n_samples=self.config.n_samples,
            device=self.device,
            return_numpy=False,
        )
        
        # Train
        final_loss = train_specialist(
            model,
            states.cpu().numpy(),
            controls.cpu().numpy(),
            static_params.cpu().numpy(),
            accels.cpu().numpy(),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            device=self.device,
            progress=progress,
        )
        
        if progress:
            print(f"  Specialist trained, final loss: {final_loss:.6f}")
        
        return model
    
    def _compute_s2gpt_error(
        self,
        dynamic_params: VehicleParams,
        states: torch.Tensor,
        controls: torch.Tensor,
        static_params: torch.Tensor,
    ) -> float:
        """
        Compute S²GPT error indicator Δ_{S²GPT}^{n-1}(μ).
        
        Uses sparse collocation set for efficient evaluation.
        Error = max reconstruction error on sparse grid.
        """
        if not self.specialists:
            return float("inf")
        
        # Get sparse indices
        sparse_idx = self.sparse_set.all_indices
        if not sparse_idx:
            # Fallback: use subset of points
            sparse_idx = list(range(min(100, len(states))))
        
        # Subset to sparse grid
        states_s = states[sparse_idx]
        controls_s = controls[sparse_idx]
        static_s = static_params[sparse_idx]
        
        # Compute target accelerations from physics
        target = self._compute_oracle_target(states_s, controls_s, static_s, dynamic_params)
        
        # Evaluate all basis functions at sparse points
        basis_evals = []
        for specialist, trained_p in zip(self.specialists, self.trained_params):
            pred = self._evaluate_specialist(specialist, states_s, controls_s, static_s)
            basis_evals.append(pred)
        
        # Stack: [M, 3, N_basis]
        H = torch.stack(basis_evals, dim=2)
        M = len(sparse_idx)
        H_flat = H.view(M * 3, len(self.specialists))
        target_flat = target.view(M * 3)
        
        try:
            # Least squares for optimal weights
            sol = torch.linalg.lstsq(H_flat, target_flat).solution
            recon = H_flat @ sol
            err = torch.mean((recon - target_flat) ** 2).item()
        except RuntimeError:
            err = float("inf")
        
        return err
    
    def _apply_geim(self, new_eval: torch.Tensor) -> None:
        """Apply GEIM to update basis sparsification."""
        self.W.append(new_eval)
        
        geim_result = geim(
            xi_list=self.W,
            existing_indices=self.sparse_set.basis_indices,
            existing_betas=self.geim_betas,
        )
        
        self.W[-1] = geim_result.xi
        self.sparse_set.basis_indices = geim_result.sparse_indices
        self.geim_betas.append(geim_result.beta)
    
    def _apply_eim(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        static_params: torch.Tensor,
        dynamic_params: VehicleParams,
    ) -> None:
        """Apply EIM to update residual sparsification."""
        if not self.specialists:
            return
        
        # Compute residual: specialist prediction - oracle target
        target = self._compute_oracle_target(states, controls, static_params, dynamic_params)
        latest_eval = self._evaluate_specialist(
            self.specialists[-1], states, controls, static_params
        )
        residual = latest_eval - target
        
        self.residuals.append(residual)
        
        eim_result = eim(
            residuals=self.residuals,
            existing_indices=self.sparse_set.residual_indices,
        )
        
        self.sparse_set.residual_indices = eim_result.sparse_indices
    
    def build_library(
        self,
        candidates: Optional[List[VehicleParams]] = None,
        progress: bool = True,
    ) -> HSSLibrary:
        """
        Main S²GPT-PINN loop (Algorithm 2).
        
        1. Initialize with first candidate
        2. Greedy loop:
           - Compute error for all candidates on sparse grid
           - Select worst-represented candidate
           - Train specialist with domain randomization
           - Apply GEIM/EIM to update sparse grid
        """
        if candidates is None:
            print(f"Generating {self.config.n_candidates} dynamic candidates...")
            candidates = sample_candidates_dynamic(
                n_candidates=self.config.n_candidates,
                seed=self.config.seed,
            )
        
        # Get dense grid
        states, controls, static_params = self._get_dense_grid(candidates[0])
        
        print(f"\nStarting S²GPT-PINN library building...")
        print(f"  Dense grid size: {len(states)}")
        print(f"  Candidates: {len(candidates)}")
        print(f"  Max basis: {self.config.max_basis}")
        
        # Step 1: Initialize with first candidate
        first_params = candidates[0]
        print(f"\nStep 1: Initialize with first candidate (μ={first_params.pacejka_D_f:.2f})")
        
        first_specialist = self._train_specialist(first_params, progress=progress)
        self.specialists.append(first_specialist)
        self.trained_params.append(first_params)
        
        # Initial GEIM
        first_eval = self._evaluate_specialist(first_specialist, states, controls, static_params)
        self._apply_geim(first_eval)
        
        self.max_errors.append(float("inf"))  # No error for first
        self.sparse_sizes.append(self.sparse_set.size)
        
        # Main greedy loop
        pbar = tqdm(
            range(1, self.config.max_basis),
            desc="S²GPT Library",
            disable=not progress,
        )
        
        for step in pbar:
            # Compute error for all candidates
            errors = []
            already_trained = set(id(p) for p in self.trained_params)
            
            for params in candidates:
                if id(params) in already_trained:
                    errors.append(0.0)
                else:
                    err = self._compute_s2gpt_error(params, states, controls, static_params)
                    errors.append(err)
            
            # Greedy selection
            worst_idx = int(np.argmax(errors))
            worst_err = errors[worst_idx]
            worst_params = candidates[worst_idx]
            
            self.max_errors.append(worst_err)
            pbar.set_postfix({
                "err": f"{worst_err:.4f}",
                "μ": f"{worst_params.pacejka_D_f:.2f}",
                "sparse": self.sparse_set.size,
            })
            
            # Check convergence
            if worst_err < self.config.tol and step > 0:
                print(f"\nConverged at step {step} with error {worst_err:.6f}")
                break
            
            # Train specialist
            specialist = self._train_specialist(worst_params, progress=False)
            self.specialists.append(specialist)
            self.trained_params.append(worst_params)
            
            # Apply GEIM
            new_eval = self._evaluate_specialist(specialist, states, controls, static_params)
            self._apply_geim(new_eval)
            
            # Apply EIM
            self._apply_eim(states, controls, static_params, worst_params)
            
            self.sparse_sizes.append(self.sparse_set.size)
        
        # Build library object
        library = HSSLibrary()
        for i, (specialist, params) in enumerate(zip(self.specialists, self.trained_params)):
            library.add_specialist(
                specialist,
                {
                    "pacejka_D_f": params.pacejka_D_f,
                    "pacejka_D_r": params.pacejka_D_r,
                    "pacejka_B_f": params.pacejka_B_f,
                    "pacejka_B_r": params.pacejka_B_r,
                    "cd": params.cd,
                },
                specialist_id=i,
            )
        
        print(f"\n✓ Library built with {library.num_specialists} specialists")
        print(f"  Sparse grid size: {self.sparse_set.size}")
        print(f"  Final max error: {self.max_errors[-1]:.6f}")
        
        return library
    
    def get_dense_grid_for_plotting(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return dense grid as numpy for plotting."""
        states, controls, static = self._get_dense_grid()
        return states.cpu().numpy(), controls.cpu().numpy()
    
    def get_magic_points(self) -> Tuple[List[int], List[int]]:
        """Return magic point indices for plotting."""
        return self.sparse_set.basis_indices, self.sparse_set.residual_indices


def run_online_validation(
    library: HSSLibrary,
    test_params: VehicleParams,
    locked_mass: float = 1000.0,
    locked_inertia: float = 1500.0,
    n_steps: int = 300,
    scenario: str = "dry_to_wet",
    device: torch.device = DEFAULT_DEVICE,
) -> Dict:
    """
    Run online validation simulating the Phase 2 estimator.
    
    Args:
        library: Trained H-SS library
        test_params: Base vehicle parameters for test
        locked_mass: Fixed mass from Phase 1
        locked_inertia: Fixed inertia from Phase 1
        n_steps: Simulation steps
        scenario: Test scenario
        device: Torch device
        
    Returns:
        Dict with weights history, uncertainties, and trajectories
    """
    print(f"\n{'='*60}")
    print(f"Online Validation: {scenario}")
    print(f"  Locked mass: {locked_mass} kg")
    print(f"  Locked inertia: {locked_inertia} kg·m²")
    print(f"  Library size: {library.num_specialists}")
    print(f"{'='*60}")
    
    # Generate test trajectory
    states, controls, static_params, true_accels = generate_test_trajectory(
        test_params,
        n_steps=n_steps,
        locked_mass=locked_mass,
        locked_inertia=locked_inertia,
        scenario=scenario,
        device=device,
    )
    
    # Create online simulator
    simulator = VirtualOnlineSimulator(
        library=library,
        window_size=25,
        dirichlet_alpha=1.0,
        device=device,
    )
    
    # Run online estimation
    weights_history = []
    uncertainty_history = []
    predicted_accels = []
    
    for t in tqdm(range(n_steps), desc="Online estimation", leave=False):
        state = states[t].cpu().numpy()
        control = controls[t].cpu().numpy()
        static = static_params[t].cpu().numpy()
        y_obs = true_accels[t].cpu().numpy()
        
        # Predict with locked static params
        pred = simulator.predict_next(state, control, static)
        predicted_accels.append(pred)
        
        # Update posterior
        simulator.update_posterior(state, control, static, y_obs)
        
        # Record
        weights_history.append(simulator.get_weights().copy())
        uncertainty_history.append(simulator.get_uncertainty())
    
    # Convert to arrays
    weights_history = np.array(weights_history)
    uncertainty_history = np.array(uncertainty_history)
    predicted_accels = np.array(predicted_accels)
    
    # Compute metrics
    accel_error = np.mean((predicted_accels - true_accels.cpu().numpy()) ** 2)
    
    print(f"\nValidation Results:")
    print(f"  Mean squared acceleration error: {accel_error:.6f}")
    print(f"  Final uncertainty: {uncertainty_history[-1]:.4f}")
    print(f"  Dominant specialist: {np.argmax(weights_history[-1])}")
    
    return {
        "weights_history": weights_history,
        "uncertainty_history": uncertainty_history,
        "predicted_accels": predicted_accels,
        "true_accels": true_accels.cpu().numpy(),
        "states": states.cpu().numpy(),
        "controls": controls.cpu().numpy(),
        "accel_error": accel_error,
    }


def plot_s2gpt_convergence(
    max_errors: List[float],
    output_path: str = "s2gpt_convergence.png",
) -> None:
    """Plot S²GPT convergence (error vs library size)."""
    plt.figure(figsize=(8, 5))
    
    steps = range(1, len(max_errors) + 1)
    plt.semilogy(steps, max_errors, 'b-o', linewidth=2, markersize=6)
    
    plt.xlabel("Library Size (N specialists)", fontsize=12)
    plt.ylabel("Max Reconstruction Error (log)", fontsize=12)
    plt.title("S²GPT-PINN Convergence", fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot: {output_path}")


def plot_online_adaptation(
    results: Dict,
    n_specialists: int,
    output_path: str = "online_adaptation.png",
) -> None:
    """Plot online adaptation results (stacked weights + uncertainty)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    weights = results["weights_history"]
    uncertainty = results["uncertainty_history"]
    t = np.arange(len(weights))
    
    # Stacked weights
    ax1 = axes[0]
    ax1.stackplot(t, weights.T, alpha=0.8)
    ax1.set_ylabel("Specialist Weights", fontsize=11)
    ax1.set_title("Online Adaptation: Weight Evolution", fontsize=12)
    ax1.legend([f"S{i}" for i in range(n_specialists)], 
               loc='upper right', fontsize=8, ncol=min(4, n_specialists))
    ax1.set_ylim(0, 1)
    
    # Uncertainty trace
    ax2 = axes[1]
    ax2.plot(t, uncertainty, 'r-', linewidth=1.5)
    ax2.set_ylabel("Trace(Σ)", fontsize=11)
    ax2.set_title("Posterior Uncertainty", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Acceleration error
    ax3 = axes[2]
    accel_err = np.mean((results["predicted_accels"] - results["true_accels"]) ** 2, axis=1)
    ax3.plot(t, accel_err, 'b-', linewidth=1)
    ax3.set_ylabel("MSE Accel", fontsize=11)
    ax3.set_xlabel("Time Step", fontsize=11)
    ax3.set_title("Instantaneous Acceleration Error", fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved adaptation plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="H-SS Greedy Specialist Selection with S²GPT-PINN")
    parser.add_argument("--test_mode", action="store_true", help="Quick test with reduced config")
    parser.add_argument("--max_basis", type=int, default=8, help="Maximum library size")
    parser.add_argument("--n_candidates", type=int, default=50, help="Number of candidates")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--n_samples", type=int, default=4096, help="Training samples")
    parser.add_argument("--output_dir", type=str, default="./models_hss", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    if args.test_mode:
        print("\n⚡ TEST MODE: Reduced configuration")
        config = S2GPTConfig(
            n_candidates=5,
            max_basis=3,
            dense_grid_size=256,
            n_samples=512,
            epochs=50,
            batch_size=128,
            hidden_dim=32,
            n_layers=2,
            seed=args.seed,
        )
    else:
        config = S2GPTConfig(
            n_candidates=args.n_candidates,
            max_basis=args.max_basis,
            dense_grid_size=1024,
            n_samples=args.n_samples,
            epochs=args.epochs,
            batch_size=256,
            hidden_dim=64,
            n_layers=3,
            seed=args.seed,
        )
    
    print(f"\nConfiguration:")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # PART 1: Offline S²GPT-PINN Library Building
    # ========================================
    print(f"\n{'='*60}")
    print("PART 1: Offline Library Building (S²GPT-PINN)")
    print(f"{'='*60}")
    
    builder = S2GPTLibraryBuilder(config, device=device)
    library = builder.build_library(progress=True)
    
    # Save library
    library.save(str(output_dir))
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Plot convergence
    plot_s2gpt_convergence(
        builder.max_errors,
        str(output_dir / "s2gpt_convergence.png"),
    )
    
    # ========================================
    # PART 2: Online Validation
    # ========================================
    print(f"\n{'='*60}")
    print("PART 2: Online Validation")
    print(f"{'='*60}")
    
    # Test scenario: Heavy car (1500kg) on dry-to-wet transition
    test_params = VehicleParams(
        m=1500.0,  # Will be overridden by locked_mass
        Iz=2500.0,  # Will be overridden by locked_inertia
        pacejka_D_f=0.9,  # Nominal friction (will change in scenario)
        pacejka_D_r=0.9,
    )
    
    results = run_online_validation(
        library,
        test_params,
        locked_mass=1500.0,
        locked_inertia=2500.0,
        n_steps=200 if args.test_mode else 500,
        scenario="dry_to_wet",
        device=device,
    )
    
    # Plot adaptation
    plot_online_adaptation(
        results,
        library.num_specialists,
        str(output_dir / "online_adaptation.png"),
    )
    
    # Save validation results
    np.savez(
        output_dir / "validation_results.npz",
        weights_history=results["weights_history"],
        uncertainty_history=results["uncertainty_history"],
        predicted_accels=results["predicted_accels"],
        true_accels=results["true_accels"],
    )
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Library size: {library.num_specialists} specialists")
    print(f"Sparse grid: {builder.sparse_set.size} points")
    print(f"Final S²GPT error: {builder.max_errors[-1]:.6f}")
    print(f"Validation accel MSE: {results['accel_error']:.6f}")
    print(f"\nOutputs saved to: {output_dir}/")
    
    return library


if __name__ == "__main__":
    main()
