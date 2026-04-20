#!/usr/bin/env python3
"""
Virtual Online Simulator for H-SS (HYDRA Self-Supervised)

Implements TWO prediction methods:
1. Linear Blending (LPV): ŷ = Σ wᵢ · Ψᵢ(x, u, m, Iz)  [Legacy, has LPV error]
2. Parameter Re-Projection: ŷ = Oracle(x, u; m, Iz, Σ wᵢ · Φᵢ)  [Physically exact]

The Re-Projection approach eliminates the LPV assumption by:
- Blending PARAMETERS (friction, aero, tire coefficients) using weights
- Querying the Physics Oracle with the blended parameters
- This is physically exact by definition

Flow:
1. Receive observation y_obs (acceleration from sensors)
2. Update Dirichlet posterior to get weights w
3. Blend dynamic parameters: Φ_blend = Σ wᵢ · Φᵢ
4. Query Oracle: ŷ = Oracle(x, u; m_locked, Iz_locked, Φ_blend)
"""

import numpy as np
import time
import torch
import sys
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Union, Dict
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from hydra_pob import VehicleParams, DynamicBicycleOracle

# Import H-SS components
from h_ss_specialist import HSSLibrary, HSSSpecialist


class VirtualOnlineSimulator:
    """
    Phase 2 Online Estimator for H-SS with Parameter Re-Projection.
    
    Implements both Linear Blending (LPV) and Parameter Re-Projection.
    Re-Projection is physically exact and should be preferred.
    
    Key Methods:
        predict_linear(state, control, static_params) -> LPV blended acceleration
        predict_physics_based(state, control, static_params) -> Re-Projected acceleration
        get_blended_dynamic_params() -> weighted average of dynamic parameters
    """

    def __init__(
        self,
        library: Optional[HSSLibrary] = None,
        window_size: int = 50,
        dirichlet_alpha: float = 1.0,
        noise_precision: float = 1.0,
        device: torch.device = None,
    ):
        """
        Initialize the H-SS online simulator.

        Args:
            library: HSSLibrary with conditioned specialists
            window_size: Number of most recent samples to maintain
            dirichlet_alpha: Prior concentration (uniform = 1.0)
            noise_precision: Likelihood precision (β) for observation noise
            device: Torch device for specialist inference
        """
        self.library = library
        self.window_size = window_size
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_precision = noise_precision
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Window buffer: deque of (specialist_predictions, target_observation) tuples
        self.window_buffer: deque = deque(maxlen=window_size)
        
        # Posterior parameters (Dirichlet)
        self.mu: Optional[np.ndarray] = None  # Posterior mean weights
        self.Sigma: Optional[np.ndarray] = None  # Dirichlet covariance
        self.alpha: Optional[np.ndarray] = None  # Variational parameters
        
        # Performance tracking (must be initialized before initialize_for_library)
        self.update_latencies: List[float] = []
        self.uncertainties: List[float] = []
        self.optimization_failures: int = 0
        
        # Initialize if library provided
        if library is not None:
            self.initialize_for_library(library)

    def initialize_for_library(self, library: HSSLibrary):
        """
        Initialize posterior for H-SS library.
        
        Args:
            library: HSSLibrary with conditioned specialists
        """
        self.library = library
        num_specialists = library.num_specialists
        
        # Initialize uniform Dirichlet prior
        self.alpha = np.ones(num_specialists) * self.dirichlet_alpha
        S = np.sum(self.alpha)
        self.mu = self.alpha / S
        self.Sigma = (np.diag(self.mu) - np.outer(self.mu, self.mu)) / (S + 1)
        
        self.window_buffer.clear()
        self.update_latencies.clear()
        self.uncertainties.clear()
        self.optimization_failures = 0

    def _get_specialist_predictions(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray,
    ) -> np.ndarray:
        """
        Query all specialists with locked static parameters.
        
        Args:
            state: [3,] array (vx, vy, omega)
            control: [2,] array (delta, throttle)
            static_params: [2,] array (locked_m, locked_Iz) from Phase 1
            
        Returns:
            predictions: [3, num_specialists] array
        """
        return self.library.get_predictions(state, control, static_params)

    def get_blended_dynamic_params(self) -> Dict[str, float]:
        """
        Compute weighted average of dynamic parameters using current weights.
        
        Returns:
            Dict of blended dynamic parameters: {param_name: blended_value}
            
        This is the KEY to Parameter Re-Projection:
            Φ_blend = Σᵢ wᵢ · Φᵢ
        """
        if self.mu is None:
            raise RuntimeError("Simulator not initialized")
        
        # Dynamic parameters to blend (from specialist manifest)
        blend_keys = [
            'pacejka_B_f', 'pacejka_B_r',
            'pacejka_C_f', 'pacejka_C_r',
            'pacejka_D_f', 'pacejka_D_r',
            'pacejka_E_f', 'pacejka_E_r',
            'cd', 'cm1',
        ]
        
        # Default values for missing params
        defaults = VehicleParams()
        
        blended = {}
        for key in blend_keys:
            blended[key] = 0.0
            for i in range(self.library.num_specialists):
                params = self.library.param_manifest.get(str(i), {})
                value = params.get(key, getattr(defaults, key, 0))
                blended[key] += self.mu[i] * value
        
        return blended

    def get_blended_vehicle_params(
        self,
        locked_m: float,
        locked_Iz: float,
    ) -> VehicleParams:
        """
        Create VehicleParams with locked static params + blended dynamic params.
        
        Args:
            locked_m: Fixed mass from Phase 1 (kg)
            locked_Iz: Fixed inertia from Phase 1 (kg·m²)
            
        Returns:
            VehicleParams with combined static (fixed) and dynamic (blended) params
        """
        blended_dynamic = self.get_blended_dynamic_params()
        
        return VehicleParams(
            m=locked_m,
            Iz=locked_Iz,
            **blended_dynamic
        )

    def predict_physics_based(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray,
    ) -> np.ndarray:
        """
        Predict acceleration using PARAMETER RE-PROJECTION.
        
        This is physically exact by definition:
            1. Blend dynamic parameters: Φ_blend = Σᵢ wᵢ · Φᵢ
            2. Query Oracle: ŷ = Oracle(x, u; m_locked, Iz_locked, Φ_blend)
        
        Args:
            state: [3,] array (vx, vy, omega)
            control: [2,] array (delta, throttle)
            static_params: [2,] array (locked_m, locked_Iz)
            
        Returns:
            prediction: [3,] array (dvx, dvy, domega)
        """
        if self.mu is None:
            raise RuntimeError("Simulator not initialized. Call initialize_for_library() first.")
        
        locked_m, locked_Iz = static_params[0], static_params[1]
        
        # Step 1: Blend dynamic parameters using weights
        blended_params = self.get_blended_vehicle_params(locked_m, locked_Iz)
        
        # Step 2: Query Physics Oracle with blended parameters
        oracle = DynamicBicycleOracle(blended_params, device=self.device)
        
        # Convert to torch tensors
        state_t = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        control_t = torch.from_numpy(control).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            accel = oracle.accelerations(state_t, control_t)
        
        return accel.squeeze().cpu().numpy()

    def predict_linear(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray,
    ) -> np.ndarray:
        """
        Predict acceleration using LINEAR BLENDING (LPV approach).
        
        This has LPV error due to tire nonlinearity:
            ŷ = Σᵢ wᵢ · Ψᵢ(x, u, m, Iz)
        
        Args:
            state: [3,] array (vx, vy, omega)
            control: [2,] array (delta, throttle)
            static_params: [2,] array (locked_m, locked_Iz)
            
        Returns:
            prediction: [3,] array (dvx, dvy, domega)
        """
        if self.mu is None:
            raise RuntimeError("Simulator not initialized. Call initialize_for_library() first.")
        
        # Get predictions from all specialists with locked static params
        # Shape: [3, num_specialists]
        preds = self._get_specialist_predictions(state, control, static_params)
        
        # Blend using posterior mean weights (LPV assumption)
        # prediction = sum_i w_i * pred_i for each output dimension
        prediction = preds @ self.mu  # [3,]
        
        return prediction

    def predict_next(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray,
    ) -> np.ndarray:
        """
        Default prediction method - uses Parameter Re-Projection.
        
        For legacy LPV behavior, use predict_linear() instead.
        """
        return self.predict_physics_based(state, control, static_params)

    def update_posterior(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray,
        y_obs: np.ndarray,
    ) -> float:
        """
        Update posterior using new observation.
        
        Adds observation to window, then recomputes Dirichlet posterior.
        
        Args:
            state: [3,] array (vx, vy, omega)
            control: [2,] array (delta, throttle)
            static_params: [2,] array (locked_m, locked_Iz)
            y_obs: [3,] array (observed acceleration from sensors)
            
        Returns:
            latency: Time taken for update (seconds)
        """
        start_time = time.time()
        
        # Get specialist predictions with locked static params
        preds = self._get_specialist_predictions(state, control, static_params)
        
        # Store in window buffer: (predictions [3, N_spec], observation [3,])
        y_obs = np.asarray(y_obs).flatten()
        self.window_buffer.append((preds, y_obs))
        
        # Recompute posterior from window
        self._compute_batch_posterior()
        
        latency = time.time() - start_time
        self.update_latencies.append(latency)
        
        uncertainty = np.trace(self.Sigma) if self.Sigma is not None else 0.0
        self.uncertainties.append(uncertainty)
        
        return latency

    def _compute_vi_elbo_negative(
        self,
        alpha: np.ndarray,
        Phi: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Compute negative ELBO for variational inference.
        """
        num_specialists = len(alpha)
        
        # Dirichlet statistics
        alpha_safe = np.maximum(alpha, 1e-10)
        S = np.sum(alpha_safe)
        mu = alpha_safe / S
        Sigma_dir = (np.diag(mu) - np.outer(mu, mu)) / (S + 1)
        
        # Likelihood term (sum over output dimensions)
        likelihood_term = 0.0
        num_samples, num_outputs, _ = Phi.shape
        
        for j in range(num_outputs):
            Phi_j = Phi[:, j, :]
            y_j = y[:, j]
            
            Phi_T_Phi_j = Phi_j.T @ Phi_j
            trace_term = np.trace(Phi_T_Phi_j @ (Sigma_dir + np.outer(mu, mu)))
            quad_term = 2 * y_j @ (Phi_j @ mu)
            
            likelihood_term += self.noise_precision * (trace_term - quad_term)
        
        likelihood_term *= 0.5
        
        # KL divergence
        alpha_0 = np.ones(num_specialists) * self.dirichlet_alpha
        S_0 = np.sum(alpha_0)
        S_safe = np.sum(alpha_safe)
        
        kl_term = (
            gammaln(S_safe) - np.sum(gammaln(alpha_safe)) -
            (gammaln(S_0) - np.sum(gammaln(alpha_0))) +
            np.sum((alpha_safe - alpha_0) * (digamma(alpha_safe) - digamma(S_safe)))
        )
        
        return likelihood_term + kl_term

    def _compute_batch_posterior(self):
        """Compute posterior from window buffer using VI."""
        if len(self.window_buffer) == 0:
            S = np.sum(self.alpha)
            self.mu = self.alpha / S
            self.Sigma = (np.diag(self.mu) - np.outer(self.mu, self.mu)) / (S + 1)
            return
        
        num_samples = len(self.window_buffer)
        num_outputs = 3
        num_specialists = self.library.num_specialists
        
        Phi = np.zeros((num_samples, num_outputs, num_specialists))
        y = np.zeros((num_samples, num_outputs))
        
        for i, (preds, obs) in enumerate(self.window_buffer):
            Phi[i, :, :] = preds
            y[i, :] = obs
        
        def objective(alpha):
            return self._compute_vi_elbo_negative(alpha, Phi, y)
        
        bounds = [(1e-6, None) for _ in range(num_specialists)]
        x0 = self.alpha if self.alpha is not None else np.ones(num_specialists)
        
        result = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            self.alpha = result.x
        else:
            self.optimization_failures += 1
            self.alpha = x0
        
        S = np.sum(self.alpha)
        self.mu = self.alpha / S
        self.Sigma = (np.diag(self.mu) - np.outer(self.mu, self.mu)) / (S + 1)

    def get_weights(self) -> np.ndarray:
        """Get current posterior mean weights."""
        return self.mu.copy() if self.mu is not None else None

    def get_uncertainty(self) -> float:
        """Get current uncertainty (trace of covariance)."""
        return np.trace(self.Sigma) if self.Sigma is not None else 0.0

    def get_dominant_specialist(self) -> int:
        """Get index of specialist with highest weight."""
        if self.mu is None:
            return -1
        return int(np.argmax(self.mu))

    def get_estimated_friction(self) -> float:
        """Get blended friction coefficient (convenience method)."""
        blended = self.get_blended_dynamic_params()
        return (blended.get('pacejka_D_f', 0.8) + blended.get('pacejka_D_r', 0.8)) / 2

    def get_mean_latency(self) -> float:
        """Get mean update latency in seconds."""
        return np.mean(self.update_latencies) if self.update_latencies else 0.0

    def get_convergence_metrics(self) -> dict:
        """Get performance and convergence metrics."""
        blended = self.get_blended_dynamic_params() if self.mu is not None else {}
        return {
            'mean_uncertainty': np.mean(self.uncertainties) if self.uncertainties else 0.0,
            'mean_latency_ms': 1000 * self.get_mean_latency(),
            'window_utilization': len(self.window_buffer) / self.window_size,
            'final_uncertainty': self.get_uncertainty(),
            'num_updates': len(self.update_latencies),
            'optimization_failures': self.optimization_failures,
            'dominant_specialist': self.get_dominant_specialist(),
            'estimated_friction': blended.get('pacejka_D_f', 0),
            'estimated_cd': blended.get('cd', 0),
        }

    def reset(self):
        """Reset simulator state while keeping library."""
        if self.library is not None:
            self.initialize_for_library(self.library)


if __name__ == "__main__":
    # Quick test comparing Linear vs Re-Projection
    print("Testing VirtualOnlineSimulator: Linear vs Re-Projection")
    print("="*60)
    
    # Load library
    from pathlib import Path
    model_dir = Path(__file__).parent / "models_hss"
    
    if model_dir.exists():
        library = HSSLibrary(str(model_dir))
        print(f"Loaded {library.num_specialists} specialists")
        
        simulator = VirtualOnlineSimulator(
            library=library,
            window_size=10,
            dirichlet_alpha=1.0,
        )
        
        # Test point
        state = np.array([20.0, 0.5, 0.1])
        control = np.array([0.1, 0.3])
        static = np.array([900.0, 1500.0])
        
        # Compare methods
        pred_linear = simulator.predict_linear(state, control, static)
        pred_physics = simulator.predict_physics_based(state, control, static)
        
        print(f"\nState: vx={state[0]:.1f}, vy={state[1]:.2f}, ω={state[2]:.2f}")
        print(f"Control: δ={control[0]:.2f}, T={control[1]:.2f}")
        print(f"Static: m={static[0]:.0f}kg, Iz={static[1]:.0f}kg·m²")
        
        print(f"\nLinear (LPV):    dvx={pred_linear[0]:.4f}, dvy={pred_linear[1]:.4f}, dω={pred_linear[2]:.4f}")
        print(f"Re-Projection:   dvx={pred_physics[0]:.4f}, dvy={pred_physics[1]:.4f}, dω={pred_physics[2]:.4f}")
        
        gap = np.linalg.norm(pred_linear - pred_physics)
        print(f"\nLPV Gap: ||Linear - Physics|| = {gap:.4f} m/s²")
        
        blended = simulator.get_blended_dynamic_params()
        print(f"\nBlended params: μ_f={blended['pacejka_D_f']:.3f}, Cd={blended['cd']:.3f}")
    else:
        print(f"Model directory not found: {model_dir}")
        print("Run greedy_specialist_selection_hss.py first.")
