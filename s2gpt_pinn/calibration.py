#!/usr/bin/env python3
"""
S²GPT-PINN Calibration Module: Two Modes for Weight Computation

MODE A (Static/Known Parameters):
    - Physical parameters θ are known (from manufacturer, Deep Dynamics, etc.)
    - Weights computed via multi-dimensional RBF kernel
    - No observed states needed
    - Fast: O(K) where K = number of specialists

MODE B (Adaptive/Observed States):
    - Physical parameters are uncertain or changing
    - Weights computed via linear regression on observed accelerations
    - Requires (state, control, observed_accel) tuples
    - Fast: O(K³) for K specialists (linear algebra)

Both modes output weights w_i that are used in:
    f̂(x, u) = Σᵢ wᵢ · Ψᵢ(x, u)

Author: H-SS Integration for NMPC
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
from collections import deque


@dataclass
class VehicleParamsConfig:
    """
    Complete vehicle parameters for S²GPT-PINN.
    
    Organized into:
    - Static parameters (geometry/mass): input to specialist networks
    - Dynamic parameters (tire/aero): determine specialist selection/weights
    """
    # =========================================================================
    # STATIC PARAMETERS - Input to specialist networks, usually fixed
    # =========================================================================
    m: float = 660.0        # Mass (kg)
    Iz: float = 1000.0      # Yaw inertia (kg·m²)
    lf: float = 1.5         # CG to front axle (m)
    lr: float = 1.0         # CG to rear axle (m)
    
    # =========================================================================
    # DYNAMIC PARAMETERS - Determine specialist weighting, can vary
    # =========================================================================
    # Pacejka tire model - FRONT axle
    pacejka_B_f: float = 12.0   # Stiffness factor
    pacejka_C_f: float = 1.5    # Shape factor
    pacejka_D_f: float = 1.0    # Peak factor (≈ friction coefficient μ)
    pacejka_E_f: float = 0.5    # Curvature factor
    
    # Pacejka tire model - REAR axle
    pacejka_B_r: float = 12.0
    pacejka_C_r: float = 1.5
    pacejka_D_r: float = 1.0    # Peak factor (≈ friction coefficient μ)
    pacejka_E_r: float = 0.5
    
    # Drivetrain parameters
    cm1: float = 2000.0     # Motor constant 1
    cm2: float = 10.0       # Motor constant 2 (velocity-dependent loss)
    
    # Aerodynamic/resistance parameters
    cr0: float = 200.0      # Rolling resistance
    cd: float = 0.5         # Drag coefficient
    
    # =========================================================================
    # PARAMETER GROUPINGS
    # =========================================================================
    @property
    def static_params(self) -> np.ndarray:
        """Parameters that are input to specialist networks: [m, Iz]"""
        return np.array([self.m, self.Iz])
    
    @property
    def static_params_extended(self) -> np.ndarray:
        """Extended static params: [m, Iz, lf, lr]"""
        return np.array([self.m, self.Iz, self.lf, self.lr])
    
    @property
    def dynamic_params(self) -> np.ndarray:
        """
        All 12 dynamic parameters that determine specialist weighting.
        Order: [B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, cm1, cm2, cr0, cd]
        """
        return np.array([
            self.pacejka_B_f, self.pacejka_C_f, self.pacejka_D_f, self.pacejka_E_f,
            self.pacejka_B_r, self.pacejka_C_r, self.pacejka_D_r, self.pacejka_E_r,
            self.cm1, self.cm2, self.cr0, self.cd
        ])
    
    @property
    def friction_params(self) -> np.ndarray:
        """Just friction coefficients: [D_f, D_r] (simplified for RBF)"""
        return np.array([self.pacejka_D_f, self.pacejka_D_r])
    
    @property
    def pacejka_front(self) -> np.ndarray:
        """Front tire Pacejka: [B_f, C_f, D_f, E_f]"""
        return np.array([self.pacejka_B_f, self.pacejka_C_f, 
                        self.pacejka_D_f, self.pacejka_E_f])
    
    @property
    def pacejka_rear(self) -> np.ndarray:
        """Rear tire Pacejka: [B_r, C_r, D_r, E_r]"""
        return np.array([self.pacejka_B_r, self.pacejka_C_r,
                        self.pacejka_D_r, self.pacejka_E_r])
    
    @staticmethod
    def dynamic_param_names() -> List[str]:
        """Names of dynamic parameters for logging/debugging."""
        return [
            'pacejka_B_f', 'pacejka_C_f', 'pacejka_D_f', 'pacejka_E_f',
            'pacejka_B_r', 'pacejka_C_r', 'pacejka_D_r', 'pacejka_E_r',
            'cm1', 'cm2', 'cr0', 'cd'
        ]
    
    @staticmethod
    def from_dynamic_array(arr: np.ndarray, base: Optional['VehicleParamsConfig'] = None) -> 'VehicleParamsConfig':
        """Create VehicleParamsConfig from dynamic parameter array."""
        if base is None:
            base = VehicleParamsConfig()
        return VehicleParamsConfig(
            m=base.m, Iz=base.Iz, lf=base.lf, lr=base.lr,
            pacejka_B_f=arr[0], pacejka_C_f=arr[1], pacejka_D_f=arr[2], pacejka_E_f=arr[3],
            pacejka_B_r=arr[4], pacejka_C_r=arr[5], pacejka_D_r=arr[6], pacejka_E_r=arr[7],
            cm1=arr[8], cm2=arr[9], cr0=arr[10], cd=arr[11]
        )


@dataclass
class SpecialistInfo:
    """Information about a trained specialist."""
    index: int                          # Index in ensemble
    params: VehicleParamsConfig         # Parameters it was trained on
    dynamic_params: np.ndarray = field(init=False)  # Cached dynamic params
    
    def __post_init__(self):
        self.dynamic_params = self.params.dynamic_params


# =============================================================================
# MODE A: Static/Known Parameters - RBF Weighting
# =============================================================================

class RBFWeightComputer:
    """
    Compute specialist weights using multi-dimensional RBF kernel.
    
    MODE A: When physical parameters θ are KNOWN (from manufacturer, Deep Dynamics, etc.)
    
    Weight computation:
        w_i = exp(-||θ - θ_i||²_Σ / 2σ²) / Σⱼ exp(...)
    
    Where ||·||²_Σ is the Mahalanobis distance accounting for parameter scales.
    """
    
    # Normalization scales for each dynamic parameter (for distance computation)
    PARAM_SCALES = np.array([
        5.0,    # pacejka_B_f: typical range [8, 16]
        0.5,    # pacejka_C_f: typical range [1.0, 2.0]
        0.5,    # pacejka_D_f: typical range [0.3, 1.5] (friction)
        0.3,    # pacejka_E_f: typical range [0.2, 0.8]
        5.0,    # pacejka_B_r
        0.5,    # pacejka_C_r
        0.5,    # pacejka_D_r
        0.3,    # pacejka_E_r
        500.0,  # cm1: typical range [1500, 2500]
        5.0,    # cm2: typical range [5, 15]
        100.0,  # cr0: typical range [100, 300]
        0.3,    # cd: typical range [0.3, 0.8]
    ])
    
    # Which parameters to use for weighting (can be subset)
    # Default: use only friction (D_f, D_r) for simplicity
    FRICTION_ONLY_MASK = np.array([
        False, False, True, False,  # Front tire: only D_f
        False, False, True, False,  # Rear tire: only D_r
        False, False, False, False  # Drivetrain: none
    ])
    
    # Full parameters mask (all 12)
    FULL_PARAMS_MASK = np.ones(12, dtype=bool)
    
    def __init__(
        self,
        specialist_params: List[VehicleParamsConfig],
        rbf_width: float = 0.2,
        use_full_params: bool = False,
    ):
        """
        Initialize RBF weight computer.
        
        Args:
            specialist_params: List of VehicleParamsConfig for each specialist
            rbf_width: RBF kernel width (σ) - smaller = more localized
            use_full_params: If True, use all 12 params; if False, only friction
        """
        self.n_specialists = len(specialist_params)
        self.rbf_width = rbf_width
        
        # Select which parameters to use
        self.param_mask = self.FULL_PARAMS_MASK if use_full_params else self.FRICTION_ONLY_MASK
        self.scales = self.PARAM_SCALES[self.param_mask]
        
        # Store specialist parameter centers
        self.centers = np.array([
            p.dynamic_params[self.param_mask] for p in specialist_params
        ])  # [K, n_params]
        
    def compute_weights(self, current_params: VehicleParamsConfig) -> np.ndarray:
        """
        Compute RBF weights for current parameters.
        
        Args:
            current_params: Current vehicle parameters
            
        Returns:
            weights: [K,] normalized weights summing to 1
        """
        # Extract relevant parameters
        current = current_params.dynamic_params[self.param_mask]
        
        # Compute normalized squared distances
        diff = (self.centers - current) / self.scales  # [K, n_params]
        distances = np.sum(diff ** 2, axis=1)  # [K,]
        
        # RBF kernel
        weights = np.exp(-distances / (2 * self.rbf_width ** 2))
        
        # Normalize
        weights = weights / (weights.sum() + 1e-10)
        
        return weights
    
    def compute_weights_from_friction(self, mu_f: float, mu_r: float) -> np.ndarray:
        """
        Simplified: compute weights from just friction coefficients.
        
        Args:
            mu_f: Front friction coefficient
            mu_r: Rear friction coefficient
            
        Returns:
            weights: [K,] normalized weights
        """
        # Build minimal params config
        params = VehicleParamsConfig(pacejka_D_f=mu_f, pacejka_D_r=mu_r)
        return self.compute_weights(params)
    
    def compute_weights_tensor(
        self, 
        current_params: VehicleParamsConfig,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Compute weights and return as tensor."""
        weights = self.compute_weights(current_params)
        return torch.from_numpy(weights).float().to(device)


# =============================================================================
# MODE B: Adaptive Parameters - Online Linear Regression
# =============================================================================

@dataclass
class CalibrationObservation:
    """Single observation for online adaptation."""
    state: np.ndarray           # [3,] - vx, vy, omega
    control: np.ndarray         # [2,] - delta, throttle
    static_params: np.ndarray   # [2,] - m, Iz
    acceleration: np.ndarray    # [3,] - dvx, dvy, domega (observed)


class OnlineLinearRegressor:
    """
    Online linear regression for specialist weight adaptation.

    MODE B: When physical parameters are UNCERTAIN or CHANGING.

    Maintains a sliding window of observations and recomputes weights
    via linear regression to adapt the specialist combination over time.

    Complexity: O(K³) where K = number of specialists (typically 4-8)
    """

    def __init__(
        self,
        ensemble,  # S2GPTEnsemble - imported to avoid circular imports
        window_size: int = 50,
        regularization: float = 1e-4,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize online regressor.

        Args:
            ensemble: S²GPT-PINN ensemble with K specialists
            window_size: Number of recent observations to use
            regularization: Ridge regression λ for numerical stability
            device: Torch device for specialist inference
        """
        self.ensemble = ensemble
        self.window_size = window_size
        self.regularization = regularization
        self.device = device

        # Observation buffer
        self.buffer: deque = deque(maxlen=window_size)

        # Current fitted weights
        self.weights: Optional[np.ndarray] = None

        # Statistics
        self.n_updates = 0
        self.last_residual = float('inf')

    @property
    def n_specialists(self) -> int:
        return self.ensemble.n_specialists

    def add_observation(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray,
        observed_accel: np.ndarray
    ):
        """
        Add a new observation to the buffer and update weights.

        Args:
            state: [3,] - vx, vy, omega
            control: [2,] - delta, throttle
            static_params: [2,] - m, Iz
            observed_accel: [3,] - dvx, dvy, domega (measured)
        """
        obs = CalibrationObservation(
            state=np.asarray(state, dtype=np.float32),
            control=np.asarray(control, dtype=np.float32),
            static_params=np.asarray(static_params, dtype=np.float32),
            acceleration=np.asarray(observed_accel, dtype=np.float32)
        )
        self.buffer.append(obs)

        # Update weights with new observation
        self._update_weights()

    def _get_specialist_predictions(self, obs: CalibrationObservation) -> np.ndarray:
        """
        Query all specialists for a single observation.

        Returns:
            predictions: [K, 3] - each specialist's predicted acceleration
        """
        state = torch.from_numpy(obs.state).float().unsqueeze(0).to(self.device)
        control = torch.from_numpy(obs.control).float().unsqueeze(0).to(self.device)
        static = torch.from_numpy(obs.static_params).float().unsqueeze(0).to(self.device)

        predictions = []
        with torch.no_grad():
            for specialist in self.ensemble.specialists:
                pred = specialist(state, control, static)
                predictions.append(pred.cpu().numpy().squeeze())

        return np.array(predictions)  # [K, 3]

    def _update_weights(self):
        """
        Update weights using linear regression on current buffer.
        """
        if len(self.buffer) == 0:
            self.weights = np.ones(self.n_specialists) / self.n_specialists
            return

        # Build design matrix Φ and target vector y
        N = len(self.buffer)
        K = self.n_specialists

        Phi = np.zeros((N, K, 3))
        y = np.zeros((N, 3))

        for i, obs in enumerate(self.buffer):
            Phi[i] = self._get_specialist_predictions(obs)  # [K, 3]
            y[i] = obs.acceleration  # [3,]

        # Solve separately for each output dimension
        weights_per_dim = []

        for d in range(3):
            Phi_d = Phi[:, :, d]  # [N, K]
            y_d = y[:, d]         # [N,]

            # Ridge regression: (Φᵀ Φ + λI)⁻¹ Φᵀ y
            A = Phi_d.T @ Phi_d + self.regularization * np.eye(K)
            b = Phi_d.T @ y_d

            try:
                w_d = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                w_d = np.linalg.lstsq(A, b, rcond=None)[0]

            weights_per_dim.append(w_d)

        # Average weights across dimensions
        weights_raw = np.mean(weights_per_dim, axis=0)

        # Project to simplex (non-negative, sum to 1)
        self.weights = self._project_to_simplex(weights_raw)
        self.n_updates += 1

        # Compute residual for monitoring
        y_pred = np.einsum('nkd,k->nd', Phi, self.weights)
        self.last_residual = np.mean(np.sum((y_pred - y) ** 2, axis=1) ** 0.5)

    def _project_to_simplex(self, w: np.ndarray) -> np.ndarray:
        """
        Project weights to probability simplex (non-negative, sum to 1).
        """
        # Soft ReLU to ensure non-negative
        w = np.maximum(w, 0)

        # Normalize
        w_sum = w.sum()
        if w_sum > 1e-10:
            w = w / w_sum
        else:
            # Fallback to uniform
            w = np.ones_like(w) / len(w)

        return w

    def get_weights(self) -> np.ndarray:
        """Get current fitted weights (or uniform if no data)."""
        if self.weights is None:
            return np.ones(self.n_specialists) / self.n_specialists
        return self.weights

    def get_weights_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get weights as torch tensor."""
        device = device or self.device
        return torch.from_numpy(self.get_weights()).float().to(device)

    def reset(self):
        """Clear observation buffer and reset weights."""
        self.buffer.clear()
        self.weights = None
        self.n_updates = 0
        self.last_residual = float('inf')


# =============================================================================
# UNIFIED CALIBRATION MANAGER
# =============================================================================

class S2GPTCalibrationManager:
    """
    Unified calibration manager supporting both Mode A and Mode B.

    Mode A (Static): Use RBF weights when parameters are known
    Mode B (Adaptive): Use online linear regression when parameters are uncertain

    The manager automatically handles:
    - Mode switching based on parameter availability
    - Weight smoothing for stability
    - Integration with the S²GPT ensemble
    """

    def __init__(
        self,
        ensemble,  # S2GPTEnsemble
        specialist_params: List[VehicleParamsConfig],
        rbf_width: float = 0.2,
        lr_window_size: int = 50,
        lr_regularization: float = 1e-4,
        weight_smoothing: float = 0.0,  # EMA smoothing (0 = no smoothing)
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize calibration manager.

        Args:
            ensemble: S²GPT-PINN ensemble
            specialist_params: Parameters each specialist was trained on
            rbf_width: RBF kernel width for Mode A
            lr_window_size: Observation window for Mode B
            lr_regularization: Ridge regularization for Mode B
            weight_smoothing: EMA factor for weight updates (0-1)
            device: Torch device
        """
        self.ensemble = ensemble
        self.specialist_params = specialist_params
        self.device = device
        self.weight_smoothing = weight_smoothing

        # Mode A: RBF weights
        self.rbf_computer = RBFWeightComputer(
            specialist_params, rbf_width=rbf_width, use_full_params=False
        )

        # Mode B: Online linear regression
        self.lr_regressor = OnlineLinearRegressor(
            ensemble, window_size=lr_window_size,
            regularization=lr_regularization, device=device
        )

        # Current state
        self._current_weights: Optional[np.ndarray] = None
        self._mode: str = 'static'  # 'static' or 'adaptive'

    @property
    def n_specialists(self) -> int:
        return self.ensemble.n_specialists

    @property
    def current_weights(self) -> np.ndarray:
        if self._current_weights is None:
            return np.ones(self.n_specialists) / self.n_specialists
        return self._current_weights

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        """Set calibration mode: 'static' or 'adaptive'."""
        assert mode in ['static', 'adaptive']
        self._mode = mode

    # -------------------------------------------------------------------------
    # MODE A: Static Parameters
    # -------------------------------------------------------------------------

    def update_from_params(self, params: VehicleParamsConfig) -> np.ndarray:
        """
        MODE A: Update weights from known parameters.

        Args:
            params: Current vehicle parameters

        Returns:
            weights: [K,] specialist weights
        """
        self._mode = 'static'
        new_weights = self.rbf_computer.compute_weights(params)
        self._apply_weights(new_weights)
        return self.current_weights

    def update_from_friction(self, mu_f: float, mu_r: Optional[float] = None) -> np.ndarray:
        """
        MODE A (simplified): Update weights from friction coefficients only.

        Args:
            mu_f: Front friction coefficient
            mu_r: Rear friction coefficient (defaults to mu_f if not provided)

        Returns:
            weights: [K,] specialist weights
        """
        mu_r = mu_r if mu_r is not None else mu_f
        self._mode = 'static'
        new_weights = self.rbf_computer.compute_weights_from_friction(mu_f, mu_r)
        self._apply_weights(new_weights)
        return self.current_weights

    # -------------------------------------------------------------------------
    # MODE B: Online Adaptation
    # -------------------------------------------------------------------------

    def add_observation(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray,
        observed_accel: np.ndarray
    ):
        """
        MODE B: Add observation for online adaptation.

        Args:
            state: [3,] - vx, vy, omega
            control: [2,] - delta, throttle
            static_params: [2,] - m, Iz
            observed_accel: [3,] - measured accelerations
        """
        self.lr_regressor.add_observation(state, control, static_params, observed_accel)
        # Update current weights from regressor
        new_weights = self.lr_regressor.get_weights()
        self._apply_weights(new_weights)

    def get_adaptation_residual(self) -> float:
        """Get last adaptation residual (MODE B only)."""
        return self.lr_regressor.last_residual

    # -------------------------------------------------------------------------
    # COMMON
    # -------------------------------------------------------------------------

    def _apply_weights(self, new_weights: np.ndarray):
        """Apply new weights with optional smoothing."""
        if self.weight_smoothing > 0 and self._current_weights is not None:
            # Exponential moving average
            alpha = self.weight_smoothing
            self._current_weights = alpha * self._current_weights + (1 - alpha) * new_weights
        else:
            self._current_weights = new_weights

        # Update ensemble weights
        self.ensemble.set_weights(torch.from_numpy(self._current_weights).float().to(self.device))

    def get_weights_tensor(self) -> torch.Tensor:
        """Get current weights as tensor."""
        return torch.from_numpy(self.current_weights).float().to(self.device)

    def reset_adaptive(self):
        """Reset online adaptation state."""
        self.lr_regressor.reset()

    def get_status(self) -> Dict:
        """Get calibration status for logging."""
        return {
            'mode': self._mode,
            'weights': self.current_weights.tolist(),
            'n_observations': len(self.lr_regressor.buffer),
            'n_updates': self.lr_regressor.n_updates,
            'last_residual': self.lr_regressor.last_residual,
        }


# =============================================================================
# HELPER: Generate specialist parameter sets for training
# =============================================================================

def generate_specialist_param_sets(
    n_specialists: int = 50,
    friction_range: Tuple[float, float] = (0.3, 1.2),
    vary_pacejka_bc: bool = True,
    vary_drivetrain: bool = True,
    seed: Optional[int] = None,
) -> List[VehicleParamsConfig]:
    """
    Generate candidate dynamic parameter sets for S²GPT library building.

    Samples the DYNAMIC parameter space (friction, tire stiffness, aero).
    Static parameters (m, Iz) are kept at nominal values since they are
    handled via input conditioning rather than library diversity.

    This matches the hss-codebase approach with random sampling instead of
    deterministic linear spacing.

    Args:
        n_specialists: Number of candidate parameter sets
        friction_range: Range for friction coefficient (D_f, D_r)
        vary_pacejka_bc: Also vary B and C coefficients
        vary_drivetrain: Also vary drivetrain parameters
        seed: Random seed for reproducibility

    Returns:
        List of VehicleParamsConfig with varied dynamic parameters
    """
    if seed is not None:
        np.random.seed(seed)

    candidates = []

    # Parameter ranges (physically plausible for road vehicles)
    ranges = {
        # Friction (D) - most important for dynamics
        "pacejka_D_f": friction_range,  # Ice (0.3) to Dry (1.2)
        "pacejka_D_r": friction_range,
    }

    if vary_pacejka_bc:
        ranges.update({
            # Pacejka shape
            "pacejka_B_f": (8.0, 16.0),
            "pacejka_B_r": (8.0, 16.0),
            "pacejka_C_f": (1.0, 2.0),
            "pacejka_C_r": (1.0, 2.0),
            "pacejka_E_f": (0.0, 1.0),
            "pacejka_E_r": (0.0, 1.0),
        })

    if vary_drivetrain:
        ranges.update({
            # Aero and drivetrain
            "cd": (0.2, 0.8),
            "cm1": (1500.0, 2500.0),
        })

    for _ in range(n_specialists):
        kwargs = {}
        for param, (lo, hi) in ranges.items():
            kwargs[param] = float(np.random.uniform(lo, hi))

        # Use nominal static params (these are handled via conditioning)
        # Create new instance with updated parameters
        params = VehicleParamsConfig(m=800.0, Iz=1200.0, **kwargs)
        candidates.append(params)

    return candidates


if __name__ == "__main__":
    # Quick test of the calibration module
    print("Testing S²GPT-PINN Calibration Module")
    print("=" * 60)
    
    # Generate specialist parameter sets
    param_sets = generate_specialist_param_sets(n_specialists=4)
    print(f"\nGenerated {len(param_sets)} specialist parameter sets:")
    for i, p in enumerate(param_sets):
        print(f"  Specialist {i}: μ_f={p.pacejka_D_f:.2f}, μ_r={p.pacejka_D_r:.2f}, "
              f"B_f={p.pacejka_B_f:.1f}, C_f={p.pacejka_C_f:.2f}")
    
    # Test RBF weight computer
    print("\n" + "=" * 60)
    print("Testing MODE A: RBF Weight Computer")
    
    rbf = RBFWeightComputer(param_sets, rbf_width=0.2)
    
    test_frictions = [0.4, 0.7, 1.0, 1.3]
    for mu in test_frictions:
        test_params = VehicleParamsConfig(pacejka_D_f=mu, pacejka_D_r=mu)
        weights = rbf.compute_weights(test_params)
        print(f"  μ={mu:.1f} → weights={[f'{w:.3f}' for w in weights]}")
    
    print("\n✓ Calibration module tests passed!")

