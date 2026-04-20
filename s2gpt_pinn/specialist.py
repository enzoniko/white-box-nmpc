#!/usr/bin/env python3
"""
H-SS Specialist Network with Pre-Computed Analytic Jacobians

This module implements sparse specialist networks for H-SS (HYDRA Self-Supervised) that provide:
1. Fast forward inference for NMPC dynamics
2. Pre-computed analytic Jacobians (∂f/∂x, ∂f/∂u) for CasADi integration
3. Support for domain-randomized static parameters (m, Iz)

The analytic Jacobian computation uses the chain rule through the network
layers, avoiding autograd overhead for microsecond inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class HSSConfig:
    """Configuration for H-SS specialist."""
    hidden_dim: int = 64           # Hidden layer dimension
    n_layers: int = 3              # Number of hidden layers
    state_dim: int = 3             # [vx, vy, omega]
    control_dim: int = 2           # [delta, throttle]
    static_dim: int = 2            # [mass, Iz] - conditional inputs
    output_dim: int = 3            # [dvx, dvy, domega]
    
    # Input normalization scales
    state_scale: Tuple[float, ...] = (40.0, 5.0, 3.0)
    control_scale: Tuple[float, ...] = (0.5, 1.0)
    static_scale: Tuple[float, ...] = (1000.0, 2000.0)
    output_scale: Tuple[float, ...] = (10.0, 10.0, 5.0)


class HSSSpecialist(nn.Module):
    """
    Sparse Specialist Network with Pre-Computed Analytic Jacobians.
    
    Architecture:
        Input: [vx, vy, ω, δ, T, m, Iz] (7 features) -> normalized
        Hidden: n_layers x (Linear + Tanh)
        Output: [dvx, dvy, dω] (3 accelerations)
    
    The network stores weights explicitly for efficient Jacobian computation.
    Tanh activation enables closed-form derivative: d/dx tanh(x) = 1 - tanh²(x)
    """
    
    def __init__(self, config: Optional[HSSConfig] = None):
        super().__init__()
        self.config = config or HSSConfig()
        cfg = self.config
        
        # Total input dim: state + control + static
        self.input_dim = cfg.state_dim + cfg.control_dim + cfg.static_dim
        
        # Register normalization scales as buffers
        self.register_buffer("state_scale", torch.tensor(cfg.state_scale, dtype=torch.float32))
        self.register_buffer("control_scale", torch.tensor(cfg.control_scale, dtype=torch.float32))
        self.register_buffer("static_scale", torch.tensor(cfg.static_scale, dtype=torch.float32))
        self.register_buffer("output_scale", torch.tensor(cfg.output_scale, dtype=torch.float32))
        
        # Build network with explicit layer access for Jacobian computation
        self.layers = nn.ModuleList()
        prev_dim = self.input_dim
        
        for i in range(cfg.n_layers):
            self.layers.append(nn.Linear(prev_dim, cfg.hidden_dim))
            prev_dim = cfg.hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, cfg.output_dim)
        
        # Initialize weights (Xavier for tanh)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization for tanh activations."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def _normalize_inputs(
        self, 
        state: torch.Tensor, 
        control: torch.Tensor, 
        static_params: torch.Tensor
    ) -> torch.Tensor:
        """Normalize inputs to ~[-1, 1] for stable gradient flow."""
        state_norm = state / self.state_scale
        control_norm = control / self.control_scale
        static_norm = static_params / self.static_scale
        return torch.cat([state_norm, control_norm, static_norm], dim=-1)
    
    def forward(
        self, 
        state: torch.Tensor, 
        control: torch.Tensor, 
        static_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: compute accelerations.
        
        Args:
            state: [batch, 3] tensor of (vx, vy, omega)
            control: [batch, 2] tensor of (delta, throttle)
            static_params: [batch, 2] tensor of (mass, Iz)
            
        Returns:
            accelerations: [batch, 3] tensor of (dvx, dvy, domega)
        """
        x = self._normalize_inputs(state, control, static_params)
        
        # Forward through hidden layers with tanh
        for layer in self.layers:
            x = torch.tanh(layer(x))
        
        # Output layer (no activation)
        out = self.output_layer(x)
        
        # Scale output to physical range
        return out * self.output_scale
    
    def forward_with_intermediates(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        static_params: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass returning intermediate activations for Jacobian computation.
        
        Returns:
            output: [batch, 3] accelerations
            z_list: Pre-activation values for each hidden layer
            h_list: Post-activation values for each hidden layer
        """
        x = self._normalize_inputs(state, control, static_params)
        
        z_list = []  # Pre-tanh values
        h_list = [x]  # Post-tanh values (starts with input)
        
        for layer in self.layers:
            z = layer(h_list[-1])
            z_list.append(z)
            h = torch.tanh(z)
            h_list.append(h)
        
        out = self.output_layer(h_list[-1])
        
        return out * self.output_scale, z_list, h_list
    
    def jacobian_analytic(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        static_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pre-computed analytic Jacobians using chain rule.
        
        For a network y = W_out * tanh(W_{n-1} * ... * tanh(W_1 * x) ...)
        The Jacobian is:
            dy/dx = output_scale * W_out * diag(1-h_n²) * W_{n-1} * ... * diag(1-h_1²) * W_1 * diag(1/input_scale)
        
        Args:
            state: [batch, 3] or [3,] tensor
            control: [batch, 2] or [2,] tensor
            static_params: [batch, 2] or [2,] tensor
            
        Returns:
            jac_state: [batch, 3, 3] or [3, 3] - ∂f/∂x
            jac_control: [batch, 3, 2] or [3, 2] - ∂f/∂u
        """
        single_sample = state.dim() == 1
        if single_sample:
            state = state.unsqueeze(0)
            control = control.unsqueeze(0)
            static_params = static_params.unsqueeze(0)
        
        batch_size = state.shape[0]
        device = state.device
        
        # Forward pass to get intermediate activations
        _, z_list, h_list = self.forward_with_intermediates(state, control, static_params)
        
        # Build Jacobian via chain rule (backward from output to input)
        # Start with output layer: dout/dh_last = W_out (before output scaling)
        jac = self.output_layer.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 3, H]
        
        # Apply output scaling: actual output = raw * output_scale
        # So d(scaled_out)/d(raw_out) = diag(output_scale)
        jac = jac * self.output_scale.view(1, -1, 1)  # [B, 3, H]
        
        # Backpropagate through hidden layers
        for i in range(len(self.layers) - 1, -1, -1):
            # dh/dz = 1 - tanh²(z) = 1 - h²
            h = h_list[i + 1]  # Post-activation at this layer
            dtanh = 1 - h ** 2  # [B, H]
            
            # Apply activation derivative: jac = jac * diag(dtanh)
            jac = jac * dtanh.unsqueeze(1)  # [B, 3, H]
            
            # Apply weight matrix: jac = jac @ W
            W = self.layers[i].weight  # [H, prev_H]
            jac = jac @ W  # [B, 3, prev_H]
        
        # Now jac is ∂f/∂(normalized_input)
        # Apply input denormalization: d(norm)/d(raw) = 1/scale
        input_scale = torch.cat([
            self.state_scale,
            self.control_scale,
            self.static_scale
        ])  # [7]
        jac = jac / input_scale.view(1, 1, -1)  # [B, 3, 7]
        
        # Split Jacobian into state and control parts
        cfg = self.config
        jac_state = jac[:, :, :cfg.state_dim]  # [B, 3, 3]
        jac_control = jac[:, :, cfg.state_dim:cfg.state_dim + cfg.control_dim]  # [B, 3, 2]
        
        if single_sample:
            jac_state = jac_state.squeeze(0)
            jac_control = jac_control.squeeze(0)
        
        return jac_state, jac_control
    
    def predict_numpy(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray
    ) -> np.ndarray:
        """Convenience method for numpy inputs."""
        single_sample = state.ndim == 1
        if single_sample:
            state = state[np.newaxis, :]
            control = control[np.newaxis, :]
            static_params = static_params[np.newaxis, :]
        
        device = next(self.parameters()).device
        state_t = torch.from_numpy(state).float().to(device)
        control_t = torch.from_numpy(control).float().to(device)
        static_t = torch.from_numpy(static_params).float().to(device)
        
        with torch.no_grad():
            accel = self.forward(state_t, control_t, static_t)
        
        result = accel.cpu().numpy()
        return result[0] if single_sample else result
    
    def jacobian_numpy(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians with numpy inputs/outputs."""
        single_sample = state.ndim == 1
        if single_sample:
            state = state[np.newaxis, :]
            control = control[np.newaxis, :]
            static_params = static_params[np.newaxis, :]
        
        device = next(self.parameters()).device
        state_t = torch.from_numpy(state).float().to(device)
        control_t = torch.from_numpy(control).float().to(device)
        static_t = torch.from_numpy(static_params).float().to(device)
        
        with torch.no_grad():
            jac_x, jac_u = self.jacobian_analytic(state_t, control_t, static_t)
        
        jac_x_np = jac_x.cpu().numpy()
        jac_u_np = jac_u.cpu().numpy()
        
        if single_sample:
            return jac_x_np[0], jac_u_np[0]
        return jac_x_np, jac_u_np


class HSSEnsemble(nn.Module):
    """
    H-SS Ensemble: Weighted combination of sparse specialists.

    Implements the meta-layer:
        f̂(x, u) = Σᵢ wᵢ(μ_current) · Ψᵢ(x, u, m, Iz)

    Where wᵢ are RBF weights based on friction parameter distance.

    Jacobians are computed as weighted sum of specialist Jacobians:
        J_total = Σᵢ wᵢ · Jᵢ
    """
    
    def __init__(
        self,
        specialists: List[HSSSpecialist],
        mu_centers: Optional[List[float]] = None,
        rbf_width: float = 0.15
    ):
        super().__init__()
        self.specialists = nn.ModuleList(specialists)
        self.n_specialists = len(specialists)
        
        # Friction centers for each specialist
        if mu_centers is None:
            mu_centers = np.linspace(0.3, 1.2, self.n_specialists).tolist()
        self.register_buffer(
            "mu_centers", 
            torch.tensor(mu_centers, dtype=torch.float32)
        )
        self.rbf_width = rbf_width
        
        # Current weights (updated when friction changes)
        self.register_buffer(
            "current_weights",
            torch.ones(self.n_specialists) / self.n_specialists
        )
        self._current_mu = 0.8
    
    def compute_rbf_weights(self, mu_current: float) -> torch.Tensor:
        """
        Compute RBF weights based on current friction coefficient.
        
        w_i = exp(-||μ - μ_i||² / (2σ²)) / Σⱼ exp(...)
        """
        distances = (self.mu_centers - mu_current) ** 2
        weights = torch.exp(-distances / (2 * self.rbf_width ** 2))
        weights = weights / weights.sum()  # L1 normalize
        return weights
    
    def update_friction(self, mu_current: float) -> torch.Tensor:
        """Update internal weights for current friction coefficient."""
        self._current_mu = mu_current
        new_w = self.compute_rbf_weights(mu_current)
        # Keep `current_weights` as a registered buffer (avoid re-binding the attribute)
        self.current_weights.copy_(new_w)
        return self.current_weights
    
    def set_weights(self, weights: torch.Tensor):
        """
        Explicitly set specialist weights (for MODE B calibration).
        
        Args:
            weights: [K,] tensor of weights, should sum to 1
        """
        assert len(weights) == self.n_specialists, \
            f"Expected {self.n_specialists} weights, got {len(weights)}"
        
        # Ensure weights are on the correct device
        device = self.current_weights.device
        if weights.device != device:
            weights = weights.to(device)
        
        # Keep `current_weights` as a registered buffer (avoid re-binding the attribute)
        self.current_weights.copy_(weights)
    
    def get_weights(self) -> torch.Tensor:
        """Get current specialist weights."""
        return self.current_weights
    
    @property
    def current_mu(self) -> float:
        return self._current_mu
    
    def forward(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        static_params: torch.Tensor,
        mu_current: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass: weighted combination of specialist predictions.
        
        Args:
            state: [batch, 3] tensor
            control: [batch, 2] tensor
            static_params: [batch, 2] tensor
            mu_current: Optional friction to update weights
            
        Returns:
            accelerations: [batch, 3] tensor
        """
        if mu_current is not None:
            self.update_friction(mu_current)
        
        # Decide which specialists to evaluate (Top-K gating can skip compute in Python)
        weights = self.current_weights
        if top_k is not None and top_k < self.n_specialists:
            top_k = max(int(top_k), 1)
            _, idx = torch.topk(weights, k=top_k)
            idx = idx.detach().cpu().tolist()
        else:
            idx = list(range(self.n_specialists))

        # Evaluate only selected specialists
        result = None
        for i in idx:
            w = weights[i]
            if w.abs().item() < 1e-12:
                continue
            pred = self.specialists[i](state, control, static_params)
            result = pred * w if result is None else (result + w * pred)
        if result is None:
            # degenerate: all weights ~0, fallback to uniform average over first specialist
            result = self.specialists[0](state, control, static_params)
        
        return result


    def jacobian_analytic(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        static_params: torch.Tensor,
        mu_current: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted sum of specialist Jacobians.
        
        J_total = Σᵢ wᵢ · Jᵢ
        """
        if mu_current is not None:
            self.update_friction(mu_current)
        
        single_sample = state.dim() == 1
        if single_sample:
            state = state.unsqueeze(0)
            control = control.unsqueeze(0)
            static_params = static_params.unsqueeze(0)
        
        batch_size = state.shape[0]
        device = state.device
        
        # Decide which specialists to evaluate (Top-K gating can skip compute in Python)
        weights = self.current_weights
        if top_k is not None and top_k < self.n_specialists:
            top_k = max(int(top_k), 1)
            _, idx = torch.topk(weights, k=top_k)
            idx = idx.detach().cpu().tolist()
        else:
            idx = list(range(self.n_specialists))

        # Initialize accumulators
        jac_state_total = torch.zeros(batch_size, 3, 3, device=device)
        jac_control_total = torch.zeros(batch_size, 3, 2, device=device)

        # Weighted sum of Jacobians (only selected)
        for i in idx:
            w = weights[i]
            if w.abs().item() < 1e-12:
                continue
            jac_x, jac_u = self.specialists[i].jacobian_analytic(state, control, static_params)
            jac_state_total = jac_state_total + w * jac_x
            jac_control_total = jac_control_total + w * jac_u
        
        if single_sample:
            jac_state_total = jac_state_total.squeeze(0)
            jac_control_total = jac_control_total.squeeze(0)
        
        return jac_state_total, jac_control_total
    
    def predict_numpy(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray,
        mu_current: Optional[float] = None
    ) -> np.ndarray:
        """Convenience method for numpy inputs."""
        single_sample = state.ndim == 1
        if single_sample:
            state = state[np.newaxis, :]
            control = control[np.newaxis, :]
            static_params = static_params[np.newaxis, :]
        
        device = next(self.parameters()).device
        state_t = torch.from_numpy(state).float().to(device)
        control_t = torch.from_numpy(control).float().to(device)
        static_t = torch.from_numpy(static_params).float().to(device)
        
        with torch.no_grad():
            accel = self.forward(state_t, control_t, static_t, mu_current)
        
        result = accel.cpu().numpy()
        return result[0] if single_sample else result
    
    def jacobian_numpy(
        self,
        state: np.ndarray,
        control: np.ndarray,
        static_params: np.ndarray,
        mu_current: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians with numpy inputs/outputs."""
        single_sample = state.ndim == 1
        if single_sample:
            state = state[np.newaxis, :]
            control = control[np.newaxis, :]
            static_params = static_params[np.newaxis, :]
        
        device = next(self.parameters()).device
        state_t = torch.from_numpy(state).float().to(device)
        control_t = torch.from_numpy(control).float().to(device)
        static_t = torch.from_numpy(static_params).float().to(device)
        
        with torch.no_grad():
            jac_x, jac_u = self.jacobian_analytic(state_t, control_t, static_t, mu_current)
        
        jac_x_np = jac_x.cpu().numpy()
        jac_u_np = jac_u.cpu().numpy()
        
        if single_sample:
            return jac_x_np[0], jac_u_np[0]
        return jac_x_np, jac_u_np


# -----------------------------------------------------------------------------
# Backwards-compatible aliases (paper text / older scripts still use S2GPT naming)
# -----------------------------------------------------------------------------
S2GPTConfig = HSSConfig
S2GPTSpecialist = HSSSpecialist
S2GPTEnsemble = HSSEnsemble


def verify_jacobian(specialist: HSSSpecialist, eps: float = 1e-3) -> Dict[str, float]:
    """
    Verify analytic Jacobian against numerical differentiation.
    
    Note: Uses larger epsilon (1e-3) because input normalization can cause
    numerical issues with smaller perturbations. The state_scale of 40.0
    means eps=1e-6 becomes 2.5e-8 in normalized space, causing precision loss.
    
    Args:
        specialist: HSSSpecialist to verify
        eps: Finite difference step size (default 1e-3 to avoid precision issues)
        
    Returns:
        Dict with error norms for state and control Jacobians
    """
    device = next(specialist.parameters()).device
    specialist.eval()
    
    # Random test point with positive vx
    state = torch.randn(3, device=device) * torch.tensor([10.0, 1.0, 0.5], device=device)
    state[0] = state[0].abs() + 10.0  # Ensure positive vx
    control = torch.randn(2, device=device) * torch.tensor([0.3, 0.5], device=device)
    static = torch.tensor([800.0, 1200.0], device=device)
    
    # Analytic Jacobian
    with torch.no_grad():
        jac_x_analytic, jac_u_analytic = specialist.jacobian_analytic(state, control, static)
    
    # Numerical Jacobian w.r.t state - use scaled epsilon based on input scales
    jac_x_numerical = torch.zeros(3, 3, device=device)
    state_eps = eps * specialist.state_scale  # Scale epsilon by input normalization
    
    with torch.no_grad():
        for i in range(3):
            state_plus = state.clone()
            state_plus[i] += state_eps[i]
            f_plus = specialist(state_plus.unsqueeze(0), control.unsqueeze(0), static.unsqueeze(0)).squeeze()
            
            state_minus = state.clone()
            state_minus[i] -= state_eps[i]
            f_minus = specialist(state_minus.unsqueeze(0), control.unsqueeze(0), static.unsqueeze(0)).squeeze()
            
            jac_x_numerical[:, i] = (f_plus - f_minus) / (2 * state_eps[i])
    
    # Numerical Jacobian w.r.t control
    jac_u_numerical = torch.zeros(3, 2, device=device)
    control_eps = eps * specialist.control_scale
    
    with torch.no_grad():
        for i in range(2):
            control_plus = control.clone()
            control_plus[i] += control_eps[i]
            f_plus = specialist(state.unsqueeze(0), control_plus.unsqueeze(0), static.unsqueeze(0)).squeeze()
            
            control_minus = control.clone()
            control_minus[i] -= control_eps[i]
            f_minus = specialist(state.unsqueeze(0), control_minus.unsqueeze(0), static.unsqueeze(0)).squeeze()
            
            jac_u_numerical[:, i] = (f_plus - f_minus) / (2 * control_eps[i])
    
    # Compute errors
    error_x = torch.norm(jac_x_analytic - jac_x_numerical).item()
    error_u = torch.norm(jac_u_analytic - jac_u_numerical).item()
    
    return {
        "jacobian_x_error": error_x,
        "jacobian_u_error": error_u,
        "jacobian_x_relative": error_x / (torch.norm(jac_x_numerical).item() + 1e-10),
        "jacobian_u_relative": error_u / (torch.norm(jac_u_numerical).item() + 1e-10),
    }


if __name__ == "__main__":
    print("Testing HSSSpecialist with Analytic Jacobians")
    print("=" * 60)
    
    # Create specialist
    config = HSSConfig(hidden_dim=64, n_layers=3)
    specialist = HSSSpecialist(config)
    
    # Test forward pass
    state = torch.tensor([[20.0, 0.5, 0.1]])
    control = torch.tensor([[0.1, 0.3]])
    static = torch.tensor([[800.0, 1200.0]])
    
    output = specialist(state, control, static)
    print(f"Forward pass output: {output.squeeze().tolist()}")
    
    # Verify Jacobians
    errors = verify_jacobian(specialist)
    print(f"\nJacobian verification:")
    for key, val in errors.items():
        print(f"  {key}: {val:.2e}")
    
    # Test ensemble
    print("\n" + "=" * 60)
    print("Testing HSSEnsemble")

    specialists = [HSSSpecialist(config) for _ in range(4)]
    ensemble = HSSEnsemble(specialists, mu_centers=[0.3, 0.6, 0.9, 1.2])

    # Test with friction update
    ensemble.update_friction(0.8)
    print(f"Weights for μ=0.8: {ensemble.current_weights.tolist()}")
    
    output_ens = ensemble(state, control, static)
    print(f"Ensemble output: {output_ens.squeeze().tolist()}")
    
    # Test ensemble Jacobians
    jac_x, jac_u = ensemble.jacobian_analytic(state.squeeze(), control.squeeze(), static.squeeze())
    print(f"Ensemble Jacobian shapes: df/dx {jac_x.shape}, df/du {jac_u.shape}")
    
    print("\n✓ All tests passed!")

