#!/usr/bin/env python3
"""
H-SS Conditioned Specialist Network Architecture

Feed-forward MLP that maps (State, Control, StaticParams) -> Acceleration.

CONDITIONED SPECIALIST APPROACH:
- Static Parameters (m, Iz): Passed as INPUTS to the network (known from Phase 1)
- Dynamic Parameters (μ, Cd, B, C, E): Captured by LIBRARY DIVERSITY
- Each specialist represents ONE friction/aero configuration
- Mass/Inertia domain randomization during training teaches physics relationship a ∝ F/m

This solves the combinatorial explosion:
- Instead of O(mass × friction) specialists
- We have O(friction) specialists that accept mass as input
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm


class HSSSpecialist(nn.Module):
    """
    H-SS Conditioned Specialist Network.
    
    Input: (vx, vy, ω, δ, T, m, Iz) - state, control, and static params (7 features)
    Output: (dvx, dvy, dω) - state derivatives (3 accelerations)
    
    The dynamic parameters (μ, Cd, etc.) are captured by library diversity.
    Static parameters (m, Iz) are explicit inputs, enabling mass generalization.
    """
    
    def __init__(self, hidden_dim: int = 64, n_layers: int = 3):
        """
        Initialize H-SS conditioned specialist network.
        
        Args:
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        # Input: [vx, vy, ω, δ, T, m_norm, Iz_norm] = 7 features
        # Output: [dvx, dvy, dω] = 3 accelerations
        input_dim = 7
        output_dim = 3
        
        # Input normalization scales
        # State: [vx (m/s), vy (m/s), omega (rad/s)]
        # Control: [delta (rad), throttle (-1 to 1)]
        # Static: [mass (kg), inertia (kg*m^2)]
        self.register_buffer("state_scale", torch.tensor([40.0, 5.0, 3.0]))
        self.register_buffer("control_scale", torch.tensor([0.5, 1.0]))
        self.register_buffer("static_scale", torch.tensor([1000.0, 2000.0]))  # m, Iz
        
        # Output scaling (typical acceleration magnitudes)
        self.register_buffer("output_scale", torch.tensor([10.0, 10.0, 5.0]))
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, control: torch.Tensor,
                static_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict accelerations conditioned on static parameters.
        
        Args:
            state: [batch, 3] tensor of (vx, vy, omega)
            control: [batch, 2] tensor of (delta, throttle)
            static_params: [batch, 2] tensor of (mass, Iz)
            
        Returns:
            accelerations: [batch, 3] tensor of (dvx, dvy, domega)
        """
        # Normalize inputs
        state_norm = state / self.state_scale
        control_norm = control / self.control_scale
        static_norm = static_params / self.static_scale
        
        # Concatenate all features
        x = torch.cat([state_norm, control_norm, static_norm], dim=1)
        
        # Forward through network
        out = self.net(x)
        
        # Scale output to physical range
        return out * self.output_scale
    
    def predict_numpy(self, state: np.ndarray, control: np.ndarray,
                     static_params: np.ndarray) -> np.ndarray:
        """
        Convenience method for numpy inputs.
        
        Args:
            state: [batch, 3] or [3,] array of (vx, vy, omega)
            control: [batch, 2] or [2,] array of (delta, throttle)
            static_params: [batch, 2] or [2,] array of (mass, Iz)
            
        Returns:
            accelerations: [batch, 3] or [3,] array
        """
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
        
        if single_sample:
            return result[0]
        return result


class HSSLibrary:
    """
    Manager for H-SS conditioned specialist library.
    
    Each specialist represents ONE dynamic parameter configuration (friction/aero).
    Static parameters (m, Iz) are passed at inference time.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize H-SS library.
        
        Args:
            model_dir: Directory containing trained specialist models
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.specialists: List[HSSSpecialist] = []
        self.param_manifest: Dict[str, Dict] = {}  # ID -> dynamic params dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model_dir and self.model_dir.exists():
            self._load_library()
    
    def _load_library(self):
        """Load all specialists from model directory."""
        import json
        
        manifest_path = self.model_dir / "param_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.param_manifest = json.load(f)
        
        # Try to load config for architecture info
        config_path = self.model_dir / "config.json"
        hidden_dim = 64
        n_layers = 3
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                hidden_dim = config.get('hidden_dim', 64)
                n_layers = config.get('n_layers', 3)
        
        model_files = sorted(self.model_dir.glob("specialist_*.pt"))
        for model_path in model_files:
            model = HSSSpecialist(hidden_dim=hidden_dim, n_layers=n_layers)
            model.load_state_dict(torch.load(model_path, map_location=self.device,
                                            weights_only=True))
            model.to(self.device)
            model.eval()
            self.specialists.append(model)
        
        print(f"Loaded {len(self.specialists)} H-SS conditioned specialists from {self.model_dir}")
    
    def add_specialist(self, model: HSSSpecialist, dynamic_params: Dict[str, float],
                      specialist_id: int):
        """
        Add a trained specialist to the library.
        
        Args:
            model: Trained HSSSpecialist
            dynamic_params: Dynamic param dict (μ, Cd, B, C, E) - NOT mass!
            specialist_id: Unique identifier
        """
        model.to(self.device)
        model.eval()
        self.specialists.append(model)
        self.param_manifest[str(specialist_id)] = dynamic_params
    
    def save(self, model_dir: str):
        """Save library to disk."""
        import json
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.specialists):
            torch.save(model.state_dict(), model_dir / f"specialist_{i:03d}.pt")
        
        with open(model_dir / "param_manifest.json", 'w') as f:
            json.dump(self.param_manifest, f, indent=2)
        
        print(f"Saved {len(self.specialists)} conditioned specialists to {model_dir}")
    
    def get_predictions(self, state: np.ndarray, control: np.ndarray,
                       static_params: np.ndarray) -> np.ndarray:
        """
        Get predictions from ALL specialists given locked static params.
        
        Args:
            state: [3,] array of (vx, vy, omega)
            control: [2,] array of (delta, throttle)
            static_params: [2,] array of (locked_m, locked_Iz)
            
        Returns:
            predictions: [3, num_specialists] array of acceleration predictions
        """
        predictions = []
        for model in self.specialists:
            pred = model.predict_numpy(state, control, static_params)
            predictions.append(pred)
        
        return np.array(predictions).T  # [3, num_specialists]
    
    def get_specialist_params(self, specialist_id: int) -> Dict[str, float]:
        """Get the dynamic parameter dict for a specialist."""
        return self.param_manifest.get(str(specialist_id), {})
    
    @property
    def num_specialists(self) -> int:
        return len(self.specialists)


def train_specialist(
    model: HSSSpecialist,
    states: np.ndarray,
    controls: np.ndarray,
    static_params: np.ndarray,
    accelerations: np.ndarray,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    progress: bool = False,
) -> float:
    """
    Train a single H-SS conditioned specialist.
    
    CRITICAL: The training data should have DOMAIN RANDOMIZED mass/inertia.
    Even though this specialist represents one friction configuration,
    the mass varies across samples to teach the network a ∝ F/m.
    
    Args:
        model: HSSSpecialist to train
        states: [n_samples, 3] array of states
        controls: [n_samples, 2] array of controls
        static_params: [n_samples, 2] array of (mass, Iz) - DOMAIN RANDOMIZED
        accelerations: [n_samples, 3] array of target accelerations
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Torch device
        progress: Show progress bar
        
    Returns:
        Final training loss
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.train()
    
    # Convert to tensors
    states_t = torch.from_numpy(states).float().to(device)
    controls_t = torch.from_numpy(controls).float().to(device)
    static_t = torch.from_numpy(static_params).float().to(device)
    accels_t = torch.from_numpy(accelerations).float().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    n_samples = len(states)
    final_loss = float('inf')
    
    epoch_iter = tqdm(range(epochs), desc="Training", disable=not progress, leave=False)
    for epoch in epoch_iter:
        permutation = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0
        
        for idx in permutation.split(batch_size):
            optimizer.zero_grad()
            
            pred = model(states_t[idx], controls_t[idx], static_t[idx])
            loss = criterion(pred, accels_t[idx])
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        final_loss = epoch_loss / n_batches
        epoch_iter.set_postfix(loss=f"{final_loss:.6f}")
    
    model.eval()
    return final_loss
