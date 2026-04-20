#!/usr/bin/env python3
"""
Train an ORCA-scale HSS specialist library over a mu_scale manifold and save it.

This is intentionally lightweight so it can run in CI / paper scripts:
- random state/control sampling in ORCA operating ranges
- supervised fit to ORCA physics oracle accelerations

Outputs:
  out_dir/
    manifest.json
    specialist_000.pt
    ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from s2gpt_pinn.orca_physics import OrcaParams, accelerations_torch
from s2gpt_pinn.specialist import HSSConfig, HSSSpecialist, HSSEnsemble


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_orca_data(
    n: int,
    device: torch.device,
    static_params: np.ndarray,
    mu_scale: float,
    vx_range=(0.2, 6.0),
    vy_range=(-1.5, 1.5),
    omega_range=(-8.0, 8.0),
    delta_range=(-0.35, 0.35),
    pwm_range=(-0.5, 1.0),
) -> Dict[str, torch.Tensor]:
    # shapes: state [n,3], control [n,2], static [n,2]
    vx = torch.empty(n, device=device).uniform_(*vx_range)
    vy = torch.empty(n, device=device).uniform_(*vy_range)
    omega = torch.empty(n, device=device).uniform_(*omega_range)
    state = torch.stack([vx, vy, omega], dim=1)

    delta = torch.empty(n, device=device).uniform_(*delta_range)
    pwm = torch.empty(n, device=device).uniform_(*pwm_range)
    control = torch.stack([delta, pwm], dim=1)

    static = torch.as_tensor(static_params, dtype=torch.float32, device=device).reshape(1, 2).repeat(n, 1)
    y = accelerations_torch(state, control, static, OrcaParams(), mu_scale=mu_scale)
    return {"state": state, "control": control, "static": static, "y": y}


def train_specialist(
    cfg: HSSConfig,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> HSSSpecialist:
    model = HSSSpecialist(cfg).to(device)
    model.train()

    ds = TensorDataset(data["state"], data["control"], data["static"], data["y"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for state, control, static, y in dl:
            pred = model(state, control, static)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def build_and_train_library(
    out_dir: Path,
    seed: int,
    device: str,
    n_specialists: int,
    hidden_dim: int,
    n_layers: int,
    mu_min: float,
    mu_max: float,
    n_train: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(seed)
    dev = torch.device(device)

    # ORCA static params are constant for these benchmarks
    orca = OrcaParams()
    static_np = np.array([orca.mass, orca.Iz], dtype=np.float32)

    mu_scales = np.linspace(mu_min, mu_max, n_specialists).astype(np.float32)
    specialists: List[HSSSpecialist] = []

    # ORCA-specific normalization: keep normalized values O(1) in the operating region.
    cfg = HSSConfig(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        state_scale=(6.0, 2.0, 10.0),
        control_scale=(0.4, 1.0),
        static_scale=(0.05, 50e-6),
        output_scale=(5.0, 5.0, 100.0),
    )
    for i, mu in enumerate(mu_scales):
        data = sample_orca_data(n_train, dev, static_np, float(mu))
        model = train_specialist(cfg, data, dev, epochs=epochs, batch_size=batch_size, lr=lr)
        specialists.append(model)
        torch.save(model.state_dict(), out_dir / f"specialist_{i:03d}.pt")

    # Build an ensemble for convenience (not strictly required for saved weights)
    ensemble = HSSEnsemble(specialists)
    # Record metadata for loading
    manifest = {
        "seed": seed,
        "device_trained": device,
        "orca_params": asdict(orca),
        "static_params": static_np.tolist(),
        "training": {
            "n_train": int(n_train),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "sampling_ranges": {
                "vx": [0.2, 6.0],
                "vy": [-1.5, 1.5],
                "omega": [-8.0, 8.0],
                "delta": [-0.35, 0.35],
                "pwm": [-0.5, 1.0],
            },
        },
        "hss_config": {
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "state_scale": list(cfg.state_scale),
            "control_scale": list(cfg.control_scale),
            "static_scale": list(cfg.static_scale),
            "output_scale": list(cfg.output_scale),
        },
        "n_specialists": n_specialists,
        "mu_scale_grid": mu_scales.tolist(),
        "files": [f"specialist_{i:03d}.pt" for i in range(n_specialists)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Save a single ensemble bundle too (optional convenience)
    torch.save(
        {"manifest": manifest, "state_dicts": [s.state_dict() for s in specialists]},
        out_dir / "ensemble_bundle.pt",
    )
    return out_dir / "manifest.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n_specialists", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--mu_min", type=float, default=0.6)
    p.add_argument("--mu_max", type=float, default=1.4)
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    manifest_path = build_and_train_library(
        out_dir=Path(args.out_dir),
        seed=args.seed,
        device=args.device,
        n_specialists=args.n_specialists,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        n_train=args.n_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"[OK] wrote {manifest_path}")


if __name__ == "__main__":
    main()


