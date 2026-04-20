#!/usr/bin/env python3
"""
Utilities for loading a trained ORCA mu-scale specialist library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from s2gpt_pinn.specialist import HSSConfig, HSSSpecialist, HSSEnsemble


@dataclass
class OrcaLibrary:
    ensemble: HSSEnsemble
    mu_scale_grid: np.ndarray  # [K]
    static_params: np.ndarray  # [2]
    hss_config: HSSConfig


def load_orca_library(lib_dir: str, device: torch.device) -> OrcaLibrary:
    lib_path = Path(lib_dir)
    manifest = json.loads((lib_path / "manifest.json").read_text())

    static_params = np.asarray(manifest["static_params"], dtype=np.float32)
    mu_grid = np.asarray(manifest["mu_scale_grid"], dtype=np.float32)
    hcfg = manifest["hss_config"]
    hss_cfg = HSSConfig(
        hidden_dim=int(hcfg["hidden_dim"]),
        n_layers=int(hcfg["n_layers"]),
        state_scale=tuple(float(x) for x in hcfg.get("state_scale", HSSConfig().state_scale)),
        control_scale=tuple(float(x) for x in hcfg.get("control_scale", HSSConfig().control_scale)),
        static_scale=tuple(float(x) for x in hcfg.get("static_scale", HSSConfig().static_scale)),
        output_scale=tuple(float(x) for x in hcfg.get("output_scale", HSSConfig().output_scale)),
    )

    specialists: List[HSSSpecialist] = []
    for f in manifest["files"]:
        s = HSSSpecialist(hss_cfg).to(device)
        sd = torch.load(lib_path / f, map_location=device)
        s.load_state_dict(sd)
        s.eval()
        specialists.append(s)

    # Use mu_scale grid as the RBF centers so Mode-A weighting matches the trained manifold.
    ensemble = HSSEnsemble(
        specialists,
        mu_centers=mu_grid.tolist(),
        rbf_width=float(np.clip(0.35 * float(mu_grid[1] - mu_grid[0]) if mu_grid.size > 1 else 0.1, 0.02, 0.5)),
    ).to(device)
    # store metadata for plots / weighting
    ensemble.specialists_info = [{"mu_scale": float(m)} for m in mu_grid.tolist()]  # type: ignore[attr-defined]
    return OrcaLibrary(ensemble=ensemble, mu_scale_grid=mu_grid, static_params=static_params, hss_config=hss_cfg)


def mode_a_weights_from_mu(
    mu_grid: np.ndarray,
    mu_current: float,
    sigma: float,
) -> np.ndarray:
    # RBF weights, normalized
    d = (mu_grid - float(mu_current)) / float(sigma)
    w = np.exp(-0.5 * d * d)
    s = float(np.sum(w))
    if s <= 0:
        return np.ones_like(w) / float(len(w))
    return w / s


def pick_in_out_mu(
    mu_min: float,
    mu_max: float,
    out_multiplier: float = 1.8,
) -> Tuple[float, float]:
    mu_in = 0.5 * (mu_min + mu_max)
    mu_out = out_multiplier * mu_max
    return float(mu_in), float(mu_out)


