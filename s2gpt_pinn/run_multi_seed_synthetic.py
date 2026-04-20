#!/usr/bin/env python3
"""
Multi-seed runner + aggregator for `s2gpt_pinn.paper_synthetic_minimal`.

Runs multiple seeds, collects `synthetic_results.json`, and aggregates mean/std
with bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _bootstrap_ci_mean(x: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, x.size, size=x.size)
        means.append(float(np.mean(x[idx])))
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _summary_stats(x: List[float], ci_resamples: int = 1000, ci_alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64)
    lo, hi = _bootstrap_ci_mean(arr, n_resamples=ci_resamples, alpha=ci_alpha, seed=seed)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "ci_lower": float(lo), "ci_upper": float(hi), "n": int(arr.size)}


def _run_one(lib_dir: str, out_dir: Path, seed: int, extra_args: List[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "s2gpt_pinn.paper_synthetic_minimal",
        "--out_dir",
        str(out_dir),
        "--lib_dir",
        lib_dir,
        "--seed",
        str(int(seed)),
        *extra_args,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--ci_resamples", type=int, default=1000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--out_json", type=str, default=None)
    args, unknown = ap.parse_known_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    per_seed: List[Tuple[int, Path]] = []
    for s in args.seeds:
        od = out_root / f"seed_{int(s)}"
        _run_one(args.lib_dir, od, int(s), unknown)
        per_seed.append((int(s), od))

    # Collect metrics
    metrics: Dict[str, List[float]] = {}
    for seed, od in per_seed:
        j = json.loads((od / "synthetic_results.json").read_text())
        # flatten key metrics into a dict of lists
        for group in ["one_step_state_error", "one_step_vx_error", "one_step_accel_error", "one_step_dvx_error"]:
            for k, v in j.get(group, {}).items():
                metrics.setdefault(f"{group}.{k}", []).append(float(v))

    summary = {
        "meta": {
            "lib_dir": str(Path(args.lib_dir).resolve()),
            "out_root": str(out_root),
            "seeds": [int(s) for s in args.seeds],
            "ci_resamples": int(args.ci_resamples),
            "ci_alpha": float(args.ci_alpha),
            "python": sys.version,
            "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
            "passthrough_args": unknown,
        },
        "metrics": {k: _summary_stats(v, args.ci_resamples, args.ci_alpha, seed=0) for k, v in metrics.items()},
    }

    out_json = Path(args.out_json).resolve() if args.out_json else (out_root / "multiseed_synthetic_summary.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()


