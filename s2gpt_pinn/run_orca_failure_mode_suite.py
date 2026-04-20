#!/usr/bin/env python3
"""
Failure mode suite for ORCA closed-loop adaptation baselines.

Runs `s2gpt_pinn.paper_orca_closedloop` under several stress cases:
  - normal
  - no_excitation (delta constrained + constant vref)
  - short_window (Mode-B window <10 and estimator updates capped)
  - extreme_mu (mu_out forced > 2.0 via out_multiplier)

Writes:
  - per-case result directories
  - a single summary JSON pointing to artifacts and key diagnostics
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def _run_case(lib_dir: str, out_dir: Path, base_args: list[str]) -> None:
    cmd = [sys.executable, "-m", "s2gpt_pinn.paper_orca_closedloop", "--lib_dir", lib_dir, "--out_dir", str(out_dir), *base_args]
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=220)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--mu_change_step", type=int, default=110)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cases = {
        "normal": ["--failure_case", "normal"],
        "no_excitation": ["--failure_case", "no_excitation"],
        "short_window": ["--failure_case", "short_window"],
        "extreme_mu": ["--failure_case", "extreme_mu"],
    }

    summary: Dict[str, Any] = {"meta": {"seed": int(args.seed)}, "cases": {}}

    for name, extra in cases.items():
        od = out_root / name
        base = [
            "--seed",
            str(int(args.seed)),
            "--T",
            str(int(args.T)),
            "--dt",
            str(float(args.dt)),
            "--horizon",
            str(int(args.horizon)),
            "--mu_change_step",
            str(int(args.mu_change_step)),
            "--top_k",
            str(int(args.top_k)),
            *extra,
        ]
        _run_case(args.lib_dir, od, base)
        summary["cases"][name] = {
            "out_dir": str(od),
            "results_json": str(od / "orca_closedloop_results.json"),
        }

    out_json = Path(args.out_json).resolve() if args.out_json else (out_root / "failure_mode_suite.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()


