#!/usr/bin/env python3
"""
Multi-seed runner + aggregator for closed-loop experiments.

Primary target:
  - `s2gpt_pinn.paper_orca_closedloop` (writes `orca_closedloop_results.json`)

This script:
  - runs seeds [42..46] (default)
  - aggregates metrics into mean/std + bootstrap 95% CI (1000 resamples)
  - performs paired t-tests (Bonferroni-corrected) for selected controller comparisons

Example:
  cd /home/enzo/HYDRA/NMPC
  python -m s2gpt_pinn.run_multi_seed_closedloop \
    --module s2gpt_pinn.paper_orca_closedloop \
    --lib_dir /home/enzo/HYDRA/NMPC/s2gpt_pinn/orca_library_trained \
    --out_root /home/enzo/HYDRA/NMPC/s2gpt_pinn/paper_orca_closedloop_multiseed \
    --T 220 --dt 0.02 --horizon 10 --mu_change_step 110 --out_multiplier 1.25 --top_k 4
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy import stats

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

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
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n": int(arr.size),
    }


def _paired_ttest(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan")
    return float(stats.ttest_rel(a, b).pvalue)


def _bonferroni(pvals: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    m = max(len(pvals), 1)
    out = {}
    for k, p in pvals.items():
        out[k] = {"p": float(p), "p_bonferroni": float(min(p * m, 1.0)), "m": float(m)}
    return out


def _run_one(module: str, lib_dir: str, out_dir: Path, seed: int, extra_args: List[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        module,
        "--lib_dir",
        lib_dir,
        "--out_dir",
        str(out_dir),
        "--seed",
        str(seed),
        *extra_args,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def _load_orca_results(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "orca_closedloop_results.json"
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", type=str, default="s2gpt_pinn.paper_orca_closedloop")
    ap.add_argument("--lib_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--ci_resamples", type=int, default=1000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--show_progress", action="store_true", help="Show per-seed progress bar and enable child tqdm output.")
    # Remaining args are passed through to the experiment module.
    args, unknown = ap.parse_known_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Run experiments
    per_seed = []
    seeds_iter = args.seeds
    if args.show_progress and tqdm is not None:
        seeds_iter = tqdm(list(args.seeds), desc="Seeds", leave=True)

    for s in seeds_iter:
        od = out_root / f"seed_{int(s)}"
        results_path = od / "orca_closedloop_results.json"
        if results_path.exists():
            msg = f"[SKIP] seed {int(s)} already complete"
            print(msg)
            if args.show_progress and tqdm is not None:
                try:
                    seeds_iter.set_postfix_str(msg)  # type: ignore[attr-defined]
                except Exception:
                    pass
        else:
            msg = f"[RUN] seed {int(s)}"
            print(msg)
            t0 = time.perf_counter()
            child_args = list(unknown)
            # If the child supports it (paper_orca_closedloop does), enable tqdm inside the rollout loop.
            if args.show_progress and "--show_progress" not in child_args:
                child_args.append("--show_progress")
            _run_one(args.module, args.lib_dir, od, int(s), child_args)
            dt_s = time.perf_counter() - t0
            msg = f"[DONE] seed {int(s)} in {dt_s/60.0:.1f} min"
            print(msg)
            if args.show_progress and tqdm is not None:
                try:
                    seeds_iter.set_postfix_str(msg)  # type: ignore[attr-defined]
                except Exception:
                    pass
        per_seed.append((int(s), od))

    # Aggregate metrics (ORCA closed-loop schema)
    scenarios = [
        ("in", "closed_loop_in_manifold"),
        ("out", "closed_loop_out_of_manifold"),
    ]

    # Collect per-seed values
    values: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    # Also keep estimator convergence steps (ints), and UKF conditioning summary
    extra: Dict[str, Any] = {"convergence_steps": {}, "ukf_cond_S": {}}

    for scen_tag, scen_key in scenarios:
        values[scen_tag] = {}
        extra["convergence_steps"][scen_tag] = {}
        extra["ukf_cond_S"][scen_tag] = {}

        for seed, od in per_seed:
            data = _load_orca_results(od)
            block = data[scen_key]
            tracking = block["tracking"]
            solve_ms = block["solve_ms"]
            est_diag = block.get("estimator_diagnostics", {})

            # initialize dict structure lazily
            for ctrl in tracking.keys():
                values[scen_tag].setdefault(ctrl, {})
                for m in ["rmse_vx", "rmse_vy", "rmse_omega"]:
                    values[scen_tag][ctrl].setdefault(m, [])
                for m in ["solve_mean_ms", "solve_p95_ms", "solve_max_ms"]:
                    values[scen_tag][ctrl].setdefault(m, [])

            for ctrl, mets in tracking.items():
                values[scen_tag][ctrl]["rmse_vx"].append(float(mets["rmse_vx"]))
                values[scen_tag][ctrl]["rmse_vy"].append(float(mets["rmse_vy"]))
                values[scen_tag][ctrl]["rmse_omega"].append(float(mets["rmse_omega"]))

                values[scen_tag][ctrl]["solve_mean_ms"].append(float(solve_ms[ctrl]["mean"]))
                values[scen_tag][ctrl]["solve_p95_ms"].append(float(solve_ms[ctrl]["p95"]))
                values[scen_tag][ctrl]["solve_max_ms"].append(float(solve_ms[ctrl]["max"]))

            # convergence steps (per seed) for estimators and Mode-B weights
            for ctrl in ["phys_mu_rls", "phys_mu_ukf"]:
                if ctrl in est_diag:
                    extra["convergence_steps"][scen_tag].setdefault(ctrl, [])
                    v = est_diag[ctrl].get("convergence_step_tol0p05_hold5", None)
                    extra["convergence_steps"][scen_tag][ctrl].append(float(np.nan if v is None else v))
            if "mode_b_weights" in est_diag:
                extra["convergence_steps"][scen_tag].setdefault("ens_w", [])
                extra["convergence_steps"][scen_tag].setdefault("ens_w_topk", [])
                v = est_diag["mode_b_weights"].get("ens_w_convergence_step_dW_le_1e-3_hold5", None)
                extra["convergence_steps"][scen_tag]["ens_w"].append(float(np.nan if v is None else v))
                v = est_diag["mode_b_weights"].get("ens_w_topk_convergence_step_dW_le_1e-3_hold5", None)
                extra["convergence_steps"][scen_tag]["ens_w_topk"].append(float(np.nan if v is None else v))

            # UKF conditioning summary: take p95 over time series for each seed
            if "phys_mu_ukf" in est_diag and "ukf_cond_S" in est_diag["phys_mu_ukf"]:
                cond_series = np.asarray(est_diag["phys_mu_ukf"]["ukf_cond_S"], dtype=np.float64)
                cond_series = cond_series[np.isfinite(cond_series)]
                if cond_series.size:
                    extra["ukf_cond_S"][scen_tag].setdefault("p95", [])
                    extra["ukf_cond_S"][scen_tag]["p95"].append(float(np.percentile(cond_series, 95)))

    # Compute summaries
    summary: Dict[str, Any] = {
        "meta": {
            "module": str(args.module),
            "lib_dir": str(Path(args.lib_dir).resolve()),
            "out_root": str(out_root),
            "seeds": [int(s) for s in args.seeds],
            "ci_resamples": int(args.ci_resamples),
            "ci_alpha": float(args.ci_alpha),
            "python": sys.version,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "passthrough_args": unknown,
        },
        "scenarios": {},
        "tests": {},
        "extra": extra,
    }

    for scen_tag, _scen_key in scenarios:
        summary["scenarios"][scen_tag] = {}
        for ctrl, mets in values[scen_tag].items():
            summary["scenarios"][scen_tag][ctrl] = {k: _summary_stats(v, args.ci_resamples, args.ci_alpha, seed=0) for k, v in mets.items()}

    # Paired t-tests (Bonferroni): focus on vx RMSE in out-of-manifold scenario
    comparisons = [
        ("out: ens_w vs phys_mu_rls", ("out", "ens_w", "phys_mu_rls", "rmse_vx")),
        ("out: ens_w vs phys_mu_ukf", ("out", "ens_w", "phys_mu_ukf", "rmse_vx")),
        ("out: phys_mu vs phys_mu_rls", ("out", "phys_mu", "phys_mu_rls", "rmse_vx")),
        ("out: phys_mu vs phys_mu_ukf", ("out", "phys_mu", "phys_mu_ukf", "rmse_vx")),
    ]
    pvals = {}
    for label, (scen, a_ctrl, b_ctrl, metric) in comparisons:
        if a_ctrl in values[scen] and b_ctrl in values[scen]:
            pvals[label] = _paired_ttest(values[scen][a_ctrl][metric], values[scen][b_ctrl][metric])
        else:
            pvals[label] = float("nan")
    summary["tests"]["paired_ttests_rmse_vx_out"] = _bonferroni(pvals)

    out_json = Path(args.out_json).resolve() if args.out_json else (out_root / "multiseed_summary.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()


