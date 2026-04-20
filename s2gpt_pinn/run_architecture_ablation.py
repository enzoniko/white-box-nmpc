#!/usr/bin/env python3
"""
Phase 3: Architecture Ablation Study

Systematically characterize the efficiency boundary by varying neural surrogate architecture parameters.
This study maps how architectural choices affect the computational efficiency vs. adaptive performance trade-off.

Parameter Grid:
  - n_specialists: [4, 8, 16]
  - hidden_dim: [32, 64, 128]
  - n_layers: [2, 3, 4]
  - top_k: [2, 4, 8]

For each configuration:
  1. Train specialist library with specified architecture
  2. Run identical-OCP efficiency comparison (BayesRace harness)
  3. Run closed-loop adaptation test (in-manifold and out-of-manifold)
  4. Extract computational metrics (sparsity, timing, memory)

Output:
  - Trained libraries per configuration
  - Efficiency comparison results
  - Closed-loop tracking results
  - Aggregated summary JSON
  - Visualization plots (Pareto, heatmaps, scatter plots)

Example:
  cd /home/enzo/HYDRA/NMPC
  MPLBACKEND=Agg python -m s2gpt_pinn.run_architecture_ablation \
    --out_root /home/enzo/HYDRA/NMPC/s2gpt_pinn/phase3_ablation_results \
    --n_workers 4 \
    --skip_slow_configs
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import platform
import psutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None  # Fallback if seaborn not available

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


_REPO_ROOT = Path(__file__).resolve().parents[1]  # .../NMPC


@dataclass
class ArchitectureConfig:
    """Single architecture configuration."""
    n_specialists: int
    hidden_dim: int
    n_layers: int
    top_k: int

    def __str__(self) -> str:
        return f"specs_{self.n_specialists}_hd{self.hidden_dim}_layers{self.n_layers}_topk{self.top_k}"

    def to_dict(self) -> Dict[str, int]:
        return {
            "n_specialists": self.n_specialists,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "top_k": self.top_k,
        }


# Parameter grid as specified
PARAM_GRID = {
    'n_specialists': [4, 8, 16],
    'hidden_dim': [32, 64, 128],
    'n_layers': [2, 3, 4],
    'top_k': [2, 4, 8],
}


def generate_configurations() -> List[ArchitectureConfig]:
    """Generate all parameter combinations."""
    configs = []
    for n_spec, hd, n_lay, tk in itertools.product(
        PARAM_GRID['n_specialists'],
        PARAM_GRID['hidden_dim'],
        PARAM_GRID['n_layers'],
        PARAM_GRID['top_k'],
    ):
        # Skip invalid top_k > n_specialists
        if tk > n_spec:
            continue
        configs.append(ArchitectureConfig(
            n_specialists=n_spec,
            hidden_dim=hd,
            n_layers=n_lay,
            top_k=tk,
        ))
    return configs


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def train_library(config: ArchitectureConfig, out_root: Path, seed: int = 42) -> Tuple[bool, Path]:
    """Train specialist library for given architecture configuration."""
    lib_dir = out_root / str(config) / "library"
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Check if already trained
    manifest_path = lib_dir / "manifest.json"
    if manifest_path.exists():
        return True, lib_dir

    # Train library
    cmd = [
        sys.executable, "-m", "s2gpt_pinn.train_orca_library",
        "--out_dir", str(lib_dir),
        "--seed", str(seed),
        "--device", "cpu",
        "--n_specialists", str(config.n_specialists),
        "--hidden_dim", str(config.hidden_dim),
        "--n_layers", str(config.n_layers),
        "--n_train", "80000",  # Fixed training data size
        "--epochs", "6",
        "--batch_size", "2048",
        "--lr", "1e-3",
    ]

    retcode, stdout, stderr = _run_cmd(cmd, cwd=_REPO_ROOT, timeout=3600.0)  # 1 hour timeout
    if retcode != 0:
        return False, lib_dir

    if not manifest_path.exists():
        return False, lib_dir

    return True, lib_dir


def run_efficiency_comparison(config: ArchitectureConfig, lib_dir: Path, out_root: Path, horizon: int = 10) -> Tuple[bool, Dict[str, Any]]:
    """Run efficiency comparison (BayesRace harness)."""
    config_dir = out_root / str(config)
    eff_dir = config_dir / "efficiency"
    eff_dir.mkdir(parents=True, exist_ok=True)

    eff_json = eff_dir / "efficiency_comparison.json"
    if eff_json.exists():
        try:
            with eff_json.open() as f:
                return True, json.load(f)
        except Exception:
            pass

    # Run efficiency comparison
    cmd = [
        sys.executable, "-m", "bayes_race.mpc.run_paper_efficiency_comparison",
        "--out_dir", str(eff_dir),
        "--out_json", str(eff_json),
        "--lib_dir", str(lib_dir),
        "--n_steps", "40",
        "--horizon", str(horizon),
        "--Ts", "0.02",
        "--seed", "42",
        "--mu_fixed", "1.0",
        "--ipopt_max_iter", "100",
        "--ipopt_print_level", "0",
        "--timing_stats_solves", "1",
    ]

    retcode, stdout, stderr = _run_cmd(cmd, cwd=_REPO_ROOT / "bayesrace", timeout=1800.0)  # 30 min timeout
    if retcode != 0:
        return False, {"error": f"Command failed: {stderr}"}

    if not eff_json.exists():
        return False, {"error": "Output JSON not created"}

    try:
        with eff_json.open() as f:
            return True, json.load(f)
    except Exception as e:
        return False, {"error": f"Failed to parse JSON: {e}"}


def run_closed_loop(config: ArchitectureConfig, lib_dir: Path, out_root: Path, horizon: int = 10) -> Tuple[bool, Dict[str, Any]]:
    """Run closed-loop adaptation test."""
    config_dir = out_root / str(config)
    closedloop_dir = config_dir / "closedloop"
    closedloop_dir.mkdir(parents=True, exist_ok=True)

    closedloop_json = closedloop_dir / "orca_closedloop_results.json"
    if closedloop_json.exists():
        try:
            with closedloop_json.open() as f:
                return True, json.load(f)
        except Exception:
            pass

    # Run closed-loop experiment
    cmd = [
        sys.executable, "-m", "s2gpt_pinn.paper_orca_closedloop",
        "--out_dir", str(closedloop_dir),
        "--lib_dir", str(lib_dir),
        "--seed", "42",
        "--dt", "0.02",
        "--T", "220",
        "--horizon", str(horizon),
        "--top_k", str(config.top_k),
        "--vref_low", "1.5",
        "--vref_high", "3.0",
        "--vref_change_step", "120",
        "--mu_change_step", "110",
        "--out_multiplier", "1.25",
        "--ipopt_max_iter", "80",
    ]

    retcode, stdout, stderr = _run_cmd(cmd, cwd=_REPO_ROOT, timeout=3600.0)  # 1 hour timeout
    if retcode != 0:
        return False, {"error": f"Command failed: {stderr}"}

    if not closedloop_json.exists():
        return False, {"error": "Output JSON not created"}

    try:
        with closedloop_json.open() as f:
            return True, json.load(f)
    except Exception as e:
        return False, {"error": f"Failed to parse JSON: {e}"}


def extract_metrics(config: ArchitectureConfig, eff_results: Dict[str, Any], closedloop_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all metrics from efficiency and closed-loop results."""
    metrics = {
        "config": config.to_dict(),
    }

    # Efficiency metrics
    if "models" in eff_results and "surrogate_native" in eff_results["models"]:
        surr = eff_results["models"]["surrogate_native"]
        metrics["efficiency"] = {
            "build_ms": surr.get("build_ms", np.nan),
            "solve_mean_ms": surr.get("solve_ms", {}).get("mean_ms", np.nan),
            "solve_std_ms": surr.get("solve_ms", {}).get("std_ms", np.nan),
            "solve_p95_ms": surr.get("solve_ms", {}).get("p95_ms", np.nan),
            "solve_max_ms": surr.get("solve_ms", {}).get("max_ms", np.nan),
            "jac_eval_ms": surr.get("dynamics_bench", {}).get("jac_ms_per_call", np.nan),
            "hess_eval_ms": surr.get("symbolic_bench", {}).get("hess_lag_ms_per_call", np.nan),
            "linear_solver_ms": surr.get("ipopt_verbose_solve_ms", {}).get("ipopt_timing_stats", {}).get("timing_statistics", {}).get("PDSystemSolverTotal", {}).get("cpu_s", np.nan) * 1000.0 if surr.get("ipopt_verbose_solve_ms", {}).get("ipopt_timing_stats", {}).get("timing_statistics") else np.nan,
        }
        # Structural metrics
        jac_sp = surr.get("jacobian_sparsity", {})
        metrics["structural"] = {
            "jacobian_density": jac_sp.get("density_percent", np.nan),
            "graph_nodes": surr.get("nlp_meta", {}).get("n_decision", np.nan),
            "memory_mb": np.nan,  # Not directly available, would need separate measurement
        }
    else:
        metrics["efficiency"] = {}
        metrics["structural"] = {}

    # Performance metrics
    if "closed_loop_in_manifold" in closedloop_results and "closed_loop_out_of_manifold" in closedloop_results:
        in_man = closedloop_results["closed_loop_in_manifold"]
        out_man = closedloop_results["closed_loop_out_of_manifold"]
        
        # Use ens_w_topk as the neural surrogate controller
        tracking_in = in_man.get("tracking", {}).get("ens_w_topk", {})
        tracking_out = out_man.get("tracking", {}).get("ens_w_topk", {})
        solve_in = in_man.get("solve_ms", {}).get("ens_w_topk", {})
        
        metrics["performance"] = {
            "in_manifold_rmse_vx": tracking_in.get("rmse_vx", np.nan),
            "out_manifold_rmse_vx": tracking_out.get("rmse_vx", np.nan),
            "adaptation_latency_ms": solve_in.get("mean", np.nan),
            "convergence_steps": in_man.get("estimator_diagnostics", {}).get("mode_b_weights", {}).get("ens_w_topk_convergence_step_dW_le_1e-3_hold5", np.nan),
        }
    else:
        metrics["performance"] = {}

    return metrics


def process_configuration(args: Tuple[ArchitectureConfig, Path, int, int]) -> Tuple[ArchitectureConfig, Dict[str, Any], Optional[str]]:
    """Process a single configuration (for parallel execution)."""
    config, out_root, horizon, seed = args
    error_msg = None

    try:
        # Step 1: Train library
        success, lib_dir = train_library(config, out_root, seed=seed)
        if not success:
            return config, {}, f"Training failed for {config}"

        # Step 2: Efficiency comparison
        success, eff_results = run_efficiency_comparison(config, lib_dir, out_root, horizon=horizon)
        if not success:
            return config, {}, f"Efficiency comparison failed for {config}: {eff_results.get('error', 'unknown')}"

        # Step 3: Closed-loop
        success, closedloop_results = run_closed_loop(config, lib_dir, out_root, horizon=horizon)
        if not success:
            return config, {}, f"Closed-loop failed for {config}: {closedloop_results.get('error', 'unknown')}"

        # Step 4: Extract metrics
        metrics = extract_metrics(config, eff_results, closedloop_results)

        return config, metrics, None

    except Exception as e:
        return config, {}, f"Exception in {config}: {e}"


def generate_visualizations(configs: List[ArchitectureConfig], results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate visualization plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter valid results
    valid_results = [r for r in results if "efficiency" in r and "performance" in r and r.get("efficiency", {}).get("solve_mean_ms") is not None and not np.isnan(r["efficiency"].get("solve_mean_ms", np.nan))]

    if len(valid_results) < 2:
        print(f"[WARN] Not enough valid results for visualization ({len(valid_results)}/{len(results)})")
        return

    # 1. Efficiency-Performance Pareto Frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    solve_times = [r["efficiency"]["solve_mean_ms"] for r in valid_results]
    rmse_in = [r["performance"]["in_manifold_rmse_vx"] for r in valid_results]
    
    scatter = ax.scatter(solve_times, rmse_in, c=range(len(valid_results)), cmap='viridis', s=100, alpha=0.6)
    ax.set_xlabel("Solve Time (ms)")
    ax.set_ylabel("In-Manifold RMSE vx (m/s)")
    ax.set_title("Efficiency-Performance Pareto Frontier")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Configuration Index")
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_frontier.png", dpi=200)
    plt.close()

    # 2. Architecture Sensitivity Heatmaps
    # Prepare data for heatmaps
    n_specs = sorted(set(r["config"]["n_specialists"] for r in valid_results))
    hdims = sorted(set(r["config"]["hidden_dim"] for r in valid_results))
    
    # Heatmap: n_specialists vs hidden_dim, value = solve time
    solve_heatmap = np.full((len(hdims), len(n_specs)), np.nan)
    for r in valid_results:
        if "efficiency" in r and "solve_mean_ms" in r["efficiency"]:
            i = hdims.index(r["config"]["hidden_dim"])
            j = n_specs.index(r["config"]["n_specialists"])
            solve_heatmap[i, j] = r["efficiency"]["solve_mean_ms"]

    fig, ax = plt.subplots(figsize=(8, 6))
    if sns:
        sns.heatmap(solve_heatmap, annot=True, fmt='.1f', xticklabels=n_specs, yticklabels=hdims, cmap='YlOrRd', ax=ax)
    else:
        im = ax.imshow(solve_heatmap, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(n_specs)))
        ax.set_xticklabels(n_specs)
        ax.set_yticks(range(len(hdims)))
        ax.set_yticklabels(hdims)
        for i in range(len(hdims)):
            for j in range(len(n_specs)):
                if not np.isnan(solve_heatmap[i, j]):
                    ax.text(j, i, f'{solve_heatmap[i, j]:.1f}', ha='center', va='center')
        plt.colorbar(im, ax=ax)
    ax.set_xlabel("Number of Specialists")
    ax.set_ylabel("Hidden Dimension")
    ax.set_title("Mean Solve Time (ms) by Architecture")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_solve_time.png", dpi=200)
    plt.close()

    # Heatmap: RMSE improvement (out - in)
    rmse_heatmap = np.full((len(hdims), len(n_specs)), np.nan)
    for r in valid_results:
        if "performance" in r and "in_manifold_rmse_vx" in r["performance"] and "out_manifold_rmse_vx" in r["performance"]:
            i = hdims.index(r["config"]["hidden_dim"])
            j = n_specs.index(r["config"]["n_specialists"])
            rmse_diff = r["performance"]["out_manifold_rmse_vx"] - r["performance"]["in_manifold_rmse_vx"]
            rmse_heatmap[i, j] = rmse_diff

    fig, ax = plt.subplots(figsize=(8, 6))
    if sns:
        sns.heatmap(rmse_heatmap, annot=True, fmt='.3f', xticklabels=n_specs, yticklabels=hdims, cmap='RdYlGn', center=0, ax=ax)
    else:
        im = ax.imshow(rmse_heatmap, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
        ax.set_xticks(range(len(n_specs)))
        ax.set_xticklabels(n_specs)
        ax.set_yticks(range(len(hdims)))
        ax.set_yticklabels(hdims)
        for i in range(len(hdims)):
            for j in range(len(n_specs)):
                if not np.isnan(rmse_heatmap[i, j]):
                    ax.text(j, i, f'{rmse_heatmap[i, j]:.3f}', ha='center', va='center')
        plt.colorbar(im, ax=ax)
    ax.set_xlabel("Number of Specialists")
    ax.set_ylabel("Hidden Dimension")
    ax.set_title("RMSE Degradation (Out - In Manifold)")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_rmse_degradation.png", dpi=200)
    plt.close()

    # 3. Top-K Speedup Analysis
    topk_values = sorted(set(r["config"]["top_k"] for r in valid_results))
    solve_by_topk = {tk: [] for tk in topk_values}
    for r in valid_results:
        if "efficiency" in r and "solve_mean_ms" in r["efficiency"]:
            tk = r["config"]["top_k"]
            solve_by_topk[tk].append(r["efficiency"]["solve_mean_ms"])

    fig, ax = plt.subplots(figsize=(8, 6))
    topk_means = [np.mean(solve_by_topk[tk]) if solve_by_topk[tk] else np.nan for tk in topk_values]
    topk_stds = [np.std(solve_by_topk[tk]) if solve_by_topk[tk] else np.nan for tk in topk_values]
    ax.errorbar(topk_values, topk_means, yerr=topk_stds, marker='o', capsize=5, capthick=2)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Mean Solve Time (ms)")
    ax.set_title("Top-K Speedup Analysis")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "topk_speedup.png", dpi=200)
    plt.close()

    # 4. Jacobian Density Impact
    jac_densities = [r.get("structural", {}).get("jacobian_density", np.nan) for r in valid_results]
    jac_eval_times = [r.get("efficiency", {}).get("jac_eval_ms", np.nan) for r in valid_results]
    
    valid_pairs = [(d, t) for d, t in zip(jac_densities, jac_eval_times) if not (np.isnan(d) or np.isnan(t))]
    if valid_pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        densities, times = zip(*valid_pairs)
        ax.scatter(densities, times, alpha=0.6, s=100)
        ax.set_xlabel("Jacobian Density (%)")
        ax.set_ylabel("Jacobian Evaluation Time (ms)")
        ax.set_title("Jacobian Density vs Evaluation Cost")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "jacobian_density_scatter.png", dpi=200)
        plt.close()

    print(f"[OK] Generated visualizations in {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, default=str((_REPO_ROOT / "s2gpt_pinn" / "phase3_ablation_results").resolve()))
    p.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers (max 4 recommended)")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_slow_configs", action="store_true", help="Skip configurations that exceed 2000ms mean solve time")
    p.add_argument("--max_configs", type=int, default=None, help="Limit number of configurations to process (for testing)")
    p.add_argument("--show_progress", action="store_true", help="Show tqdm progress bars")
    args = p.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Generate all configurations
    all_configs = generate_configurations()
    print(f"[INFO] Generated {len(all_configs)} configurations")

    # Limit if requested
    if args.max_configs:
        all_configs = all_configs[:args.max_configs]
        print(f"[INFO] Limited to {args.max_configs} configurations for testing")

    # Check which are already complete
    completed = set()
    summary_json = out_root / "ablation_summary.json"
    if summary_json.exists():
        try:
            with summary_json.open() as f:
                existing = json.load(f)
                completed = {tuple(c["config"].values()) for c in existing.get("configurations", []) if "efficiency" in c and "performance" in c}
        except Exception:
            pass

    # Filter out completed
    remaining = [c for c in all_configs if tuple(c.to_dict().values()) not in completed]
    print(f"[INFO] {len(completed)} already complete, {len(remaining)} remaining")

    if not remaining:
        print("[INFO] All configurations already complete!")
        return

    # Process configurations
    n_workers = min(args.n_workers, 4, len(remaining))  # Cap at 4 to avoid memory overload
    print(f"[INFO] Processing {len(remaining)} configurations with {n_workers} workers")

    task_args = [(c, out_root, args.horizon, args.seed) for c in remaining]

    results: List[Dict[str, Any]] = []
    errors: List[Tuple[ArchitectureConfig, str]] = []

    if n_workers == 1:
        # Sequential processing with progress bar
        iterator = tqdm(task_args, desc="Configurations") if (args.show_progress and tqdm) else task_args
        for task_arg in iterator:
            config, metrics, error = process_configuration(task_arg)
            if error:
                errors.append((config, error))
                print(f"[ERROR] {config}: {error}")
            else:
                results.append(metrics)
                # Skip slow configs if requested
                if args.skip_slow_configs and metrics.get("efficiency", {}).get("solve_mean_ms", 0) > 2000:
                    print(f"[SKIP] {config}: solve time {metrics['efficiency']['solve_mean_ms']:.1f}ms > 2000ms")
    else:
        # Parallel processing
        with multiprocessing.Pool(processes=n_workers) as pool:
            iterator = tqdm(pool.imap(process_configuration, task_args), total=len(task_args), desc="Configurations") if (args.show_progress and tqdm) else pool.imap(process_configuration, task_args)
            for config, metrics, error in iterator:
                if error:
                    errors.append((config, error))
                    print(f"[ERROR] {config}: {error}")
                else:
                    results.append(metrics)
                    # Skip slow configs if requested
                    if args.skip_slow_configs and metrics.get("efficiency", {}).get("solve_mean_ms", 0) > 2000:
                        print(f"[SKIP] {config}: solve time {metrics['efficiency']['solve_mean_ms']:.1f}ms > 2000ms")

    # Load existing results if any
    existing_results = []
    if summary_json.exists():
        try:
            with summary_json.open() as f:
                existing_data = json.load(f)
                existing_results = existing_data.get("configurations", [])
        except Exception:
            pass

    # Merge results
    all_results = existing_results + results

    # Create summary
    summary = {
        "meta": {
            "script": "s2gpt_pinn/run_architecture_ablation.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware": {
                "cpu": platform.processor() or "unknown",
                "ram_gb": psutil.virtual_memory().total / (1024**3),
            },
            "parameter_grid": PARAM_GRID,
            "total_configurations": len(all_configs),
            "completed": len(all_results),
            "errors": len(errors),
        },
        "configurations": all_results,
        "errors": [{"config": c.to_dict(), "error": e} for c, e in errors],
    }

    # Save summary
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved summary to {summary_json}")

    # Generate visualizations
    if results:
        generate_visualizations(all_configs, all_results, out_root)

    print(f"[DONE] Processed {len(results)} configurations, {len(errors)} errors")


if __name__ == "__main__":
    main()

