#!/usr/bin/env python3
"""
S²GPT-PINN Performance Benchmarking and Validation Suite

Compares S²GPT-PINN neural surrogate against physics-based baselines:
1. Inference latency (forward pass + Jacobian)
2. MPC solve time
3. Prediction accuracy vs physics oracle
4. Closed-loop tracking performance

Target metrics (per NMPC spec):
- MPC solve time: <2ms
- Prediction accuracy: <5% error vs physics
- 5-10x speedup over RK4 + symbolic differentiation
"""

import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .specialist import S2GPTConfig, S2GPTSpecialist, S2GPTEnsemble, verify_jacobian
from .training import VehicleParams, compute_physics_accelerations


@dataclass
class BenchmarkResults:
    """Results from benchmarking run."""
    # Latency (milliseconds)
    forward_mean_ms: float
    forward_std_ms: float
    forward_p95_ms: float
    jacobian_mean_ms: float
    jacobian_std_ms: float
    jacobian_p95_ms: float
    total_mean_ms: float
    
    # Accuracy
    accel_rmse: float
    accel_max_error: float
    accel_relative_error: float
    
    # Jacobian correctness
    jacobian_x_error: float
    jacobian_u_error: float
    
    # MPC solve time (if applicable)
    mpc_solve_mean_ms: Optional[float] = None
    mpc_solve_std_ms: Optional[float] = None
    mpc_solve_p95_ms: Optional[float] = None


def benchmark_inference_latency(
    ensemble: S2GPTEnsemble,
    n_iterations: int = 1000,
    warm_up: int = 100,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Benchmark neural dynamics inference latency.
    
    Tests single-point evaluation (typical MPC usage).
    
    Returns:
        Dict with timing statistics in milliseconds
    """
    if device is None:
        device = next(ensemble.parameters()).device
    
    # Generate random test points
    states = torch.randn(n_iterations, 3, device=device) * torch.tensor([10.0, 1.0, 0.5], device=device)
    states[:, 0] += 20.0  # Shift vx to positive
    controls = torch.randn(n_iterations, 2, device=device) * torch.tensor([0.3, 0.5], device=device)
    static = torch.tensor([[800.0, 1200.0]], device=device).expand(n_iterations, -1)
    
    # Warm up
    for i in range(warm_up):
        _ = ensemble(states[i:i+1], controls[i:i+1], static[i:i+1])
        _ = ensemble.jacobian_analytic(states[i], controls[i], static[i])
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Time forward pass (single point)
    times_forward = []
    for i in range(n_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = ensemble(states[i:i+1], controls[i:i+1], static[i:i+1])
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_forward.append(time.perf_counter() - start)
    
    # Time Jacobian computation
    times_jacobian = []
    for i in range(n_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = ensemble.jacobian_analytic(states[i], controls[i], static[i])
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_jacobian.append(time.perf_counter() - start)
    
    times_forward = np.array(times_forward) * 1000  # ms
    times_jacobian = np.array(times_jacobian) * 1000  # ms
    times_total = times_forward + times_jacobian
    
    return {
        "forward_mean_ms": np.mean(times_forward),
        "forward_std_ms": np.std(times_forward),
        "forward_p95_ms": np.percentile(times_forward, 95),
        "forward_max_ms": np.max(times_forward),
        "jacobian_mean_ms": np.mean(times_jacobian),
        "jacobian_std_ms": np.std(times_jacobian),
        "jacobian_p95_ms": np.percentile(times_jacobian, 95),
        "jacobian_max_ms": np.max(times_jacobian),
        "total_mean_ms": np.mean(times_total),
        "total_std_ms": np.std(times_total),
        "total_p95_ms": np.percentile(times_total, 95),
    }


def benchmark_accuracy(
    ensemble: S2GPTEnsemble,
    test_params: List[VehicleParams],
    n_test_points: int = 1000,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Benchmark prediction accuracy against physics oracle.
    
    Tests across multiple friction conditions.
    
    Returns:
        Dict with accuracy metrics
    """
    if device is None:
        device = next(ensemble.parameters()).device
    
    all_errors = []
    all_relative_errors = []
    
    for params in test_params:
        # Generate test data
        vx = torch.empty(n_test_points, device=device).uniform_(5.0, 40.0)
        vy = torch.empty(n_test_points, device=device).uniform_(-3.0, 3.0)
        omega = torch.empty(n_test_points, device=device).uniform_(-2.0, 2.0)
        delta = torch.empty(n_test_points, device=device).uniform_(-0.5, 0.5)
        throttle = torch.empty(n_test_points, device=device).uniform_(-0.5, 0.8)
        
        state = torch.stack([vx, vy, omega], dim=1)
        control = torch.stack([delta, throttle], dim=1)
        static = torch.tensor([[params.m, params.Iz]], device=device).expand(n_test_points, -1)
        
        # Physics target
        target = compute_physics_accelerations(state, control, static, params)
        
        # Neural prediction
        mu = params.pacejka_D_f
        with torch.no_grad():
            pred = ensemble(state, control, static, mu_current=mu)
        
        # Compute errors
        errors = (pred - target).abs()
        all_errors.append(errors.cpu().numpy())
        
        target_mag = target.abs().mean(dim=1, keepdim=True).clamp(min=0.1)
        rel_errors = (errors / target_mag).mean(dim=1)
        all_relative_errors.append(rel_errors.cpu().numpy())
    
    all_errors = np.concatenate(all_errors)
    all_relative_errors = np.concatenate(all_relative_errors)
    
    return {
        "accel_rmse": np.sqrt(np.mean(all_errors ** 2)),
        "accel_mean_error": np.mean(all_errors),
        "accel_max_error": np.max(all_errors),
        "accel_relative_error": np.mean(all_relative_errors),
        "accel_p95_error": np.percentile(all_errors, 95),
    }


def benchmark_jacobian_accuracy(
    ensemble: S2GPTEnsemble,
    n_test_points: int = 100,
    eps: float = 1e-5,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Verify analytic Jacobians against numerical differentiation.
    
    Returns:
        Dict with Jacobian accuracy metrics
    """
    if device is None:
        device = next(ensemble.parameters()).device
    
    errors_x = []
    errors_u = []
    
    for _ in range(n_test_points):
        state = torch.randn(3, device=device) * torch.tensor([10.0, 1.0, 0.5], device=device)
        state[0] += 20.0
        control = torch.randn(2, device=device) * torch.tensor([0.3, 0.5], device=device)
        static = torch.tensor([800.0, 1200.0], device=device)
        
        # Analytic Jacobian
        jac_x_analytic, jac_u_analytic = ensemble.jacobian_analytic(state, control, static)
        
        # Numerical Jacobian (state)
        jac_x_numerical = torch.zeros(3, 3, device=device)
        for i in range(3):
            state_plus = state.clone()
            state_plus[i] += eps
            f_plus = ensemble(state_plus.unsqueeze(0), control.unsqueeze(0), static.unsqueeze(0)).squeeze()
            
            state_minus = state.clone()
            state_minus[i] -= eps
            f_minus = ensemble(state_minus.unsqueeze(0), control.unsqueeze(0), static.unsqueeze(0)).squeeze()
            
            jac_x_numerical[:, i] = (f_plus - f_minus) / (2 * eps)
        
        # Numerical Jacobian (control)
        jac_u_numerical = torch.zeros(3, 2, device=device)
        for i in range(2):
            control_plus = control.clone()
            control_plus[i] += eps
            f_plus = ensemble(state.unsqueeze(0), control_plus.unsqueeze(0), static.unsqueeze(0)).squeeze()
            
            control_minus = control.clone()
            control_minus[i] -= eps
            f_minus = ensemble(state.unsqueeze(0), control_minus.unsqueeze(0), static.unsqueeze(0)).squeeze()
            
            jac_u_numerical[:, i] = (f_plus - f_minus) / (2 * eps)
        
        errors_x.append(torch.norm(jac_x_analytic - jac_x_numerical).item())
        errors_u.append(torch.norm(jac_u_analytic - jac_u_numerical).item())
    
    return {
        "jacobian_x_mean_error": np.mean(errors_x),
        "jacobian_x_max_error": np.max(errors_x),
        "jacobian_u_mean_error": np.mean(errors_u),
        "jacobian_u_max_error": np.max(errors_u),
    }


def benchmark_mpc_solve_time(
    ensemble: S2GPTEnsemble,
    n_solves: int = 100,
    horizon: int = 20,
    dt: float = 0.02
) -> Dict[str, float]:
    """
    Benchmark MPC solve time using S²GPT-PINN dynamics.
    
    Returns:
        Dict with MPC solve time statistics
    """
    from .nmpc import S2GPTNMPCSimple, S2GPTMPCConfig
    
    config = S2GPTMPCConfig(horizon=horizon, dt=dt)
    mpc = S2GPTNMPCSimple(ensemble, config)
    
    # Random initial states and references
    x0s = np.random.randn(n_solves, 3) * np.array([5.0, 0.5, 0.2]) + np.array([20.0, 0.0, 0.0])
    vrefs = np.random.uniform(15.0, 30.0, n_solves)
    u_prev = np.array([0.0, 0.3])
    
    # Warm up
    for i in range(min(10, n_solves)):
        _, _ = mpc.solve(x0s[i], vrefs[i], u_prev)
    
    # Timed solves
    times = []
    for i in range(n_solves):
        _, solve_time = mpc.solve(x0s[i], vrefs[i], u_prev)
        times.append(solve_time)
    
    times = np.array(times)
    
    return {
        "mpc_solve_mean_ms": np.mean(times),
        "mpc_solve_std_ms": np.std(times),
        "mpc_solve_p95_ms": np.percentile(times, 95),
        "mpc_solve_max_ms": np.max(times),
        "mpc_solve_min_ms": np.min(times),
    }


def run_full_benchmark(
    ensemble: S2GPTEnsemble,
    test_friction_values: List[float] = [0.4, 0.7, 1.0],
    device: torch.device = None
) -> BenchmarkResults:
    """
    Run complete benchmark suite.
    
    Returns:
        BenchmarkResults with all metrics
    """
    if device is None:
        device = next(ensemble.parameters()).device
    
    print("=" * 60)
    print("S²GPT-PINN Performance Benchmark")
    print("=" * 60)
    
    # 1. Inference latency
    print("\n1. Inference Latency...")
    latency = benchmark_inference_latency(ensemble, n_iterations=1000, device=device)
    print(f"   Forward pass: {latency['forward_mean_ms']:.3f} ± {latency['forward_std_ms']:.3f} ms")
    print(f"   Jacobian:     {latency['jacobian_mean_ms']:.3f} ± {latency['jacobian_std_ms']:.3f} ms")
    print(f"   Total:        {latency['total_mean_ms']:.3f} ± {latency['total_std_ms']:.3f} ms")
    
    # 2. Prediction accuracy
    print("\n2. Prediction Accuracy...")
    test_params = [VehicleParams(pacejka_D_f=mu, pacejka_D_r=mu) for mu in test_friction_values]
    accuracy = benchmark_accuracy(ensemble, test_params, n_test_points=1000, device=device)
    print(f"   RMSE:     {accuracy['accel_rmse']:.4f} m/s²")
    print(f"   Max err:  {accuracy['accel_max_error']:.4f} m/s²")
    print(f"   Rel err:  {accuracy['accel_relative_error']*100:.2f}%")
    
    # 3. Jacobian accuracy
    print("\n3. Jacobian Accuracy...")
    jacobian = benchmark_jacobian_accuracy(ensemble, n_test_points=100, device=device)
    print(f"   df/dx error: {jacobian['jacobian_x_mean_error']:.2e}")
    print(f"   df/du error: {jacobian['jacobian_u_mean_error']:.2e}")
    
    # 4. MPC solve time
    print("\n4. MPC Solve Time...")
    try:
        mpc = benchmark_mpc_solve_time(ensemble, n_solves=50)
        print(f"   Mean:  {mpc['mpc_solve_mean_ms']:.2f} ms")
        print(f"   P95:   {mpc['mpc_solve_p95_ms']:.2f} ms")
        print(f"   Max:   {mpc['mpc_solve_max_ms']:.2f} ms")
    except Exception as e:
        print(f"   Skipped: {e}")
        mpc = {}
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    target_mpc_time = 2.0  # ms
    target_accuracy = 5.0  # % relative error
    
    mpc_time = mpc.get('mpc_solve_mean_ms', float('inf'))
    rel_error = accuracy['accel_relative_error'] * 100
    
    mpc_pass = "✓" if mpc_time < target_mpc_time else "✗"
    acc_pass = "✓" if rel_error < target_accuracy else "✗"
    
    print(f"MPC solve time:    {mpc_time:.2f} ms (target <{target_mpc_time} ms) [{mpc_pass}]")
    print(f"Prediction error:  {rel_error:.2f}% (target <{target_accuracy}%) [{acc_pass}]")
    
    return BenchmarkResults(
        forward_mean_ms=latency['forward_mean_ms'],
        forward_std_ms=latency['forward_std_ms'],
        forward_p95_ms=latency['forward_p95_ms'],
        jacobian_mean_ms=latency['jacobian_mean_ms'],
        jacobian_std_ms=latency['jacobian_std_ms'],
        jacobian_p95_ms=latency['jacobian_p95_ms'],
        total_mean_ms=latency['total_mean_ms'],
        accel_rmse=accuracy['accel_rmse'],
        accel_max_error=accuracy['accel_max_error'],
        accel_relative_error=accuracy['accel_relative_error'],
        jacobian_x_error=jacobian['jacobian_x_mean_error'],
        jacobian_u_error=jacobian['jacobian_u_mean_error'],
        mpc_solve_mean_ms=mpc.get('mpc_solve_mean_ms'),
        mpc_solve_std_ms=mpc.get('mpc_solve_std_ms'),
        mpc_solve_p95_ms=mpc.get('mpc_solve_p95_ms'),
    )


if __name__ == "__main__":
    print("Creating test ensemble for benchmarking...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create untrained ensemble for testing
    config = S2GPTConfig(hidden_dim=64, n_layers=3)
    specialists = [S2GPTSpecialist(config).to(device) for _ in range(4)]
    ensemble = S2GPTEnsemble(specialists, mu_centers=[0.3, 0.6, 0.9, 1.2]).to(device)
    
    # Run benchmark
    results = run_full_benchmark(ensemble, device=device)

