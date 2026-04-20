"""
S²GPT-PINN: Sparse Specialist Neural Surrogate for NMPC

This package provides a complete neural surrogate system for replacing
physics-based vehicle dynamics in NMPC controllers.

Key Features:
- S2GPTSpecialist: Sparse neural network with PRE-COMPUTED ANALYTIC JACOBIANS
- S2GPTEnsemble: Weighted combination of specialists for adaptation
- Two calibration modes:
    MODE A (Static): RBF weights when physical params are known
    MODE B (Adaptive): Linear regression when params are uncertain
- CasADi callback integration for real-time MPC (<2ms solve time target)
- H-SS training pipeline with domain randomization

Architecture:
    f̂(x, u) = Σᵢ wᵢ · Ψᵢ(x, u, m, Iz)
    
Where:
    - Ψᵢ are sparse specialist networks (32-64 hidden units)
    - wᵢ are weights from RBF (Mode A) or linear regression (Mode B)
    - Static params (m, Iz) are conditioning inputs
    - Jacobians are computed analytically via chain rule: J = Σᵢ wᵢ · Jᵢ

Usage:
    from s2gpt_pinn import (
        S2GPTEnsemble, S2GPTDynamics, S2GPTMPCConfig,
        S2GPTCalibrationManager, VehicleParamsConfig
    )
    
    # Load ensemble and create calibration manager
    ensemble = load_library("./s2gpt_models", device)
    calibrator = S2GPTCalibrationManager(ensemble, specialist_params)
    
    # MODE A: Known parameters
    calibrator.update_from_friction(mu_f=0.8)
    
    # MODE B: Adaptive from observations
    calibrator.add_observation(state, control, static, observed_accel)
    calibrator.calibrate_from_observations()
    
    # MPC solve uses current weights automatically
    mpc = S2GPTNMPCSimple(ensemble, S2GPTMPCConfig())
    u_opt, solve_time = mpc.solve(x0, vref, u_prev)
"""

__version__ = "0.1.0"

# Core specialist models with analytic Jacobians
# (Internally H-SS; legacy S2GPT* aliases are provided for older scripts.)
from .specialist import (
    HSSConfig,
    HSSSpecialist,
    HSSEnsemble,
    S2GPTConfig,
    S2GPTSpecialist,
    S2GPTEnsemble,
    verify_jacobian,
)

# CasADi integration
from .casadi_callbacks import (
    CasadiExportConfig,
    export_specialist_to_casadi,
    export_ensemble_to_casadi,
    HSSDynamicsCallback,
)

# NMPC controllers
from .nmpc import (
    S2GPTNMPC,
    S2GPTNMPCSimple,
    S2GPTMPCConfig,
)

# Training pipeline
from .training import (
    TrainingConfig,
    VehicleParams,
    HSSLibraryBuilder,
    load_library,
    train_specialist,
    sample_candidate_params,
)

# Benchmarking
from .benchmark import (
    BenchmarkResults,
    run_full_benchmark,
    benchmark_inference_latency,
    benchmark_accuracy,
    benchmark_jacobian_accuracy,
    benchmark_mpc_solve_time,
)

# Calibration (MODE A: RBF, MODE B: Linear Regression)
from .calibration import (
    VehicleParamsConfig,
    SpecialistInfo,
    RBFWeightComputer,
    OnlineLinearRegressor,
    S2GPTCalibrationManager,
    generate_specialist_param_sets,
)

__all__ = [
    # Version
    "__version__",
    # Core models
    "HSSConfig",
    "HSSSpecialist",
    "HSSEnsemble",
    "S2GPTConfig",
    "S2GPTSpecialist",
    "S2GPTEnsemble",
    "verify_jacobian",
    # CasADi integration
    "CasadiExportConfig",
    "export_specialist_to_casadi",
    "export_ensemble_to_casadi",
    "HSSDynamicsCallback",
    # MPC
    "S2GPTNMPC",
    "S2GPTNMPCSimple",
    "S2GPTMPCConfig",
    # Training
    "TrainingConfig",
    "VehicleParams",
    "HSSLibraryBuilder",
    "load_library",
    "train_specialist",
    "sample_candidate_params",
    # Benchmarking
    "BenchmarkResults",
    "run_full_benchmark",
    "benchmark_inference_latency",
    "benchmark_accuracy",
    "benchmark_jacobian_accuracy",
    "benchmark_mpc_solve_time",
    # Calibration
    "VehicleParamsConfig",
    "SpecialistInfo",
    "RBFWeightComputer",
    "OnlineLinearRegressor",
    "S2GPTCalibrationManager",
    "generate_specialist_param_sets",
]

