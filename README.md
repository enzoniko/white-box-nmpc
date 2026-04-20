# HYDRA Project: Multi-Specialist Vehicle Dynamics Architecture

**Final Implementation Report**  
**Author:** Enzo Spotorno  
**Date:** December 2025

---

## Executive Summary

This repository contains the complete implementation, training infrastructure, and validation framework for the **HYDRA (Hybrid Dynamics Representation Architecture)** project. HYDRA tests the hypothesis that diverse vehicle fleet dynamics can be efficiently represented by blending a small number of "specialist" neural network models.

**Key Results:**
- ✅ 91 vehicle variants simulated across 9 vehicle categories
- ✅ 91 physics-informed neural network (PINN) specialists trained (100% success rate)
- ✅ Kolmogorov N-Width hypothesis validated on predicted states (1 SVD component captures 99% energy)
- ✅ Convex blending achieves 1.07% reconstruction error on predicted trajectories
- ⚠️ Conservative data generation limits full hypothesis stress-testing

---

## Repository Structure

```
AES/
├── README.md                                    # This file
├── KOLMOGOROV_N_WIDTH_COMPARATIVE_ANALYSIS.md  # Detailed hypothesis testing results
│
├── HYDRA/                                       # Deep Dynamics model training & analysis
│   └── deep-dynamics/                           # Physics-constrained neural network framework
│       ├── deep_dynamics/                       # Core model code
│       │   ├── cfgs/model/                      # 91 vehicle-specific YAML configs
│       │   ├── data/                            # Training data (CSV → NPZ)
│       │   ├── model/                           # PINN architecture (train.py, evaluate.py)
│       │   └── output/                          # Trained model weights
│       ├── tools/                               # Analysis & automation scripts
│       │   ├── train_all_deep_dynamics.py       # Batch training automation
│       │   ├── analyze_trained_models.py        # Model evaluation & metrics
│       │   ├── kolmogorov_n_width_states.py     # State trajectory hypothesis test
│       │   ├── kolmogorov_n_width_coefficients.py # Coefficient hypothesis test
│       │   ├── greedy_specialist_selection_*.py # Specialist basis selection
│       │   └── *.md                             # Analysis summaries
│       ├── OPENCAR_TO_DEEPDYNAMICS_WORKFLOW.md  # Data pipeline documentation
│       └── README.md                            # Deep Dynamics framework docs
│
├── Open-Car-Dynamics/                           # Vehicle simulation & data generation
│   └── OpenCarDynamics/enzo_solution/hydra/     # HYDRA simulation infrastructure
│       ├── configs/                             # 9 base vehicle DNA configurations
│       ├── variant_configs/                     # 91 vehicle variant configurations
│       ├── experiment_results/                  # Simulation output data (91 CSVs)
│       ├── run_dna_experiments.py               # Simulation automation
│       ├── create_dna_variants.py               # Vehicle variant generator
│       ├── batch_convert_opencar_to_deepdynamics.py # Data format converter
│       ├── kolmogorov_n_width_hypothesis_test.py    # Ground truth analysis
│       ├── greedy_specialist_selection_test.py      # Ground truth specialist selection
│       ├── generate_experiment_visualizations.py    # Visualization generator
│       └── README.md                            # Hydra simulation docs
│
├── HYDRA_visualizations/                        # All generated figures & GIFs
│   ├── README.md                                # Figure inventory & semantics
│   ├── kolmogorov_n_width_*.png                 # SVD analysis plots
│   ├── greedy_specialist_selection_*.png        # Specialist selection plots
│   ├── static_*.png                             # Fleet dynamics overview
│   ├── animated_*.gif                           # Time-evolution animations
│   └── analysis_plots/                          # Model-specific plots
│
└── perception/                                  # (Separate project - not HYDRA related)
```

---

## Pipeline Overview

### Phase 1: Data Generation (Open-Car-Dynamics)

**Location:** `Open-Car-Dynamics/OpenCarDynamics/enzo_solution/hydra/`

1. **Create Vehicle Variants** (`create_dna_variants.py`)
   - Input: 9 base vehicle DNA configurations
   - Output: 91 physically-plausible variants (10 per base + 1 additional)
   - Variations: Mass (±15%), inertia, drag, tire parameters (±10-20%)

2. **Run Simulations** (`run_dna_experiments.py`)
   - Uses Open-Car-Dynamics double-track vehicle model
   - Open-loop control inputs (step steer, acceleration, braking)
   - ~100s simulation per vehicle at 50Hz (5,141 timesteps)
   - Output: 192-column CSV per vehicle

3. **Convert to Deep Dynamics Format** (`batch_convert_opencar_to_deepdynamics.py`)
   - Maps 192 columns → 17 columns (core dynamics)
   - Extracts vehicle parameters (mass, wheelbase, tire coefficients)
   - Generates YAML configs with GuardLayer bounds

### Phase 2: Model Training (Deep Dynamics)

**Location:** `HYDRA/deep-dynamics/`

1. **Parse Training Data** (`tools/csv_parser.py`)
   - Input: 17-column CSV
   - Output: NPZ with windowed features (horizon=5)

2. **Train Specialists** (`tools/train_all_deep_dynamics.py`)
   - Architecture: GRU backbone + Physics Guard layer
   - Output: 3 predicted states (vx, vy, ω) + 17 physical coefficients
   - 91 models trained (100% success rate)

3. **Evaluate Models** (`tools/analyze_trained_models.py`)
   - RMSE computation per vehicle type
   - Parameter distribution analysis
   - State evolution plots

### Phase 3: Hypothesis Testing

**Ground Truth Analysis** (`Open-Car-Dynamics/.../hydra/`):
- `kolmogorov_n_width_hypothesis_test.py` → 39 SVD components for 99% energy
- `greedy_specialist_selection_test.py` → 4.63× reduction with 20 specialists

**Predicted States Analysis** (`HYDRA/deep-dynamics/tools/`):
- `kolmogorov_n_width_states.py` → **1 SVD component** for 99% energy
- `greedy_specialist_selection_states.py` → **36.27× reduction**

**Predicted Coefficients Analysis**:
- `kolmogorov_n_width_coefficients.py` → 1 SVD component (trivial clustering)
- Coefficient variance < 10⁻⁵ across windows (static parameters)

---

## Key Results Summary

| Metric | Ground Truth | Predicted States | Predicted Coefficients |
|--------|--------------|------------------|------------------------|
| **SVD Components (99%)** | 39 | **1** | 1 |
| **Simplex Blend Error** | 70.91% | **1.07%** | 155% |
| **Autoencoder Latent Dim** | 10 | **2** | 3 |
| **Greedy Reduction Factor** | 4.63× | **36.27×** | 3.08×10²⁶× |
| **Interpretation** | Complex, diverse | Low-dimensional manifold | Trivial clustering |

---

## Quick Start

### 1. Generate Visualizations
```bash
cd Open-Car-Dynamics/OpenCarDynamics/enzo_solution/hydra/
python3 generate_experiment_visualizations.py
```

### 2. Run Hypothesis Test on Ground Truth
```bash
cd Open-Car-Dynamics/OpenCarDynamics/enzo_solution/hydra/
python3 kolmogorov_n_width_hypothesis_test.py
python3 greedy_specialist_selection_test.py
```

### 3. Run Hypothesis Test on Predictions
```bash
cd HYDRA/deep-dynamics/tools/
conda activate deep_dynamics
python3 kolmogorov_n_width_states.py
python3 greedy_specialist_selection_states.py
```

### 4. Analyze Trained Models
```bash
cd HYDRA/deep-dynamics/tools/
conda activate deep_dynamics
python3 analyze_trained_models.py --deep-dynamics-path ../
```

---

## Vehicle Fleet Configuration

| Category | Base Config | Variants | Mass Range | Wheelbase |
|----------|-------------|----------|------------|-----------|
| Compact Hatchback | `config_compact_hatchback.yml` | 10 | 1,150-1,550 kg | 2.53m |
| Mid-Size Sedan | `config_mid_size_sedan.yml` | 10 | 1,320-1,780 kg | 2.80m |
| Sports Coupe | `config_sports_coupe.yml` | 10 | 1,230-1,670 kg | 2.51m |
| Compact SUV | `config_compact_suv.yml` | 10 | 1,400-1,900 kg | 2.70m |
| Full-Size SUV | `config_full_size_suv.yml` | 10 | 2,125-2,875 kg | 3.13m |
| Minivan | `config_minivan.yml` | 10 | 1,785-2,415 kg | 3.13m |
| Pickup Truck (Unladen) | `config_pickup_truck_unladen.yml` | 10 | 1,870-2,530 kg | 3.15m |
| Pickup Truck (Laden) | `config_pickup_truck_laden.yml` | 10 | 2,380-3,220 kg | 3.15m |
| Electric Vehicle Sedan | `config_electric_vehicle_sedan.yml` | 10 | 1,570-2,130 kg | 2.80m |
| **Total** | **9 base** | **91 variants** | | |

---

## Model Performance

All 91 Deep Dynamics specialists achieved:
- **Longitudinal velocity (vx):** RMSE 0.064 - 0.130 m/s
- **Lateral velocity (vy):** RMSE 0.0006 - 0.026 m/s  
- **Yaw rate (ω):** RMSE 0.0000 - 0.0013 rad/s

Best performer: Full-Size SUV variants  
Most variable: Electric Vehicle Sedan, Mid-Size Sedan

---

## Limitations & Future Work

### Current Limitations
1. **Conservative Data Generation**: Tire parameters varied only ±10-20% to ensure numerical stability
2. **Limited Excitation**: Open-loop control kept vehicles in linear tire region
3. **Temporal Downsampling**: Predictions sampled at 5× lower frequency than ground truth
4. **Static Coefficients**: Models learned constant parameters per vehicle (variance < 10⁻⁵)

### Recommended Next Steps
1. **High-Excitation Data**: Generate datasets with aggressive maneuvers (drifting, limit braking)
2. **Time-Varying Parameters**: Modify architecture to encourage dynamic coefficient adaptation
3. **Online Gating Mechanism**: Implement real-time specialist selection/blending
4. **Closed-Loop Validation**: Test blended specialists in MPC controllers

---

## Documentation Resources

| Document | Location | Content |
|----------|----------|---------|
| **Comparative Analysis** | `KOLMOGOROV_N_WIDTH_COMPARATIVE_ANALYSIS.md` | Full hypothesis testing results |
| **Deep Dynamics Workflow** | `HYDRA/deep-dynamics/OPENCAR_TO_DEEPDYNAMICS_WORKFLOW.md` | Data pipeline documentation |
| **Model Analysis Summary** | `HYDRA/deep-dynamics/tools/ANALYSIS_SUMMARY.md` | 91-model evaluation results |
| **Coefficient Analysis** | `HYDRA/deep-dynamics/tools/COEFFICIENT_ANALYSIS_README.md` | Coefficient hypothesis testing |
| **State vs Coefficient** | `HYDRA/deep-dynamics/tools/STATE_VS_COEFFICIENT_ANALYSIS_SUMMARY.md` | Comparison of analysis approaches |
| **Visualization Guide** | `HYDRA_visualizations/README.md` | Figure inventory & semantics |
| **Hydra Simulation** | `Open-Car-Dynamics/.../hydra/README.md` | Simulation infrastructure |
| **Deep Dynamics Framework** | `HYDRA/deep-dynamics/README.md` | Original framework documentation |

---

## Key Scripts Reference

### Data Generation & Conversion
| Script | Location | Purpose |
|--------|----------|---------|
| `create_dna_variants.py` | `Open-Car-Dynamics/.../hydra/` | Generate 91 vehicle variants |
| `run_dna_experiments.py` | `Open-Car-Dynamics/.../hydra/` | Run simulations in Docker |
| `batch_convert_opencar_to_deepdynamics.py` | `Open-Car-Dynamics/.../hydra/` | Convert CSV→Deep Dynamics format |

### Model Training & Analysis
| Script | Location | Purpose |
|--------|----------|---------|
| `train_all_deep_dynamics.py` | `HYDRA/deep-dynamics/tools/` | Batch training automation |
| `analyze_trained_models.py` | `HYDRA/deep-dynamics/tools/` | Model evaluation & plots |
| `csv_parser.py` | `HYDRA/deep-dynamics/deep_dynamics/tools/` | Prepare training data |

### Hypothesis Testing
| Script | Location | Purpose |
|--------|----------|---------|
| `kolmogorov_n_width_hypothesis_test.py` | `Open-Car-Dynamics/.../hydra/` | Ground truth SVD analysis |
| `kolmogorov_n_width_states.py` | `HYDRA/deep-dynamics/tools/` | Predicted states SVD |
| `kolmogorov_n_width_coefficients.py` | `HYDRA/deep-dynamics/tools/` | Predicted coefficients SVD |
| `greedy_specialist_selection_*.py` | Both locations | Specialist basis selection |

### Visualization
| Script | Location | Purpose |
|--------|----------|---------|
| `generate_experiment_visualizations.py` | `Open-Car-Dynamics/.../hydra/` | Static figures + animated GIFs |
| `analyze_experiments_simple.py` | `Open-Car-Dynamics/.../hydra/` | Interactive experiment viewer |

---

## Citation

This project builds upon the Deep Dynamics framework:

```bibtex
@ARTICLE{10499707,
  author={Chrosniak, John and Ning, Jingyun and Behl, Madhur},
  journal={IEEE Robotics and Automation Letters}, 
  title={Deep Dynamics: Vehicle Dynamics Modeling With a Physics-Constrained 
         Neural Network for Autonomous Racing}, 
  year={2024},
  volume={9},
  number={6},
  pages={5292-5297},
  doi={10.1109/LRA.2024.3388847}
}
```

---

## Resources Used in This README

This documentation was compiled from the following project resources:

1. **`KOLMOGOROV_N_WIDTH_COMPARATIVE_ANALYSIS.md`** - Numerical results and methodology
2. **`HYDRA/deep-dynamics/README.md`** - Deep Dynamics framework documentation
3. **`HYDRA/deep-dynamics/OPENCAR_TO_DEEPDYNAMICS_WORKFLOW.md`** - Data pipeline details
4. **`HYDRA/deep-dynamics/tools/ANALYSIS_SUMMARY.md`** - Model performance metrics
5. **`HYDRA/deep-dynamics/tools/STATE_VS_COEFFICIENT_ANALYSIS_SUMMARY.md`** - Analysis comparison
6. **`HYDRA/deep-dynamics/tools/COEFFICIENT_ANALYSIS_README.md`** - Coefficient testing docs
7. **`Open-Car-Dynamics/OpenCarDynamics/enzo_solution/hydra/README.md`** - Simulation setup
8. **`Open-Car-Dynamics/OpenCarDynamics/enzo_solution/hydra/control_scripts/README.md`** - Control protocol
9. **`Open-Car-Dynamics/OpenCarDynamics/enzo_solution/README.md`** - Exercise structure
10. **`HYDRA_visualizations/README.md`** - Figure inventory

---

*Last updated: December 2025*

