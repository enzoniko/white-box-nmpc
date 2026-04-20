# White-Box Neural Ensemble for Vehicular Plasticity
### Quantifying the Efficiency Cost of Symbolic Auditability in Adaptive NMPC

[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CasADi 3.6](https://img.shields.io/badge/CasADi-3.6-orange.svg)](https://web.casadi.org/)

This repository contains the complete implementation and experimental suite for the paper **"White-Box Neural Ensemble for Vehicular Plasticity: Quantifying the Efficiency Cost of Symbolic Auditability in Adaptive NMPC"**. 

The architecture addresses **Vehicular Plasticity**—the ability to adapt to varying operating regimes without retraining—by arbitrating among frozen, regime-specific neural specialists using a **Governor QP**. To ensure maximal transparency for safety-critical deployment (ISO 26262), the ensemble dynamics are maintained as a fully traversable symbolic graph in CasADi.

---

## 🚀 Key Features

*   **White-Box NMPC**: Retains an explicit symbolic representation (CasADi SX) at runtime, allowing direct inspection of the derivative graph and Jacobians.
*   **Vehicular Plasticity**: Handles abrupt regime shifts (friction changes) and fleet heterogeneity (mass/drag variations) without online backpropagation.
*   **Governor QP**: A convex optimization-based blending mechanism that ensures the resulting dynamics remain within the verified convex hull of the model library.
*   **Physics-Informed Specialists**: Sparse ensemble of $N=8$ specialists trained using a hybrid Adam/L-BFGS protocol to minimize epistemic error.

---

## 📂 Repository Structure

*   `s2gpt_pinn/`: **Core codebase** for the White-Box Ensemble and NMPC solver.
    *   `nmpc.py`: Implementation of the NMPC with the Governor weight-blending logic.
    *   `specialist.py`: Neural specialist definitions and loading.
    *   `physics_casadi.py`: Symbolic physics utilities for CasADi integration.
*   `hss-codebase/`: **Hierarchical Specialist Selection** logic and greedy library construction.
*   `bayesrace/`: (Modified) **Racing Simulator** and reference trajectory generation (Oracle vs. Naive).
*   `deep-dynamics/`: (Modified) **Vehicle Physics Models** providing the ground-truth generator.

---

## 📊 Experiments & Reproducibility

### Phase I: Computational Benchmarks
Quantifies the "cost to compute zeros"—the structural transparency penalty of symbolic graph traversal.
```bash
python s2gpt_pinn/paper_experiments.py --mode efficiency
```

### Phase II: Functional Plasticity Validation
Validates closed-loop tracking under abrupt friction shifts ($\mu \in [0.5, 1.25]$) and compound heterogeneity.
```bash
python s2gpt_pinn/paper_orca_closedloop.py --scenario severe_friction
```

---

## 📈 Benchmarking Results

Our experiments establish an empirical baseline for the efficiency price of strict white-box implementation:
*   **Symbolic Latency**: Symbolic graph maintenance increases solver latency by **72–102×** versus compiled parametric physics models.
*   **Adaptation Speed**: The Governor provides rapid adaptation (~7.3 ms), significantly outperforming JIT recompilation (~752 ms).
*   **Tracking Fidelity**: The hybrid-trained PINN specialists close the gap to the ideal ODE performance ceiling, mitigating meter-scale drift to <0.1m.

---

## 📝 Citation

If you use this codebase or architecture in your research, please cite:

```bibtex
@article{spotorno2026whitebox,
  title={White-Box Neural Ensemble for Vehicular Plasticity: Quantifying the Efficiency Cost of Symbolic Auditability in Adaptive NMPC},
  author={Spotorno, Enzo Nicol{\'a}s and Wagner, Matheus and Fr{\"o}hlich, Ant{\^o}nio Augusto},
  journal={arXiv preprint arXiv:2602.01516},
  year={2026}
}
```

---

## 📧 Contact
For questions regarding the implementation or the HYDRA architecture, contact:
**Enzo Nicolás Spotorno** - [enzoniko@lisha.ufsc.br](mailto:enzoniko@lisha.ufsc.br)
**LISHA Lab** - Federal University of Santa Catarina (UFSC)