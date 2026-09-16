# Morning Block: QML Foundations and Hybrid Workflows

## Overview

This block builds the foundation for the whole tutorial: why simulation throughput and rigorous validation decide which QML experiments are feasible at all, how to bring up a GPU environment with no local installation, and how to program quantum kernels with CUDA-Q. Participants move from single-qubit states and gates through noise modeling to variational quantum algorithms.

The second half assembles the tutorial's first complete hybrid workflow: a Transformer trained as a meta-optimizer for QAOA, proposing parameter updates for unseen problem instances instead of running a classical optimizer from scratch each time. This is the regime where a circuit is evaluated many thousands of times inside a classical training loop, which is precisely what makes simulation throughput the limiting factor. The CUDA-Q kernels and GPU sampling introduced here are reused throughout the afternoon block.

<br>

## Session Breakdown

| Topic | Duration | Presenter | Institution |
| :--- | :--- | :--- | :--- |
| Welcome, QML Validation Motivation, and Stack Overview | 10 min | Yun-Yuan Wang | NVIDIA |
| GPU Environment Setup on NVIDIA Brev | 10 min | Ming-Kang Ho | NCHC |
| Quantum Kernel Programming with CUDA-Q | 30 min | Yun-Yuan Wang | NVIDIA |
| Transformer-Based Optimization for QAOA | 35 min | Kuan-Cheng (Louis) Chen | JIJ Inc. |
| Q&A | 5 min | All | — |

<br>

## Topics Covered

### Validating and Scaling QML on Classical Accelerators

- Where quantum circuit simulation sits in a practical QML workflow, and when state-vector, tensor-network, or analytic formulations are the right choice.
- The role of CUDA-Q and cuQuantum in developing and assessing large-scale QML models, and what each layer of the stack contributes.
- Roadmap for the day: what you will build, validate, and optimize across both blocks.

### GPU Environment Setup

- Each participant sets up a containerized GPU environment on a pre-launched **NVIDIA Brev** instance (dedicated A100 40GB), requiring no local installation or software dependency conflicts.

### Quantum Kernel Programming with CUDA-Q

The hands-on notebook walks through five progressive learning objectives:

1. **Quantum States and Visualization** — Represent and visualize qubit states on the Bloch sphere using CUDA-Q.
2. **Single-Qubit Quantum Programs** — Gate operations (X, H, U3), superposition, measurement, and custom gate registration.
3. **Multi-Qubit Programming with Entanglement** — Quantum registers, controlled gates, Bell state preparation, and noisy simulation.
4. **Nested Quantum Kernels** — Modular programming with reusable quantum subroutines for building complex circuits hierarchically.
5. **Advanced CUDA-Q Kernels** — Adjoint operations, mid-circuit conditional measurement, and variational quantum algorithms (VQA) with both built-in CUDA-Q optimizers and third-party optimizers (SciPy).

### Learned Optimization for Variational Models

- **Meta-learning for quantum optimization:** Train a Transformer-based optimizer to predict high-quality parameter updates for QAOA, generalizing across diverse problem instances without per-instance classical optimization overhead.
- **End-to-end training pipeline:** Seamlessly integrating CUDA-Q and PyTorch — from graph featurization through unrolled learned optimization to inference-time refinement with classical local search.
- **Advanced — Execution paths and honest timing:** The notebook exposes a CUDA-Q path wrapped in `torch.autograd.Function` with parameter-shift gradients, alongside a pure-PyTorch statevector fallback for portability; fallback runs should be read as functional checks, not as accelerated results.

<br>

## Notebooks

| Notebook | Topic | Contributor | Institution |
| :--- | :--- | :--- | :--- |
| `00_cudaq_basics.ipynb` | QC Fundamentals and CUDA-Q Programming | Yun-Yuan Wang | NVIDIA |
| `01_transformer_qaoa.ipynb` | Transformer-Based QAOA Optimization | Kuan-Cheng (Louis) Chen | JIJ Inc. |

<br>

## Prerequisites

- Working proficiency in Python and Jupyter notebooks.
- Basic linear algebra (vectors, matrices, tensor products).
- Prior familiarity with PyTorch and quantum machine learning is helpful for the second half, but not required.
- No prior quantum computing experience is assumed — this block covers everything from the ground up.
