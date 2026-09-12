# Scalable Validation and Optimized Simulation for Quantum Machine Learning

> IEEE Quantum Week 2026 (QCE26) tutorial `TUT::QML::QSIMU::519`. This `qce26` branch holds the QCE26 edition of the materials.

## Abstract

Quantum machine learning (QML) is progressing toward larger and more complex models, characterized by increasing qubit counts, deeper circuits, and tighter integration with classical machine learning workflows. In this context, efficient quantum circuit simulation and rigorous validation are becoming central to the development and assessment of large-scale QML models — the bottleneck is rarely the algorithm itself, but whether a model can be simulated fast enough to train on and validated rigorously enough to trust. This tutorial examines how GPU-optimized simulation enables scalable QML workflows, covering hybrid model architectures, quantum-inspired methods, and the layered GPU software stack behind them: [**CUDA-Q**](https://developer.nvidia.com/cuda-q) for expressing quantum kernels, PyTorch-compatible integration, [**cuQuantum/cuTensorNet**](https://developer.nvidia.com/cuquantum-sdk) for GPU-accelerated tensor-network execution, and [**cuTensor**](https://developer.nvidia.com/cutensor) for advanced contraction optimization.

The material builds a single continuous workflow across five notebooks. It begins with the programming model and GPU environment, then works through four families of QML model, each of which stresses the simulation stack in a different way:

- **Variational models and learned optimization:** A Transformer meta-optimizer that predicts QAOA parameter updates and generalizes across unseen problem instances, trained end-to-end with CUDA-Q and PyTorch — the case where a circuit is evaluated many thousands of times inside a classical training loop.
- **Quantum sequence models:** Quantum Fast Weight Programmers, where a compact parameterized circuit reprograms the weights of a classical network, giving temporal modeling without an explicit recurrent hidden state.
- **Quantum-inspired architectures:** Kolmogorov-Arnold Network blocks whose activations are evaluated as tensor networks, cutting parameter counts in LLM feed-forward layers from function fitting up to GPT scale.
- **Quantum kernel methods:** A support vector machine whose kernel matrix is produced by a quantum feature map, captured once as a tensor network and then contracted in batch across every pair of training points.

The through-line is the same discipline in each case: express the model, choose the simulation backend that matches its structure, measure honestly, and scale — so that large-scale quantum algorithms can be validated, developed, and optimized prior to deployment on QPUs. All materials are provided as reusable notebook examples, so participants leave with practical tools and workflows they can extend in their own QML research and development.

<br>

## Target Audience

This tutorial is intended for practitioners and researchers interested in quantum machine learning, efficient quantum circuit simulation, and scalable model design. Prior familiarity with PyTorch and quantum machine learning is helpful, but not required. It is especially well suited for those seeking a practical introduction to QML validation and to scaling quantum-inspired models on modern classical accelerator platforms.

<br>

## Tech Stack & Environment

Each participant is provided with a dedicated **NVIDIA Brev** instance:

- **GPU:** NVIDIA A100 (40GB)
- **Software:** CUDA-Q, cuQuantum SDK (cuTensorNet), cuTensor, PyTorch
- **Interface:** Jupyter Lab / Notebooks

The stack is layered, and the tutorial moves down through it: CUDA-Q expresses the quantum kernel, cuQuantum selects the simulation backend — state vector or tensor network — and cuTensor executes the contractions underneath.

<br>

## 📅 Tutorial Outline (3 Hours)

The tutorial is delivered as two 90-minute blocks. The morning block establishes the programming model, the GPU environment, and the first complete hybrid workflow; the afternoon block scales three further model families and the simulation strategy each one demands.

### Morning Block — QML Foundations and Hybrid Workflows (10:00 – 11:30 AM EDT)

- **Focus:** Establishing the CUDA-Q programming model, the GPU environment, and an end-to-end hybrid QML training loop.
- **Key Topics:**
    - Why simulation throughput and validation govern which QML experiments are feasible at all, and how CUDA-Q and cuQuantum fit together.
    - GPU environment setup on NVIDIA Brev — no local installation or dependency conflicts.
    - Programming quantum kernels: qubits, gates, circuits, measurement, and noise modeling on the GPU `nvidia` target.
    - **Learned Optimization for Variational Models:** Meta-learning an optimizer for QAOA that generalizes across problem instances, with CUDA-Q and PyTorch in a single training pipeline.

### Afternoon Block — Scalable QML Models and Simulation (1:00 – 2:30 PM EDT)

- **Focus:** Three QML model families at scale, and the simulation technique each one requires.
- **Key Topics:**
    - **Quantum Fast Weight Programmers (QFWP):** A parameterized quantum circuit that reprograms the weights of a classical network, giving sequence modeling without explicit recurrence.
    - **Quantum-Inspired Kolmogorov-Arnold Networks (QKAN-LLM):** Data re-uploading activations as parameter-efficient replacements for MLP blocks, from function fitting through GPT-2-scale transformer blocks on the cuTensorNet solver path.
    - **Quantum-Enhanced SVM:** A classical SVM with a quantum feature-map kernel, captured once as a tensor network and contracted in batch via cuTensorNet, with a multi-stream cuTensor backend for further HPC scaling.

<br>

## 🚀 Getting Started

### 1. Launch Environment

If you are attending the live session, log in to your **NVIDIA Brev** dashboard to start your A100 instance.

### 2. Clone the Repository

Once inside your Jupyter environment, open a terminal and run:

```bash
git clone <repository-url>
cd QCE26_tutorials-scalable-qml
git checkout qce26
```
Note: The repository will be made publicly available before the event. All materials are pre-tested on the same NVIDIA Brev environment provided to participants.

### 3. Set Up the Environment

The Brev instance will be launched with a preconfigured Docker image with all dependencies already installed — just open Jupyter and start.

To reproduce the environment from scratch elsewhere, build the image in this repository:

```bash
docker build -t cudaq-qce26:cu132 .

# Serve Jupyter Lab on :8888 (this is what a Brev Launchable runs)
docker run --gpus all -p 8888:8888 -e JUPYTER_TOKEN=qce26 \
    -v "$(pwd):/workspace" cudaq-qce26:cu132

# Or drop into a shell / run a one-off command
docker run -it --gpus all -v "$(pwd):/workspace" cudaq-qce26:cu132 -c bash
docker run --gpus all cudaq-qce26:cu132 -c qce26-verify
```

`qce26-verify` reports the torch / CUDA-Q / cuQuantum / qkan versions and checks the GPU targets. The published image is [`jiunchengj81589/cudaq-qce26:cu132`](https://hub.docker.com/r/jiunchengj81589/cudaq-qce26).

It is based on `nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.15.1` and adds the CUDA 13.2 toolchain, since the published CUDA-Q images ship CUDA 13.0 and the tutorial's PyTorch and qkan builds target 13.2. If you already have a CUDA-Q container running, `bash run.sh` from the repository root performs the same setup in place.

The Dockerfile is self-contained — it inlines the dependency pins, so no other file from the repository is needed to build it. Python packages are installed with [`uv`](https://docs.astral.sh/uv/).

> **Note:** CUDA-Q and PyTorch's bundled `triton` each embed their own LLVM, and whichever loads second aborts the kernel with `Option 'debug-counter' registered more than once!`. The image installs an import hook that pulls `triton` in just ahead of the first `import cudaq`, so notebooks need no import discipline. `qce26-verify` re-tests this.

<br>

## Notebook Highlights

| Notebook | Block | Topic | Contributor | Institution |
| :--- | :--- | :--- | :--- | :--- |
| `00_cudaq_basics.ipynb` | Morning | QC Fundamentals and CUDA-Q Programming | Yun-Yuan Wang | NVIDIA |
| `01_transformer_qaoa.ipynb` | Morning | Transformer-Based QAOA Optimization | Kuan-Cheng Chen | Imperial College London |
| `02_qfwp.ipynb` | Afternoon | Quantum Fast Weight Programmers | Samuel Yen-Chi Chen | Wells Fargo |
| `03_qkan_basics.ipynb` | Afternoon | Quantum-Inspired Kolmogorov-Arnold Networks | Jiun-Cheng Jiang | NVIDIA |
| `04_cutn-qsvm.ipynb` | Afternoon | Quantum-Enhanced Support Vector Machine | Tai-Yue Li | NCHC |

<br>

## 📚 Resources

- [NVIDIA CUDA-Q Documentation](https://nvidia.github.io/cuda-quantum/latest/index.html)
- [cuQuantum SDK Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/index.html)
- [NVIDIA Brev](https://developer.nvidia.com/brev)
- [IEEE Quantum Week (QCE26)](https://qce.quantum.ieee.org/2026/)
