# Afternoon Block: Scalable QML Models and Simulation

## Overview

This block works through three QML model families in sequence, each of which stresses the simulation stack differently: a quantum sequence model that generates classical weights, a quantum-inspired architecture that replaces parameter-heavy layers, and a quantum kernel method that scales through batched tensor-network contraction. The unifying question is practical rather than conceptual — what does it actually take to train and validate each of these at a useful size?

By the end of this block, participants will understand how a quantum circuit can generate the fast weights of a classical sequence model, how tensor-network reformulations of Kolmogorov-Arnold layers cut parameter counts in LLM blocks, and how cuTensorNet contraction scales a quantum kernel matrix well past what current hardware can produce.

<br>

## Session Breakdown

| Topic | Duration | Presenter | Institution |
| :--- | :--- | :--- | :--- |
| Quantum Fast Weight Programmers | 20 min | Samuel Yen-Chi Chen | Wells Fargo |
| Quantum-Inspired Architectures for LLMs | 35 min | Jiun-Cheng Jiang | NVIDIA |
| Scalable Quantum-Enhanced Support Vector Machines | 30 min | Tai-Yue Li | NCHC |
| Q&A | 5 min | All | — |

<br>

## Topics Covered

### Quantum Fast Weight Programmers

- **Fast weights instead of recurrence:** A programmer network emits multiplicative updates to a slow network's weight matrix, carrying sequence context without an explicit recurrent hidden state.
- **The quantum programmer:** A compact parameterized quantum circuit plays the programmer role, so the parameter budget grows with circuit depth rather than with hidden-state width.
- **Advanced — Simulating the programmer at scale:** The programmer circuit is small enough for state-vector simulation, so evaluating it through cuQuantum and batching across the sequence keeps the classical training loop, rather than the circuit, on the critical path.

### Quantum-Inspired Architectures for Large Language Models

- **QKAN-LLM as an introductory example:** [Quantum-inspired Kolmogorov-Arnold Networks](https://arxiv.org/abs/2509.14026) as an LLM backbone, using data re-uploading activations for parameter reduction and function fitting.
- **Advanced — GPT-scale training and billion-parameter inference:** Train a GPT-scale (~100M parameter) model and demonstrate pretrained QKAN-LLM inference at billion-parameter scale, routing every activation through the `cutn` (cuTensorNet) solver path; without cuQuantum installed the same code path still runs as a local fallback, so those timings are functional rather than accelerated.

### Scalable Quantum-Enhanced Support Vector Machines

- **Quantum kernel methods briefing:** [cuQuantum-accelerated QSVM](https://iopscience.iop.org/article/10.1088/2632-2153/adb4ba) leverages quantum feature spaces for classical SVM, encoding classical data into expressive quantum representations via batched contraction acceleration.
- **Advanced — Multi-stream HPC optimization with cuTensor:** Capture the circuit topology once with `cudaq_einsum`, reuse it across every feature-vector pair by swapping only the parameterized rotation tensors, and drive multiple CUDA streams to keep the GPU saturated on large kernel matrices.

<br>

## Notebooks

| Notebook | Topic | Contributor | Institution |
| :--- | :--- | :--- | :--- |
| `02_qfwp.ipynb` | Quantum Fast Weight Programmers | Samuel Yen-Chi Chen | Wells Fargo |
| `03_qkan_basics.ipynb` | Quantum-Inspired Kolmogorov-Arnold Networks | Jiun-Cheng Jiang | NVIDIA |
| `04_cutn-qsvm.ipynb` | Quantum-Enhanced Support Vector Machine | Tai-Yue Li | NCHC |
