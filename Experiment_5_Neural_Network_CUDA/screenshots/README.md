# Experiment 5 Screenshots Guide

Capture the following genuine terminal/editor screens from the CUDA development environment (NVIDIA GPU Workstation / Visual Studio Developer Prompt / Google Colab / Cloud GPU):

| Filename | Purpose / Description | Expected Screen Content |
|:---|:---|:---|
| `01_source_code.png` | Source code in editor | VS Code or Notepad showing `neural_network.cu` Phase 1 & Phase 2 kernels |
| `02_environment_check.png` | Hardware & compiler check | Running `nvidia-smi` and `nvcc --version` |
| `03_compilation.png` | Successful compilation | Running `nvcc -O2 neural_network.cu -o neural_network.exe` |
| `04_kernel_validation.png` | Kernel unit tests | Top part of execution showing Matrix multiplication + bias: PASSED, ReLU: PASSED, Sigmoid: PASSED |
| `05_execution_and_speedup.png` | Full benchmark output | Bottom part of execution showing CPU compute time, GPU kernel-only time, end-to-end time, and speedup ratios |
