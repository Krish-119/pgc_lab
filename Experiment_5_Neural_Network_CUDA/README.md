# Experiment 5: Neural Network Layers in CUDA

## 1. Experiment Title
**Implementation and Evaluation of Neural Network Forward-Pass Layers Using Custom CUDA Kernels**

---

## 2. Aim
To design, implement, and evaluate basic neural-network layers using custom CUDA kernels (matrix multiplication with bias, ReLU activation, and sigmoid activation). Execute the identical two-layer feed-forward inference pass on both the CPU and GPU, validate numerical correctness within tolerance, measure execution times (kernel-only and end-to-end), and calculate empirical GPU speedups.

---

## 3. Problem Statement
Deep learning models rely heavily on dense matrix operations and non-linear element-wise activations. Performing these forward propagation layers on a single-core CPU becomes a major latency bottleneck during real-time inference.

The objective is to:
1. Implement a two-layer feed-forward neural network forward pass:
   - **Layer 1**: $Z_1 = X \times W_1 + b_1$, followed by $A_1 = \text{ReLU}(Z_1)$
   - **Layer 2**: $Z_2 = A_1 \times W_2 + b_2$, followed by $Y = \text{Sigmoid}(Z_2)$
2. Implement custom CUDA kernels for each operation:
   - 2D grid matrix multiplication kernel with fused bias addition.
   - 1D grid element-wise ReLU activation kernel ($\max(0, x)$).
   - 1D grid element-wise Sigmoid activation kernel ($1 / (1 + e^{-x})$).
3. Validate numerical accuracy by verifying that individual kernel unit tests pass and that the maximum absolute difference between CPU and GPU outputs is strictly $< 10^{-4}$.
4. Quantify performance differences between CPU compute time, GPU kernel-only time, and GPU end-to-end time (including PCIe memory transfers).

---

## 4. Introduction
Neural networks process high-dimensional tensor transformations during inference. In standard multi-layer perceptrons (MLPs), each layer computes an affine transformation (matrix multiplication plus bias vector) followed by a non-linear activation function. 

While CPUs execute these loops sequentially or with modest SIMD vectorization, Graphics Processing Units (GPUs) contain thousands of streaming cores designed to execute millions of lightweight threads concurrently using the SIMT (Single Instruction, Multiple Threads) architecture. This experiment explores writing native CUDA kernels from scratch to perform GPU-accelerated inference without relying on external deep learning frameworks.

---

## 5. Theory

### 5.1 Network Architecture
The experiment evaluates an inference pass through a two-layer fully connected network:
- **Input Matrix $X$**: $\text{BATCH} \times \text{INPUT\_SIZE} = 256 \times 128$
- **Weight Matrix $W_1$**: $\text{INPUT\_SIZE} \times \text{HIDDEN\_SIZE} = 128 \times 64$
- **Bias Vector $b_1$**: $\text{HIDDEN\_SIZE} = 64$
- **Hidden Activation $A_1$**: $256 \times 64$
- **Weight Matrix $W_2$**: $\text{HIDDEN\_SIZE} \times \text{OUTPUT\_SIZE} = 64 \times 10$
- **Bias Vector $b_2$**: $\text{OUTPUT\_SIZE} = 10$
- **Final Output $Y$**: $256 \times 10$

### 5.2 Layer Formulations

#### 1. Matrix Multiplication with Bias
For matrix $A$ ($M \times K$) and matrix $B$ ($K \times N$), output matrix $C$ has dimensions $M \times N$:
$$C[\text{row}, \text{col}] = \sum_{k=0}^{K-1} A[\text{row}, k] \times B[k, \text{col}] + \text{bias}[\text{col}]$$
In CUDA, each thread is mapped to one output element $(\text{row}, \text{col})$. Threads outside matrix boundaries are guarded with boundary checks (`row < M && col < N`).

#### 2. ReLU Activation
$$\text{ReLU}(x) = \max(0.0f, x)$$
Negative activations are clamped to zero; positive values remain unchanged. Each thread processes one array element using `fmaxf(0.0f, val)`.

#### 3. Sigmoid Activation
$$\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$$
Squashes real-valued inputs into the interval $(0, 1)$. Each thread computes $1.0f / (1.0f + \expf(-x))$.

### 5.3 Execution Pipeline & Stream Semantics
CUDA kernels launched sequentially on the default stream (`stream 0`) execute in strict FIFO launch order. This hardware synchronization guarantees that:
$$\text{MatMul1} \longrightarrow \text{ReLU} \longrightarrow \text{MatMul2} \longrightarrow \text{Sigmoid}$$
execute sequentially without data hazards between layers.

---

## 6. Concepts Used
- **Grid, Block, and Thread Hierarchies**: Configuring 2D thread blocks (`dim3(16, 16)`) and calculating block indices.
- **Global Index Calculation**:
  $$\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$
  $$\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
- **CUDA Event Profiling**: Using `cudaEventRecord()` and `cudaEventElapsedTime()` to measure GPU kernel execution with microsecond resolution, avoiding host-CPU timer distortion.
- **Host vs. Device Memory Spaces**: Allocating VRAM via `cudaMalloc`, moving data via PCIe using `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost`, and deallocating with `cudaFree`.

---

## 7. Software and Hardware Requirements

### Target Hardware Environment
- **Demonstrated Workstation**: NVIDIA RTX 4500 Ada Generation GPU (24 GB GDDR6 with ECC, Ada Lovelace Architecture) / NVIDIA GeForce RTX 2050.
- **Student Host Architecture**: Intel Core Ultra 5 125H, 16 GB RAM, Intel Arc integrated graphics.

> [!NOTE] Hardware Compatibility Note
> Intel Arc integrated graphics is not an NVIDIA CUDA GPU. To compile and execute native CUDA programs locally, an NVIDIA GPU with the CUDA Toolkit and MSVC/GCC is required. For systems lacking an NVIDIA discrete GPU, alternatives include NVIDIA CUDA on WSL2 with an external/discrete NVIDIA GPU, Google Colab (Free T4 GPU runtime), or university GPU clusters.

### Software Stack
- **OS**: Windows 10/11 64-bit / Ubuntu Linux.
- **CUDA Toolkit**: CUDA Toolkit v12.x / v13.0 with `nvcc`.
- **C++ Compiler**: Microsoft Visual Studio 2022 Build Tools (MSVC `cl.exe` v143) / GCC `g++`.
- **Shell**: Developer Command Prompt for VS 2022 / PowerShell / Linux Bash.

---

## 8. Libraries Used
- `<cuda_runtime.h>`: CUDA runtime API (`cudaMalloc`, `cudaMemcpy`, `cudaEvent_t`, `cudaDeviceSynchronize`, `cudaFree`).
- `<iostream>`, `<iomanip>`: Formatted stream I/O.
- `<vector>`: Dynamic STL host buffers.
- `<chrono>`: High-resolution host steady clock for CPU and end-to-end profiling.
- `<cmath>`, `<algorithm>`: Math functions (`std::sin`, `std::cos`, `std::exp`, `std::fabs`, `std::max`).

---

## 9. Algorithm

### 9.1 GPU Forward-Pass Pipeline
1. Allocate device buffers: $dX, dW_1, db_1, dZ_1, dW_2, db_2, dZ_2$.
2. Asynchronously copy inputs and weights from host to device.
3. **Layer 1 Matrix Multiplication + Bias**:
   - Grid: $(\lceil 64/16 \rceil, \lceil 256/16 \rceil) = (4, 16)$ blocks.
   - Launch: `matrixMulBiasKernel<<<grid1, block>>>(dX, dW1, db1, dZ1, 256, 64, 128)`.
4. **Layer 1 ReLU Activation**:
   - Grid: $\lceil (256 \times 64) / 256 \rceil = 64$ blocks.
   - Launch: `reluKernel<<<64, 256>>>(dZ1, 16384)`.
5. **Layer 2 Matrix Multiplication + Bias**:
   - Grid: $(\lceil 10/16 \rceil, \lceil 256/16 \rceil) = (1, 16)$ blocks.
   - Launch: `matrixMulBiasKernel<<<grid2, block>>>(dZ1, dW2, db2, dZ2, 256, 10, 64)`.
6. **Layer 2 Sigmoid Activation**:
   - Grid: $\lceil (256 \times 10) / 256 \rceil = 10$ blocks.
   - Launch: `sigmoidKernel<<<10, 256>>>(dZ2, 2560)`.
7. Synchronize device; copy $dZ_2$ back to host memory `gpuOutput`.
8. Compare `gpuOutput` with CPU sequential output `cpuOutput`.

---

## 10. Step-by-Step Implementation
The CUDA program is constructed across 8 structured phases:

| Phase | Module Name | Architectural Function |
|:---:|:---|:---|
| **1** | Includes & Constants | Imports headers, defines `CUDA_CHECK` error handler, sets dimensions ($B=256, I=128, H=64, O=10$). |
| **2** | GPU Kernels | Implements `matrixMulBiasKernel`, `reluKernel`, and `sigmoidKernel`. |
| **3** | CPU Baseline | Reference implementation in standard C++ (`cpuForwardPass`). |
| **4** | Kernel Launcher | Sets grid/block configurations and launches the 4 GPU kernels in order. |
| **5** | Kernel Unit Tests | Tests matrix mul, ReLU, and Sigmoid on hand-verifiable minimal test matrices. |
| **6** | Main Start | Introspects GPU device, executes unit tests, generates deterministic sine/cosine inputs, times CPU. |
| **7** | GPU Warmup & Timing | Allocates VRAM, warms up GPU, measures kernel-only time with CUDA events, measures end-to-end time. |
| **8** | Output Verification | Validates $\max|CPU - GPU| < 10^{-4}$, prints first 10 outputs, frees VRAM, exits cleanly. |

---

## 11. Complete Source Code References
- **CUDA Source Code**: [`src/neural_network.cu`](src/neural_network.cu)
- **Makefile**: [`Makefile`](Makefile)

---

## 12. Compilation Commands

### Windows Developer Command Prompt (VS 2022 x64 Native)
```cmd
cd %USERPROFILE%\Documents\Experiment5_CUDA
nvcc -O2 neural_network.cu -o neural_network.exe
```

### Linux / Ubuntu Terminal
```bash
nvcc -O2 src/neural_network.cu -o neural_network
```

---

## 13. Execution Commands

### Windows
```cmd
neural_network.exe
```

### Linux
```bash
./neural_network
```

---

## 14. Sample Output
```text
EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA
GPU detected: NVIDIA RTX 4500 Ada Generation

Testing individual CUDA kernels...
Matrix multiplication + bias: PASSED
ReLU: PASSED
Sigmoid: PASSED

Forward-pass dimensions:
Input: 256 x 128
Hidden layer: 256 x 64
Output: 256 x 10

Average CPU compute time: 0.526860 ms
Average GPU kernel-only time: 0.026517 ms
Average GPU end-to-end time: 0.066010 ms
Kernel-only speedup: 19.868913x
End-to-end speedup: 7.981518x

Maximum absolute output difference: 0.000000
CPU and GPU outputs: MATCH (within tolerance)

First 10 output values (GPU):
0.498196 0.503101 0.508006 0.512911 0.517814 0.497729 0.502638 0.507548 0.512457 0.517364
```

---

## 15. Actual Execution Output & Verification

The verified experimental results recorded from the faculty demonstration run (NVIDIA RTX 4500 Ada Generation, CUDA Toolkit 13.0) are cataloged below:

### Kernel Unit Verification Table
| Test Kernel | Input Data | Expected Output | Status |
|:---|:---|:---|:---:|
| **Matrix Mul + Bias** | $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}, B = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}, \text{bias} = [1, -1]$ | $\begin{pmatrix} 23 & 27 \\ 50 & 63 \end{pmatrix}$ | **PASSED** |
| **ReLU Activation** | $[-2.0, 0.0, 3.0, -4.0, 5.0]$ | $[0.0, 0.0, 3.0, 0.0, 5.0]$ | **PASSED** |
| **Sigmoid Activation**| $[-2.0, 0.0, 2.0]$ | $[0.119203, 0.500000, 0.880797]$ | **PASSED** |

### Observation Table

| Metric | Measured Value (RTX 4500 Ada) | Unit | Status / Verification |
|:---|:---:|:---:|:---:|
| **CPU Average Compute Time** | 0.526860 | ms | Reference Baseline |
| **GPU Kernel-Only Time** | 0.026517 | ms | High-throughput compute |
| **GPU End-to-End Time** | 0.066010 | ms | Compute + PCIe Transfers |
| **Kernel-Only Speedup** | **19.868913** | × | $0.526860 / 0.026517$ |
| **End-to-End Speedup** | **7.981518** | × | $0.526860 / 0.066010$ |
| **Max Absolute Difference** | **0.000000** | — | Strictly $< 10^{-4}$ |
| **Output Match Status** | **MATCH (within tolerance)** | — | Numerically Validated |

Output log:
- Full output text: [`output/execution_output.txt`](output/execution_output.txt)
- Manual screenshot guidelines: [`screenshots/README.md`](screenshots/README.md)

---

## 16. Explanation of the Output
1. **Kernel Unit Tests**: All three isolated kernels passed unit testing on tiny matrices prior to network execution, confirming index arithmetic and mathematical formulas are strictly correct.
2. **Determinism**: Because inputs and weights are initialized using deterministic sine/cosine expressions, every GPU output matches the CPU output down to 6 decimal places ($\max\text{diff} = 0.000000$).
3. **Execution Timing Differences**:
   - **Kernel-Only ($0.026517\text{ ms}$)**: Measures pure GPU compute when tensors are already resident in VRAM. Yields **$19.87\times$ speedup**.
   - **End-to-End ($0.066010\text{ ms}$)**: Includes host-to-device PCIe transfers of $X, W_1, b_1, W_2, b_2$ and device-to-host copy of output $Y$. Yields **$7.98\times$ speedup**.
   - PCIe transfer latency accounts for $0.066010 - 0.026517 = 0.039493\text{ ms}$ ($\sim 60\%$ of total GPU execution time).

---

## 17. Performance Analysis

### Verification of Speedup Formulae
$$\text{Speedup}_{\text{kernel}} = \frac{\text{CPU Time}}{\text{GPU Kernel Time}} = \frac{0.526860\text{ ms}}{0.026517\text{ ms}} = 19.868913\times$$
$$\text{Speedup}_{\text{end-to-end}} = \frac{\text{CPU Time}}{\text{GPU End-to-End Time}} = \frac{0.526860\text{ ms}}{0.066010\text{ ms}} = 7.981518\times$$

### Why is End-to-End Speedup Lower than Kernel-Only Speedup?
PCI Express (PCIe) bus bandwidth (typically 16–32 GB/s) is orders of magnitude slower than on-chip GPU VRAM bandwidth (over 500 GB/s on modern architectures). For small batches ($B = 256$), the time required to copy tensors over the PCIe bus dominates the actual GPU execution time. For larger batch sizes (e.g., $B = 2048$ or larger image tensors), computation scales as $\mathcal{O}(N^3)$ while data transfer scales as $\mathcal{O}(N^2)$, causing end-to-end speedup to approach kernel-only speedup.

---

## 18. Advantages
- **High Concurrency**: Thousands of arithmetic threads compute dot products and activations concurrently.
- **Kernel Fusion Capability**: Bias addition is directly fused into the matrix multiplication write stage, saving an entire memory round-trip.
- **Deterministic Validation**: CPU-GPU verification guarantees zero numerical divergence before model deployment.

---

## 19. Limitations
- **Data Transfer Bottleneck**: Frequent host-to-device copies for small matrices introduce substantial PCIe latency penalties.
- **Hardware Lock-in**: Custom CUDA kernels require NVIDIA hardware and cannot run directly on Intel Arc, AMD ROCm, or Apple Metal without translation layers.
- **Shared Memory Unused**: This basic implementation uses global memory; further acceleration is possible with shared memory tiling.

---

## 20. Applications
- Autonomous driving inference pipelines (low-latency perception and trajectory scoring).
- High-frequency automated algorithmic trading networks.
- Edge AI inference engines (embedded NVIDIA Jetson modules).
- Custom inference servers serving quantized transformer and CNN models.

---

## 21. Conclusion
Custom CUDA kernels for matrix multiplication with fused bias, ReLU activation, and sigmoid activation were successfully implemented and combined into a two-layer neural-network forward pass. CPU and GPU outputs were compared, confirming exact numerical equivalence ($\max\text{diff} = 0.000000$). On the reference NVIDIA RTX 4500 Ada GPU, the custom CUDA implementation delivered a **$19.87\times$ kernel compute speedup** and a **$7.98\times$ end-to-end speedup**, clearly illustrating both the computational power of GPU acceleration and the practical impact of host-device memory transfer overhead.

---

## 22. Viva Questions and Answers

### Q1: What is a CUDA kernel?
**A:** A CUDA kernel is a function declared with the `__global__` qualifier that executes on the GPU device. When invoked, it is executed concurrently by thousands of lightweight threads across multiple streaming multiprocessors (SMs).

### Q2: What is the difference between host and device memory?
**A:** Host memory is the CPU system RAM, accessible by the host CPU. Device memory is the video RAM (VRAM) physically attached to the GPU. Data cannot be accessed directly by GPU threads while in host RAM; it must first be copied over the PCIe bus using `cudaMemcpy`.

### Q3: What do `blockIdx`, `blockDim`, and `threadIdx` represent?
**A:** 
- `threadIdx`: The unique coordinates of a thread within its current thread block.
- `blockDim`: The number of threads in each dimension of a thread block.
- `blockIdx`: The coordinates of the thread block within the execution grid.

### Q4: Why is a boundary check required in CUDA kernels?
**A:** Thread grids are allocated in multiples of the block dimensions (e.g., multiples of 16). If the problem size is not an exact multiple of the block dimension, the grid will spawn more threads than there are matrix elements. Boundary checks (`if (row < M && col < N)`) prevent out-of-bounds threads from reading or writing invalid memory addresses, preventing segmentation faults.

### Q5: Why are CUDA events used for kernel timing instead of CPU timers?
**A:** CUDA kernel launches are asynchronous with respect to the host CPU; the CPU returns immediately after launching the kernel while the GPU continues executing. CPU timers (`std::chrono`) would only measure launch queue time unless explicit synchronization is added. CUDA events (`cudaEventRecord`) record timestamps directly on the GPU hardware execution stream, providing sub-microsecond profiling accuracy.

### Q6: Why is a warm-up launch performed before timing?
**A:** The initial launch of a CUDA kernel involves one-time runtime overheads: driver initialization, GPU context creation, module loading, and clock frequency scaling (stepping out of low-power idle states). Performing a dummy warm-up pass primes the GPU, ensuring benchmark timings reflect steady-state execution.
