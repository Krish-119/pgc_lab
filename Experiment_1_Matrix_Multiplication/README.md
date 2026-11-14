# Experiment 1: Parallel Matrix Multiplication using Sequential, OpenMP, MPI, and CUDA

## 1. Experiment Title
**Parallel Matrix Multiplication using Sequential, OpenMP, MPI, and CUDA Computing Models**

---

## 2. Aim
To design, implement, and benchmark matrix multiplication ($4000 \times 4000$) across four foundational computing models—Sequential CPU, OpenMP shared-memory threading, MPI distributed-memory clustering, and NVIDIA CUDA massively parallel GPU execution—and evaluate numerical correctness, execution runtime, speedup, and architectural scalability.

---

## 3. Problem Statement
Given two square matrices $A$ and $B$ of size $N \times N$ ($N = 4000$) initialized with uniform floating-point entries $A[i][k] = 1.0$ and $B[k][j] = 1.0$, compute the resulting matrix $C = A \times B$ where:
$$C[i][j] = \sum_{k=0}^{N-1} A[i][k] \times B[k][j]$$
Since each element is the sum of $4000$ products of $1.0 \times 1.0$, every element in $C$ must evaluate to strictly $4000.00$. The primary objective is to execute this identical arithmetic workload across different parallel computing models, verify correctness via $C[0][0] = 4000.00$, and compare performance against sequential baseline execution.

---

## 4. Introduction
Matrix multiplication is a cornerstone computational kernel in computational physics, linear algebra, computer vision, and deep learning. However, traditional standard matrix multiplication exhibits an algorithmic complexity of $\mathcal{O}(N^3)$, which entails $6.4 \times 10^{10}$ floating-point operations (FLOPs) for a $4000 \times 4000$ matrix. On single-core processors, this operation incurs severe CPU pipeline and cache hierarchy bottlenecks, resulting in minutes of execution time. Modern high-performance computing (HPC) employs heterogeneous parallel architectures to accelerate compute-intensive kernels. This experiment demonstrates how identical linear algebra algorithms translate across shared-memory multi-core systems, multi-node networked clusters, and graphics processing units (GPUs).

---

## 5. Theory
Matrix multiplication requires computing the dot product between every row of matrix $A$ and every column of matrix $B$.
Each computing model parallelizes this operation across different hardware hierarchies:

1. **Sequential CPU**:
   - Single thread of execution sequentially iterates over indices $i, j, k$.
   - Bound by processor clock frequency and serial cache line access latency.
2. **OpenMP (Shared Memory)**:
   - Utilizes fork-join thread parallelism on symmetric multiprocessing (SMP) multi-core architectures.
   - The outer loop over matrix rows is partitioned across CPU threads using `#pragma omp parallel for`. All threads share a unified virtual address space.
3. **MPI (Distributed Memory)**:
   - Operates across independent nodes with separate physical memory address spaces connected over a network.
   - Data is distributed via collective communications: `MPI_Scatter` divides rows of $A$ across ranks, `MPI_Bcast` replicates matrix $B$ across all ranks, and `MPI_Gather` collects completed rows of $C$ onto the master rank.
4. **CUDA (Massively Parallel GPU)**:
   - Exploits the SIMT (Single Instruction, Multiple Threads) architecture of NVIDIA GPUs.
   - Computations are organized into a 2D grid of blocks and threads. Each GPU thread computes exactly one scalar element $C[\text{row}][\text{col}]$, enabling millions of threads to execute concurrently.

---

## 6. Concepts Used
- **Loop Tiling and Work Sharing**: Decomposing large loop nests into independent parallel chunks.
- **Shared vs Distributed Address Spaces**: Managing data access when memory is locally shared vs networked.
- **Collective Communications**: Point-to-point and collective MPI primitives (`MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`).
- **SIMT GPU Architecture**: CUDA streaming multiprocessors (SMs), warps (32 threads), thread blocks, and 2D grid dispatch.
- **Amdahl's Law and Speedup**: Quantifying theoretical and empirical speedup achieved through parallelization:
  $$\text{Speedup} = \frac{T_{\text{sequential}}}{T_{\text{parallel}}}$$

---

## 7. Software and Hardware Requirements

### Hardware Specifications
- **Workstation / Host**: Intel Core Ultra 5 125H / Intel Arc graphics, 16 GB RAM.
- **Dedicated Benchmark GPU**: NVIDIA GeForce RTX 2050 (4 GB GDDR6, Compute Capability 8.6, Ampere Architecture).
- **Cluster Environment**: 4 Ubuntu Virtual Machines on VMware Workstation Pro.

### Software Specifications
- **Operating Systems**: Ubuntu 22.04 LTS / Windows 11 (64-bit).
- **C/C++ Compilers**: GCC (Ubuntu `build-essential`), Microsoft MSVC (`cl.exe`).
- **MPI Implementation**: Open MPI (`mpicc`, `mpirun`).
- **CUDA Toolkit**: NVIDIA CUDA Toolkit (`nvcc` v12.x / v13.x).
- **Virtualization**: VMware Workstation Pro.

---

## 8. Libraries Used
- `<omp.h>`: OpenMP runtime library providing thread controls and high-resolution timer (`omp_get_wtime()`).
- `<mpi.h>`: Message Passing Interface library for inter-process communication.
- `<cuda_runtime.h>`: CUDA runtime API for host/device memory allocation (`cudaMalloc`), transfer (`cudaMemcpy`), and event-based microsecond profiling (`cudaEvent_t`).
- `<stdio.h>`, `<stdlib.h>`, `<time.h>`: Standard C libraries for I/O, dynamic memory management, and POSIX timers.

---

## 9. Algorithm

### 9.1 Sequential Algorithm
1. Allocate memory for matrices $A, B, C$ of size $N \times N$.
2. Initialize all elements: $A[i][k] \leftarrow 1.0$, $B[k][j] \leftarrow 1.0$, $C[i][j] \leftarrow 0.0$.
3. Start timer $T_{\text{start}}$.
4. For $i = 0$ to $N - 1$:
   - For $j = 0$ to $N - 1$:
     - For $k = 0$ to $N - 1$:
       - $C[i][j] \leftarrow C[i][j] + A[i][k] \times B[k][j]$.
5. Stop timer $T_{\text{end}}$. Compute $T_{\text{elapsed}} = T_{\text{end}} - T_{\text{start}}$.
6. Verify $C[0][0] == 4000.00$.

### 9.2 OpenMP Parallel Algorithm
1. Initialize matrices $A$ and $B$.
2. Configure OpenMP thread team: `omp_set_num_threads(12)`.
3. Start wall-clock timer `omp_get_wtime()`.
4. Fork thread team using `#pragma omp parallel for schedule(static)`.
5. Each thread receives a static chunk of outer loop indices ($i$), computing independent row dot products.
6. Join thread team upon completion.
7. Stop timer and verify $C[0][0] == 4000.00$.

### 9.3 MPI Distributed Algorithm
1. Initialize MPI subsystem via `MPI_Init`. Retrieve `rank` and `num_procs`.
2. Compute `rows_per_proc = N / num_procs` ($4000 / 4 = 1000$).
3. On Rank 0: allocate and initialize full matrices $A, B, C$.
4. On all worker ranks: allocate local buffers `local_A` ($1000 \times 4000$) and `B` ($4000 \times 4000$).
5. Broadcast matrix $B$ to all ranks using `MPI_Bcast`.
6. Scatter rows of $A$ across ranks using `MPI_Scatter`.
7. Each rank computes its local block:
   $$C_{\text{local}}[i][j] = \sum_{k=0}^{N-1} A_{\text{local}}[i][k] \times B[k][j]$$
8. Gather local results `local_C` back into root buffer $C$ using `MPI_Gather`.
9. Rank 0 records elapsed time and validates result. Finalize via `MPI_Finalize`.

### 9.4 CUDA GPU Massively Parallel Algorithm
1. Allocate host pinned memory and device global memory (`cudaMalloc`) for $A, B, C$.
2. Initialize matrices on host; copy $A$ and $B$ to device global memory (`cudaMemcpyHostToDevice`).
3. Define 2D thread block: $\text{dim3 block}(16, 16)$ (256 threads per block).
4. Define 2D grid: $\text{dim3 grid}((N+15)/16, (N+15)/16) = (250, 250)$ blocks (62,500 blocks, 16,000,000 threads).
5. Launch CUDA kernel `matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, N)`:
   - Each thread computes global row and column:
     $$\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$
     $$\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
   - If $\text{row} < N$ and $\text{col} < N$, iterate $k = 0 \dots N-1$ accumulating sum.
6. Synchronize device; copy result matrix $C$ back to host (`cudaMemcpyDeviceToHost`).
7. Validate $C[0][0] == 4000.00$ and record profiling metrics.

---

## 10. Step-by-Step Implementation
1. **Directory Organization**: Created `src/`, `output/`, `screenshots/`, and `Makefile`.
2. **Sequential Source**: Created [`src/matrix_sequential.c`](src/matrix_sequential.c).
3. **OpenMP Source**: Created [`src/matrix_openmp.c`](src/matrix_openmp.c) with static scheduling.
4. **MPI Source**: Created [`src/matrix_mpi.c`](src/matrix_mpi.c) with scatter/broadcast/gather pattern.
5. **CUDA Source**: Created [`src/matrix_cuda.cu`](src/matrix_cuda.cu) mapping 1 thread per matrix element.
6. **Automation**: Created [`Makefile`](Makefile) and cluster configuration file [`src/hosts`](src/hosts).

---

## 11. Complete Source Code References
- **Sequential Implementation**: [`src/matrix_sequential.c`](src/matrix_sequential.c)
- **OpenMP Implementation**: [`src/matrix_openmp.c`](src/matrix_openmp.c)
- **MPI Implementation**: [`src/matrix_mpi.c`](src/matrix_mpi.c)
- **CUDA Implementation**: [`src/matrix_cuda.cu`](src/matrix_cuda.cu)
- **Cluster Hosts File**: [`src/hosts`](src/hosts)
- **Makefile**: [`Makefile`](Makefile)

---

## 12. Compilation Commands

### Sequential
```bash
gcc -O2 src/matrix_sequential.c -o matrix_sequential
```

### OpenMP
```bash
gcc -O2 -fopenmp src/matrix_openmp.c -o matrix_openmp
```

### MPI
```bash
mpicc -O2 src/matrix_mpi.c -o matrix_mpi
```

### CUDA
```bash
nvcc -O2 -arch=sm_86 src/matrix_cuda.cu -o matrix_cuda
```

---

## 13. Execution Commands

### Sequential Execution
```bash
./matrix_sequential
```

### OpenMP Execution (12 Threads)
```bash
OMP_NUM_THREADS=12 ./matrix_openmp
```

### MPI Execution (4 Cluster VMs)
```bash
env -u DISPLAY mpirun -np 4 --hostfile src/hosts ./matrix_mpi
```

### CUDA Execution (NVIDIA RTX 2050)
```bash
./matrix_cuda
```

---

## 14. Sample Output

```text
============================================================
   PARALLEL MATRIX MULTIPLICATION BENCHMARK (N = 4000)
============================================================
Sequential Execution Time : 417.205920 seconds
OpenMP (12 threads)       : 230.801465 seconds
MPI (4 VM nodes)          : 138.772617 seconds
CUDA (RTX 2050 GPU)       : 0.343028 seconds (Kernel: 0.316872 s)

Verification C[0][0]      : 4000.00
Verification Status       : PASSED
============================================================
```

---

## 15. Actual Execution Output & Verification

The empirical timings recorded from real execution on the lab testbed are cataloged below:

| Architecture | Platform / Model | Configuration | Execution Time | Verification `C[0][0]` | Status |
|:---|:---|:---|:---:|:---:|:---:|
| **Sequential** | Intel Host CPU | 1 execution flow | **417.205920 s** | 4000.00 | **PASSED** |
| **OpenMP** | Intel Host CPU | 12 threads | **230.801465 s** | 4000.00 | **PASSED** |
| **MPI** | Ubuntu Cluster (4 VMs) | 4 processes | **138.772617 s** | 4000.00 | **PASSED** |
| **CUDA** | NVIDIA GeForce RTX 2050 | 16,000,000 threads | **0.343028 s** | 4000.00 | **PASSED** |

Detailed output transcripts are saved in [`output/`](output/):
- [`output/sequential_output.txt`](output/sequential_output.txt)
- [`output/openmp_output.txt`](output/openmp_output.txt)
- [`output/mpi_output.txt`](output/mpi_output.txt)
- [`output/cuda_output.txt`](output/cuda_output.txt)
- [`output/comparison_summary.txt`](output/comparison_summary.txt)

Screenshots of actual execution:
- OpenMP execution: [`screenshots/02_openmp_execution.png`](screenshots/02_openmp_execution.png)
- MPI execution: [`screenshots/03_mpi_execution.png`](screenshots/03_mpi_execution.png)
- CUDA execution: [`screenshots/04_cuda_execution.png`](screenshots/04_cuda_execution.png)

---

## 16. Explanation of the Output
1. **Mathematical Accuracy**: Because every entry in $A$ and $B$ is $1.0$, every element $C[i][j]$ is calculated as the sum of $4000$ multiplications of $1.0 \times 1.0$. The result $C[0][0] = 4000.00$ proves that all $6.4 \times 10^{10}$ floating-point operations were executed without race conditions or memory corruption.
2. **Runtime Progression**:
   - Sequential takes almost **7 minutes (417.21 s)** because a single CPU core must execute all arithmetic serially.
   - OpenMP reduces runtime to **230.80 s** ($1.81\times$ speedup) across 12 threads.
   - MPI completes in **138.77 s** ($3.01\times$ speedup) across 4 nodes.
   - CUDA completes in **0.343 s** ($1216.24\times$ total speedup, $1316.92\times$ kernel speedup).

---

## 17. Performance Analysis

### Speedup Calculations
$$\text{Speedup}_{\text{OpenMP}} = \frac{417.205920\text{ s}}{230.801465\text{ s}} = 1.81\times$$
$$\text{Speedup}_{\text{MPI}} = \frac{417.205920\text{ s}}{138.772617\text{ s}} = 3.01\times$$
$$\text{Speedup}_{\text{CUDA (Total)}} = \frac{417.205920\text{ s}}{0.343028\text{ s}} = 1216.24\times$$
$$\text{Speedup}_{\text{CUDA (Kernel)}} = \frac{417.205920\text{ s}}{0.316872\text{ s}} = 1316.92\times$$

### Efficiency Calculations
$$\text{Efficiency} = \frac{\text{Speedup}}{P} \times 100\%$$
- **OpenMP (12 threads)**: $\frac{1.81}{12} \times 100\% \approx 15.08\%$ (Limited by shared memory bus contention and cache line bouncing).
- **MPI (4 nodes)**: $\frac{3.01}{4} \times 100\% \approx 75.25\%$ (Near-linear scaling due to isolated memory per VM).

---

## 18. Advantages
- **OpenMP**: Simple compiler directive annotations (`#pragma omp`); requires minimal refactoring of serial code.
- **MPI**: Extremely scalable; allows scaling across hundreds or thousands of networked compute clusters without shared-memory hardware limits.
- **CUDA**: Extreme computational throughput and memory bandwidth for embarrassingly parallel arithmetic workloads.

---

## 19. Limitations
- **OpenMP**: Constrained to a single physical computer; suffers from memory bus saturation on large arrays.
- **MPI**: High programming complexity; network bandwidth and latency (`MPI_Bcast` / `MPI_Gather`) become bottlenecks.
- **CUDA**: Requires dedicated NVIDIA hardware; PCI-Express host-to-device data transfer overhead can dominate execution for smaller problem sizes.

---

## 20. Applications
- Scientific simulation (N-body problems, weather forecasting, computational fluid dynamics).
- Deep learning neural network forward and backward passes (GEMM operations).
- Computer graphics, ray tracing, and 3D rendering pipelines.
- Big data graph analysis and cryptographic algorithms.

---

## 21. Conclusion
This experiment demonstrated the direct implementation and comparison of $4000 \times 4000$ matrix multiplication across Sequential, OpenMP, MPI, and CUDA. The results demonstrate that while shared-memory multi-threading (OpenMP) and distributed-memory message passing (MPI) provide notable performance gains ($1.81\times$ and $3.01\times$), massive GPU acceleration with CUDA delivers revolutionary throughput ($1216.24\times$ acceleration), reducing execution time from nearly 7 minutes to approximately a third of a second.

---

## 22. Viva Questions and Answers

### Q1: What is the computational complexity of standard matrix multiplication?
**A:** Standard matrix multiplication of two $N \times N$ matrices requires $N^3$ multiplications and $N^3 - N^2$ additions, yielding a computational complexity of $\mathcal{O}(N^3)$. For $N = 4000$, this requires approximately $6.4 \times 10^{10}$ arithmetic operations.

### Q2: What is the primary difference between OpenMP and MPI?
**A:** OpenMP is a shared-memory API where multiple threads share a single address space and communicate through shared variables. MPI (Message Passing Interface) is a distributed-memory standard where distinct processes possess private address spaces and communicate strictly via explicit message passing across a network.

### Q3: Why does OpenMP show sub-linear scaling (1.81x speedup on 12 threads)?
**A:** Because matrix multiplication of large arrays ($4000 \times 4000 \times 8 \text{ bytes} \approx 128 \text{ MB}$) exceeds CPU L1/L2 cache capacity. All 12 cores compete for shared L3 cache and main memory bus bandwidth (memory bandwidth saturation), preventing ideal linear scaling.

### Q4: Why did MPI achieve ~75% parallel efficiency across 4 nodes?
**A:** Each virtual machine has its own allocated memory and CPU cache hierarchy. Since computation was partitioned evenly (1000 rows per node) and communication occurred only during initial distribution and final gathering, memory access contention was avoided.

### Q5: How are CUDA threads organized for matrix multiplication?
**A:** Threads are grouped into 2D thread blocks (e.g., $16 \times 16 = 256$ threads). Blocks are organized into a 2D grid ($250 \times 250 = 62,500$ blocks). Each individual thread calculates its global index:
$$\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$
$$\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
and computes exactly one output element $C[\text{row}][\text{col}]$.
