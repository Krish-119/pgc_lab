# Parallel and GPU Computing Laboratory (PGC Lab)

Complete practical laboratory implementation, comparative benchmarks, and comprehensive documentation for the Parallel and GPU Computing curriculum.

**Author:** Krish-119  
**Institution:** KLE Technological University  
**Repository:** [https://github.com/Krish-119/pgc_lab](https://github.com/Krish-119/pgc_lab)  

---

## 📚 Laboratory Experiments Index

| Exp No. | Experiment Title | Description | Technologies | Folder Link | Status |
|:---:|:---|:---|:---|:---:|:---:|
| **1** | **Parallel Matrix Multiplication** | $4000 \times 4000$ matrix multiplication across Sequential, OpenMP, MPI, and CUDA models. | C, OpenMP, Open MPI, CUDA | [Experiment_1_Matrix_Multiplication](Experiment_1_Matrix_Multiplication/) | ✅ **Completed** |
| **2** | **Multithreaded Programming** | Thread lifecycle, array work-sharing, race conditions, mutex locks, and OpenMP reductions. | C, Pthreads (`pthread`), OpenMP | [Experiment_2_Pthreads_OpenMP](Experiment_2_Pthreads_OpenMP/) | ✅ **Completed** |
| **3** | **Cache Coherence & NUMA Performance** | Snooping cache coherence simulation (MSI, MESI, MOESI), false sharing mitigation, and NUMA memory sweeps. | Python 3, C, OpenMP, `numactl`, `hwloc` | [Experiment_3_Cache_Coherence](Experiment_3_Cache_Coherence/) | ✅ **Completed** |
| **4** | **Synchronization Mechanisms** | Evaluation of race conditions, critical sections, hardware atomics, barriers, and POSIX semaphores. | C, OpenMP, POSIX Semaphores | [Experiment_4_Synchronization](Experiment_4_Synchronization/) | ✅ **Completed** |
| **5** | **Neural Network Layers in CUDA** | Custom CUDA kernels for matrix multiplication with fused bias, ReLU activation, and Sigmoid inference. | CUDA C++, C++, MSVC / GCC | [Experiment_5_Neural_Network_CUDA](Experiment_5_Neural_Network_CUDA/) | ✅ **Completed** |

---

## 📂 Laboratory Folder Structure

```text
Parallel_Computing_Lab/
│
├── Experiment_1_Matrix_Multiplication/
│   ├── README.md               # 22-section comprehensive experiment documentation
│   ├── Makefile                # Build automation for all models
│   ├── src/                    # Source files: sequential, openmp, mpi, cuda, hosts
│   ├── output/                 # Benchmark output logs and verification files
│   └── screenshots/            # Genuine execution terminal screenshots
│
├── Experiment_2_Pthreads_OpenMP/
│   ├── README.md               # 22-section comprehensive experiment documentation
│   ├── Makefile                # Build automation for all 12 programs
│   ├── src/                    # Source files: thread1, thread2, sum, race, mutex, perf
│   ├── output/                 # Multi-core execution outputs and scaling logs
│   └── screenshots/            # Thread execution output screenshot
│
├── Experiment_3_Cache_Coherence/
│   ├── README.md               # 22-section comprehensive experiment documentation
│   ├── Makefile                # Compilation for C benchmarks
│   ├── src/                    # coherence_sim.py, false_sharing.c, numa_sweep.c, compare_numa.c
│   ├── output/                 # Protocol simulator logs, false sharing tables, NUMA sweeps
│   └── screenshots/            # Manual terminal capture instructions & guide
│
├── Experiment_4_Synchronization/
│   ├── README.md               # 22-section comprehensive experiment documentation
│   ├── Exp_5_Synchronization_Mechanisms.pdf # Faculty reference document
│   ├── output/                 # Scalability and race condition comparison outputs
│   ├── screenshots/            # 5 genuine terminal screenshots (race, critical, barrier, sem, bench)
│   └── src/                    # race.c, critical.c, atomic.c, barrier.c, semaphore.c, compare_sync.c, Makefile
│
├── Experiment_5_Neural_Network_CUDA/
│   ├── README.md               # 22-section comprehensive experiment documentation
│   ├── Makefile                # nvcc build script
│   ├── src/                    # neural_network.cu (all 8 phases from faculty manual)
│   ├── output/                 # Full execution output transcript (RTX 4500 Ada reference run)
│   └── screenshots/            # Guidelines for environment, compiler, and kernel validation
│
└── README.md                   # Master index and laboratory overview (this file)
```

---

## 🔬 Experiment Summaries

### Experiment 1: Parallel Matrix Multiplication
- **Workload**: $4000 \times 4000$ matrix multiplication ($C = A \times B$), $A[i][k] = 1.0$, $B[k][j] = 1.0$.
- **Verification**: $C[0][0] = 4000.00$ verified across all implementations.
- **Results**:
  - **Sequential CPU**: $417.21\text{ s}$ ($1.00\times$ baseline).
  - **OpenMP (12 threads)**: $230.80\text{ s}$ ($1.81\times$ speedup).
  - **MPI (4 cluster VMs)**: $138.77\text{ s}$ ($3.01\times$ speedup).
  - **CUDA (RTX 2050 GPU)**: **$0.343\text{ s}$ ($1216.24\times$ speedup)**.

### Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
- **Workload**: Thread lifecycle management, array work-sharing (summing 8 integers to 360), uncoordinated race condition vs. mutex locks, and OpenMP reductions.
- **Scalability**: Benchmarked across $1, 2, 4, 6, 16$ threads. Both Pthreads and OpenMP achieve $\sim 9.6\times$ speedup at 16 threads ($\sim 60\%$ parallel efficiency due to memory bus saturation and Amdahl's serial fraction).

### Experiment 3: Cache Coherence (MSI, MESI, MOESI) and NUMA Memory Access
- **Part A (Protocol Simulation)**: Models discrete cache state transitions for 1000 iterations (4000 accesses). Demonstrates that MESI eliminates bus upgrades for private data via silent $E \to M$ transitions, while MOESI eliminates memory write-backs using Cache-to-Cache (C2C) transfers ($1999$ C2C transfers).
- **Part B (False Sharing Benchmark)**: At 4 threads, adjacent counter updates on the same 64-byte line degrade performance ($0.0347\text{ s}$), whereas 64-byte padded counters scale efficiently ($0.0180\text{ s}$, yielding a $1.93\times$ speed advantage).
- **Part C (NUMA Memory Sweep)**: 512 MB memory buffer sweep evaluating local vs. remote memory access.

### Experiment 4: Synchronization Mechanisms
- **Workload**: $10,000,000$ concurrent counter updates under high thread contention.
- **Findings**:
  - Unsynchronized execution loses over **73% of updates at 8 threads**.
  - **Atomics (`#pragma omp atomic`)** provide the highest throughput and scalability ($0.4004\text{ s}$ at 8 threads), running **$6.5\times$ faster than Critical Sections** and **$9.7\times$ faster than OpenMP Locks**.

### Experiment 5: Neural Network Layers in CUDA
- **Workload**: Two-layer feed-forward inference ($256 \times 128 \to 256 \times 64 \to 256 \times 10$) using custom CUDA kernels (matrix multiplication with fused bias, element-wise ReLU, and element-wise Sigmoid).
- **Verification**: Exact numerical equivalence between CPU and GPU forward passes ($\max\text{diff} = 0.000000 < 10^{-4}$).
- **Results (NVIDIA RTX 4500 Ada Workstation)**:
  - Average CPU compute time: $0.526860\text{ ms}$.
  - GPU kernel-only time: $0.026517\text{ ms}$ (**$19.87\times$ speedup**).
  - GPU end-to-end time: $0.066010\text{ ms}$ (**$7.98\times$ speedup**).

---

## 💻 System & Hardware Architecture

### Primary Student Workstation
- **Operating System**: Windows 11 (64-bit) & Ubuntu 22.04 LTS (WSL2 / VMware Workstation Pro).
- **Host Processor**: Intel Core Ultra 5 125H (14 cores / 18 threads, 16 GB RAM).
- **Host Graphics**: Intel Arc Integrated Graphics.
- **Compilers**: GCC 15.2.0 (`build-essential`), Python 3.12 / 3.13.

### Dedicated GPU Benchmark Testbed
- **Discrete GPU**: NVIDIA GeForce RTX 2050 (Exp 1) / NVIDIA RTX 4500 Ada Generation (Exp 5).
- **Toolchains**: NVIDIA CUDA Toolkit v12.x / v13.0 (`nvcc`), MSVC `cl.exe` (VS 2022 v143).

---

## 🛠️ Global Build & Execution Instructions

Inside Ubuntu (WSL2 / VMware):
```bash
# Experiment 1
cd Experiment_1_Matrix_Multiplication && make

# Experiment 2
cd ../Experiment_2_Pthreads_OpenMP && make

# Experiment 3
cd ../Experiment_3_Cache_Coherence && make
python3 src/coherence_sim.py 1000
./compare_numa

# Experiment 4
cd ../Experiment_4_Synchronization/src && make
./compare_sync

# Experiment 5 (on CUDA-capable workstation)
cd ../Experiment_5_Neural_Network_CUDA
nvcc -O2 src/neural_network.cu -o neural_network
./neural_network
```
