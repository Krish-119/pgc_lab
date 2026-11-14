# Experiment 4: Analyze Synchronization Mechanisms in Parallel Programs

## 1. Experiment Title
**Comparative Evaluation of Shared-Memory Synchronization Mechanisms: Race Conditions, Critical Sections, Atomic Primitives, Barriers, and Semaphores**

---

## 2. Aim
To implement, analyze, and benchmark essential shared-memory synchronization primitives in parallel programs (Race conditions, Critical Sections, Atomic operations, Barriers, and POSIX Semaphores), evaluating their impact on numerical correctness, serialization latency, lock contention, and execution scalability across varying thread configurations ($1, 2, 4, 8$ threads).

---

## 3. Problem Statement
In multi-threaded shared-memory architectures, concurrent threads accessing common data locations without synchronization trigger race conditions, resulting in lost updates and non-deterministic results. Conversely, introducing synchronization serializes execution, creating severe lock contention that diminishes parallel speedup.

The objective is to:
1. Demonstrate silent update loss in an unsynchronized parallel counter ($N = 10,000,000$).
2. Implement and verify mutual exclusion using `#pragma omp critical`, `#pragma omp atomic`, POSIX Semaphores (`sem_t`), and OpenMP Locks (`omp_lock_t`).
3. Demonstrate phase synchronization across threads using `#pragma omp barrier`.
4. Empirically quantify the trade-offs between correctness, execution latency, and scalability across $1 \to 8$ threads.

---

## 4. Introduction
Multi-threaded applications require synchronization primitives to coordinate access to shared state. When multiple threads perform a read-modify-write operation (`counter++`), the lack of atomicity at the machine-instruction level causes updates to be overwritten. 

Hardware and software synchronization mechanisms resolve this by enforcing mutual exclusion. However, these mechanisms range from coarse-grained software locks (such as OS semaphores and critical sections) to fine-grained hardware-assisted CPU primitives (such as atomic compare-and-swap). This experiment systematically evaluates each primitive under controlled high-contention conditions.

---

## 5. Theory

### 5.1 Why Synchronization is Needed
In shared-memory parallel programming, multiple threads execute within a shared address space. When multiple threads execute a read-modify-write instruction sequence on a shared variable without coordination, arbitrary OS scheduling and thread interleaving cause conflicting memory writes.

### 5.2 Anatomy of a Race Condition
The high-level statement `counter++` decomposes into three assembly instructions on x86-64:
1. `MOV eax, [counter]` (Load value from RAM/cache into CPU register)
2. `ADD eax, 1` (Increment register)
3. `MOV [counter], eax` (Store updated value back to RAM/cache)

If Thread 0 and Thread 1 execute step 1 simultaneously, both load the old value (e.g., `100`). Both increment to `101`, and both write `101` back. Two increments occurred, but the counter increased by only 1.

### 5.3 Synchronization Primitives Under Study
1. **Critical Section (`#pragma omp critical`)**:
   - Software mutual exclusion block managed by compiler runtime.
   - Only one thread can execute inside the critical block at any instant; all others block.
2. **Atomic Operation (`#pragma omp atomic`)**:
   - Hardware-assisted atomic memory update (utilizing `LOCK XADD` or `LOCK CMPXCHG` instructions on x86).
   - Locks the specific memory cache line during the single instruction cycle, bypassing OS-level context switching.
3. **Barrier Synchronization (`#pragma omp barrier`)**:
   - Phase alignment primitive. Ensures all threads reach the barrier before any thread proceeds to the subsequent execution phase.
4. **POSIX Semaphore (`sem_t`)**:
   - OS-level integer semaphore (`sem_wait`, `sem_post`). When counter is 0, calling threads transition to kernel wait queues, avoiding CPU-spinning cycles.
5. **OpenMP Lock (`omp_lock_t`)**:
   - Low-level mutex primitive offering explicit procedural lock acquisition and release (`omp_set_lock`, `omp_unset_lock`).

---

## 6. Concepts Used
- **Mutual Exclusion (Mutex)**: Preventing simultaneous access to a shared resource by multiple concurrent threads.
- **Hardware Bus Locking vs Software Queuing**: Comparing atomic CPU-level memory cycles against OS thread sleep/wake cycles.
- **Lock Contention**: The performance penalty incurred when multiple execution threads contend for the same lock simultaneously.
- **Convoy Effect**: When faster threads queue behind a thread executing a locked critical section.

---

## 7. Software and Hardware Requirements
- **Host Architecture**: Intel Core Ultra 5 125H (14 cores / 18 threads, 16 GB RAM).
- **Environment**: Ubuntu 22.04 LTS (VMware Workstation Pro / WSL2).
- **Compiler**: GCC 15.2.0 (`build-essential`).
- **Libraries**: OpenMP (`libgomp`), POSIX Threads (`pthread`).

---

## 8. Libraries Used
- `<omp.h>`: OpenMP threading, lock API (`omp_lock_t`), atomics, critical sections, and wall-clock timer (`omp_get_wtime()`).
- `<pthread.h>` & `<semaphore.h>`: POSIX threads, semaphore initialization (`sem_init`), wait (`sem_wait`), post (`sem_post`), and destruction (`sem_destroy`).
- `<stdio.h>`, `<stdlib.h>`: Standard I/O and process exit routines.

---

## 9. Algorithm

### 9.1 Comparative Benchmark Algorithm (`compare_sync.c`)
1. Define total shared increments $N = 10,000,000$.
2. Configure thread counts array: $T \in \{1, 2, 4, 8\}$.
3. For each thread count $T$:
   - **None (Race Condition)**:
     - Set `counter = 0`.
     - In parallel: `#pragma omp parallel for`: execute `counter++` uncoordinated.
     - Record elapsed time and final counter.
   - **Critical Section**:
     - Set `counter = 0`.
     - In parallel: `#pragma omp parallel for`: execute `#pragma omp critical { counter++; }`.
     - Record elapsed time and final counter.
   - **Atomic Operation**:
     - Set `counter = 0`.
     - In parallel: `#pragma omp parallel for`: execute `#pragma omp atomic counter++;`.
     - Record elapsed time and final counter.
   - **OpenMP Lock**:
     - Initialize `omp_lock_t lock`.
     - In parallel: `#pragma omp parallel for`: acquire `omp_set_lock(&lock)`, increment `counter++`, release `omp_unset_lock(&lock)`.
     - Record elapsed time and destroy lock.
4. Compare correctness ($\text{counter} == N$) and scalability across all runs.

---

## 10. Step-by-Step Implementation
1. **Folder Setup**: Created `Experiment_4_Synchronization/` with `src/`, `output/`, `screenshots/`, and `Makefile`.
2. **Race Condition Demo**: [`src/race.c`](src/race.c).
3. **Critical Section Demo**: [`src/critical.c`](src/critical.c).
4. **Atomic Primitive Demo**: [`src/atomic.c`](src/atomic.c).
5. **Barrier Coordination Demo**: [`src/barrier.c`](src/barrier.c).
6. **POSIX Semaphore Demo**: [`src/semaphore.c`](src/semaphore.c).
7. **Comprehensive Multi-Thread Scalability Benchmark**: [`src/compare_sync.c`](src/compare_sync.c).
8. **Automated Build**: [`src/Makefile`](src/Makefile).

---

## 11. Complete Source Code References
- **Race Condition**: [`src/race.c`](src/race.c)
- **Critical Section**: [`src/critical.c`](src/critical.c)
- **Atomic Operations**: [`src/atomic.c`](src/atomic.c)
- **Barrier Coordination**: [`src/barrier.c`](src/barrier.c)
- **POSIX Semaphore**: [`src/semaphore.c`](src/semaphore.c)
- **Comparative Benchmark**: [`src/compare_sync.c`](src/compare_sync.c)
- **Build Automation**: [`src/Makefile`](src/Makefile)

---

## 12. Compilation Commands
```bash
gcc -Wall -O2 -fopenmp src/race.c -o race
gcc -Wall -O2 -fopenmp src/critical.c -o critical
gcc -Wall -O2 -fopenmp src/atomic.c -o atomic
gcc -Wall -O2 -fopenmp src/barrier.c -o barrier
gcc -Wall -O2 -pthread src/semaphore.c -o semaphore
gcc -Wall -O2 -fopenmp src/compare_sync.c -o compare_sync
```
*(Or simply run `make` inside `src/`)*

---

## 13. Execution Commands
```bash
./race
./critical
./atomic
./barrier
./semaphore
./compare_sync
```

---

## 14. Sample Output
```text
Comparing synchronization mechanisms for 10000000 shared increments

Mechanism  | Threads | Final Count  | Correctness | Time (s)
---------------------------------------------------------------
None       |       1 |     10000000 |    Correct |     0.0114
Critical   |       1 |     10000000 |    Correct |     0.2164
Atomic     |       1 |     10000000 |    Correct |     0.1363
Lock       |       1 |     10000000 |    Correct |     0.2146
---------------------------------------------------------------
None       |       4 |      5303565 |  INCORRECT |     0.0253
Critical   |       4 |     10000000 |    Correct |     1.1181
Atomic     |       4 |     10000000 |    Correct |     0.3867
Lock       |       4 |     10000000 |    Correct |     2.4513
```

---

## 15. Actual Execution Output & Verification

### Measured Empirical Data (krish@KLAP)
| Mechanism | Threads ($T$) | Expected Count | Final Count | Correctness | Execution Time (s) | Update Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **None** | 1 | 10,000,000 | 10,000,000 | **Correct** | **0.0114 s** | 100.0% |
| **Critical** | 1 | 10,000,000 | 10,000,000 | **Correct** | 0.2164 s | 100.0% |
| **Atomic** | 1 | 10,000,000 | 10,000,000 | **Correct** | 0.1363 s | 100.0% |
| **Lock** | 1 | 10,000,000 | 10,000,000 | **Correct** | 0.2146 s | 100.0% |
| | | | | | | |
| **None** | 2 | 10,000,000 | 6,316,341 | ❌ **INCORRECT** | **0.0209 s** | 63.16% (36.8% lost) |
| **Critical** | 2 | 10,000,000 | 10,000,000 | **Correct** | 0.5283 s | 100.0% |
| **Atomic** | 2 | 10,000,000 | 10,000,000 | **Correct** | **0.3236 s** | 100.0% |
| **Lock** | 2 | 10,000,000 | 10,000,000 | **Correct** | 0.7448 s | 100.0% |
| | | | | | | |
| **None** | 4 | 10,000,000 | 5,303,565 | ❌ **INCORRECT** | **0.0253 s** | 53.04% (47.0% lost) |
| **Critical** | 4 | 10,000,000 | 10,000,000 | **Correct** | 1.1181 s | 100.0% |
| **Atomic** | 4 | 10,000,000 | 10,000,000 | **Correct** | **0.3867 s** | 100.0% |
| **Lock** | 4 | 10,000,000 | 10,000,000 | **Correct** | 2.4513 s | 100.0% |
| | | | | | | |
| **None** | 8 | 10,000,000 | 2,681,003 | ❌ **INCORRECT** | **0.0319 s** | 26.81% (73.2% lost) |
| **Critical** | 8 | 10,000,000 | 10,000,000 | **Correct** | 2.5889 s | 100.0% |
| **Atomic** | 8 | 10,000,000 | 10,000,000 | **Correct** | **0.4004 s** | 100.0% |
| **Lock** | 8 | 10,000,000 | 10,000,000 | **Correct** | 3.9011 s | 100.0% |

Output logs:
- Benchmark transcript: [`output/compare_sync_output.txt`](output/compare_sync_output.txt)

Screenshots:
- Environment & Race Condition: [`screenshots/01_environment_and_race_condition.png`](screenshots/01_environment_and_race_condition.png)
- Critical Section: [`screenshots/02_critical_section.png`](screenshots/02_critical_section.png)
- Barrier Coordination: [`screenshots/03_barrier_synchronization.png`](screenshots/03_barrier_synchronization.png)
- POSIX Semaphore: [`screenshots/04_posix_semaphore.png`](screenshots/04_posix_semaphore.png)
- Benchmark Scalability: [`screenshots/05_comparison_benchmark_scalability.png`](screenshots/05_comparison_benchmark_scalability.png)

---

## 16. Explanation of the Output
1. **Unsynchronized Catastrophe**: As thread count increases from $1 \to 8$, data loss escalates dramatically: at 8 threads, **73.2% of increments are lost** because threads constantly overwrite registers with stale values.
2. **Atomics Dominate Performance**: `#pragma omp atomic` achieves correctness while taking only **$0.4004\text{ s}$ at 8 threads**, which is **$6.5\times$ faster than Critical Sections ($2.5889\text{ s}$)** and **$9.7\times$ faster than OpenMP Locks ($3.9011\text{ s}$)**.
3. **Lock Contention Bottleneck**: General-purpose locks (`critical` and `omp_lock_t`) suffer from thread convoying; as more threads wait for lock release, context switching overhead causes runtime to grow linearly with thread count.

---

## 17. Performance Analysis

### Relative Latency Multipliers (at 8 Threads)
- **Baseline (Atomic)**: $1.0\times$ ($0.4004\text{ s}$)
- **Critical Section**: $\frac{2.5889}{0.4004} \approx 6.47\times$ slower
- **Explicit Lock**: $\frac{3.9011}{0.4004} \approx 9.74\times$ slower

### Observation Summary Table

| Primitive | Correctness | Performance | Scalability | Recommended Usage |
|:---|:---:|:---:|:---:|:---|
| **None** | ❌ Incorrect (Lossy) | Extremely Fast | Poor (Data loss worsens) | Read-only access or uncoordinated benchmarking |
| **Critical** | ✅ Correct | Slow | Poor (Serializes threads) | Multi-statement critical sections / complex updates |
| **Atomic** | ✅ Correct | **Very Fast** | **Excellent** | Single-variable scalar increments and accumulators |
| **Lock** | ✅ Correct | Slowest | Poor | Dynamic, fine-grained multi-lock data structures |
| **Barrier** | Phase Sync | Moderate | Moderate | Multi-phase step-by-step algorithms |
| **Semaphore** | ✅ Correct | Moderate | Moderate | Bounded buffers, producer-consumer queues |

---

## 18. Advantages
- **Atomics**: Extremely fast, lock-free hardware execution with zero OS scheduling latency.
- **Critical Sections**: Deadlock-free structured scoping managed automatically by compiler.
- **Semaphores**: Ability to guard pools of resources (counting semaphores) beyond binary locks.

---

## 19. Limitations
- **Atomics**: Restricted strictly to simple binary expressions and memory-aligned scalar types.
- **Critical Sections**: Cannot protect distinct, decoupled variables with independent locks unless named critical sections are used.
- **Semaphores**: Risk of deadlocks, priority inversion, and forgotten `sem_post()` calls.

---

## 20. Applications
- Thread-safe memory allocators and reference counting (`std::shared_ptr` internals).
- High-throughput parallel reduction engines.
- Lock-free ring buffers in real-time multimedia streaming.
- Stencil computation and multi-grid PDE solvers requiring barrier phase alignment.

---

## 21. Conclusion
This experiment demonstrated that synchronization is essential to prevent data corruption in multi-threaded programs, where unsynchronized increments lose up to 73% of updates under 8-thread concurrency. Benchmarking revealed that hardware-assisted atomic primitives (`#pragma omp atomic`) achieve the lowest latency and best scalability ($6.5\times$ faster than critical sections and $9.7\times$ faster than locks), making them the preferred synchronization choice for scalar variable updates.

---

## 22. Viva Questions and Answers

### Q1: What is the root cause of a race condition?
**A:** A race condition occurs when concurrent threads access a shared memory location simultaneously, at least one access is a write, and no synchronization coordinates the accesses. The non-atomic execution of read-modify-write sequences causes one thread to overwrite updates made by another.

### Q2: Why is `#pragma omp atomic` significantly faster than `#pragma omp critical`?
**A:** `#pragma omp atomic` compiles directly into native hardware atomic instructions (such as `LOCK XADD` on x86-64), requiring only a momentary bus or cache lock for a single CPU cycle. In contrast, `#pragma omp critical` implements software-managed mutex locking involving thread suspension, OS wait queues, and context switching.

### Q3: What is a deadlock and how can it occur with semaphores?
**A:** A deadlock occurs when two or more threads are permanently blocked, each waiting for a lock or semaphore held by the other. For instance, Thread 1 acquires Semaphore A and waits for Semaphore B, while Thread 2 acquires Semaphore B and waits for Semaphore A.

### Q4: When is `#pragma omp barrier` necessary?
**A:** A barrier is required whenever an algorithm has distinct computational phases where phase $k+1$ depends on the completed results of phase $k$ across all threads (for example, in iterative parallel PDE solvers or game simulation frames).

### Q5: What is the difference between a binary semaphore and a counting semaphore?
**A:** A binary semaphore has a maximum value of 1 and acts as a mutual exclusion lock (protecting a single resource). A counting semaphore can take values greater than 1 and is used to manage pools of identical shared resources (such as free database connections or buffer slots).
