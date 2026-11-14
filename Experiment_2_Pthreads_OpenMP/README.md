# Experiment 2: Multithreaded Programming Using Pthreads and OpenMP

## 1. Experiment Title
**Multithreaded Programming Using Pthreads and OpenMP: Thread Management, Coordination, Synchronization, and Performance Scalability**

---

## 2. Aim
To develop, analyze, and benchmark multithreaded shared-memory programs using POSIX Threads (`pthread`) and OpenMP compiler directives to study:
- Explicit vs directive-based thread creation and lifecycle management.
- Work partitioning and load distribution over arrays.
- Race conditions caused by unsynchronized concurrent shared-variable updates.
- Mutual exclusion and synchronization using Mutexes (`pthread_mutex_t`), Critical Sections (`#pragma omp critical`), and Barriers (`#pragma omp barrier`).
- Scalability, execution runtime, speedup, and parallel efficiency across 1, 2, 4, 6, and 16 threads.

---

## 3. Problem Statement
Concurrent execution within a shared address space introduces non-deterministic thread scheduling. If multiple execution flows read, modify, and write back shared resources without proper coordination, data corruption (lost updates) occurs. Conversely, excessive synchronization incurs severe serialization latency. The goal of this experiment is to:
1. Implement thread creation, array work partitioning (summing array `[10, 20, 30, 40, 50, 60, 70, 80]` to obtain total `360`), and synchronization mechanisms in both low-level Pthreads and high-level OpenMP.
2. Measure and contrast their execution times across thread counts ($1 \to 16$) against a serial baseline ($1.353219\text{ s}$), explaining the causes of sub-linear scaling.

---

## 4. Introduction
Multi-core microprocessors require multithreaded software architectures to utilize hardware parallelism effectively. Shared-memory programming primarily follows two paradigms:
1. **Pthreads (POSIX Threads)**: An explicit, low-level, library-based API (`<pthread.h>`) where the programmer explicitly creates threads, packs function arguments into structs, manages stack life-cycles, and manipulates mutex locks.
2. **OpenMP (Open Multi-Processing)**: A high-level, declarative, compiler-directive model (`<omp.h>`) that automatically manages thread pools, schedules loop iterations (`#pragma omp parallel for`), and handles reduction aggregations with minimal syntax overhead.

---

## 5. Theory
A thread is the smallest schedulable unit of execution within an operating system process. All threads in a process share code, global memory, and open file descriptors, but maintain private registers, program counters, and stack memory.

### Race Condition Mechanics
A statement such as `counter++` is executed as three distinct machine-level instructions:
1. `LOAD R1, [counter]`
2. `ADD R1, 1`
3. `STORE [counter], R1`

When two threads execute these instructions without synchronization, their sequences interleave arbitrarily. If Thread 1 and Thread 2 both read the initial value `0` concurrently, both increment it to `1`, and both write `1` back, one update is permanently lost.

### Synchronization Primitives
- **Mutex (`pthread_mutex_t`)**: A mutual exclusion lock. Only the thread holding the lock may execute the critical section.
- **Critical Section (`#pragma omp critical`)**: Compiler-managed mutual exclusion ensuring single-thread execution through structured blocks.
- **Barrier (`#pragma omp barrier` / `pthread_barrier_t`)**: Synchronizes a group of threads by forcing all threads to wait until the entire team arrives before proceeding.
- **Reduction**: OpenMP optimization where each thread maintains a private accumulator that is combined atomically into the shared variable upon loop completion.

---

## 6. Concepts Used
- **Fork-Join Execution Model**: Creating multiple concurrent worker threads from a master thread and synchronizing upon completion.
- **Work Sharing & Chunk Decomposition**: Block distribution of array indices across threads.
- **Mutual Exclusion & Lock Contention**: Trade-offs between data safety and execution serialization.
- **Amdahl's Law**: Mathematical limit of speedup governed by the sequential fraction of an application:
  $$S(p) = \frac{1}{(1 - f) + \frac{f}{p}}$$

---

## 7. Software and Hardware Requirements
- **Host System**: Windows 11 / Ubuntu 22.04 LTS (WSL2 / VMware).
- **Processor**: Intel Core Ultra 5 125H (14 cores / 18 threads, 16 GB RAM).
- **Compilers**: GCC 15.2.0 (`build-essential`).
- **Flags**: `-pthread` for Pthreads, `-fopenmp` for OpenMP.

---

## 8. Libraries Used
- `<pthread.h>`: POSIX Threads API (`pthread_create`, `pthread_join`, `pthread_mutex_init`, `pthread_mutex_lock`, `pthread_mutex_unlock`, `pthread_mutex_destroy`).
- `<omp.h>`: OpenMP runtime library (`omp_get_thread_num`, `omp_get_num_threads`, `omp_set_num_threads`, `omp_get_wtime`).
- `<stdio.h>`, `<stdlib.h>`, `<time.h>`: Standard I/O, memory management, and monotonic clock measurement.

---

## 9. Algorithm

### 9.1 Pthreads Work-Sharing Algorithm
1. Define array of 8 integers: `[10, 20, 30, 40, 50, 60, 70, 80]`.
2. Determine chunk size: `chunk = ARRAY_SIZE / NUM_THREADS` ($8 / 4 = 2$).
3. Spawn 4 threads using `pthread_create()`, passing thread ID ($0, 1, 2, 3$).
4. Each thread calculates its range `[start, end)` and sums its assigned elements into a local accumulator.
5. Threads store local sums in a global `partial_sums` array and terminate.
6. Main thread joins all threads with `pthread_join()`, sums the partial results, and verifies total is `360`.

### 9.2 OpenMP Work-Sharing and Reduction
1. Declare total accumulator `total_sum = 0`.
2. Dispatch parallel loop:
   ```c
   #pragma omp parallel for reduction(+:total_sum)
   for (int i = 0; i < 8; i++) total_sum += array[i];
   ```
3. OpenMP automatically assigns loop iterations to threads and aggregates results safely.
4. Verify `total_sum == 360`.

---

## 10. Step-by-Step Implementation
1. **Folder Setup**: Created `Experiment_2_Pthreads_OpenMP/` with `src/`, `output/`, `screenshots/`, and `Makefile`.
2. **Pthreads Development**:
   - `thread1.c`: Single thread create/join.
   - `thread2.c`: Multi-thread creation (4 threads).
   - `thread_sum.c`: Array chunk partitioning (sum = 360).
   - `race.c`: Unsynchronized counter racing.
   - `mutex.c`: Mutex synchronization (counter = 400000).
   - `pthread_perf.c`: Multi-core scalability benchmark.
3. **OpenMP Development**:
   - `omp1.c`: Parallel region and thread ID introspection.
   - `omp_sum.c`: Work sharing with reduction (sum = 360).
   - `omp_race.c`: Race condition demonstration.
   - `omp_critical.c`: Critical section protection.
   - `omp_barrier.c`: Two-phase barrier synchronization.
   - `omp_perf.c`: OpenMP multi-core scalability benchmark.
4. **Build Automation**: Structured `Makefile` supporting compilation of all binaries.

---

## 11. Complete Source Code References
- **Pthreads Programs**:
  - [`src/thread1.c`](src/thread1.c)
  - [`src/thread2.c`](src/thread2.c)
  - [`src/thread_sum.c`](src/thread_sum.c)
  - [`src/race.c`](src/race.c)
  - [`src/mutex.c`](src/mutex.c)
  - [`src/pthread_perf.c`](src/pthread_perf.c)
- **OpenMP Programs**:
  - [`src/omp1.c`](src/omp1.c)
  - [`src/omp_sum.c`](src/omp_sum.c)
  - [`src/omp_race.c`](src/omp_race.c)
  - [`src/omp_critical.c`](src/omp_critical.c)
  - [`src/omp_barrier.c`](src/omp_barrier.c)
  - [`src/omp_perf.c`](src/omp_perf.c)
- **Makefile**: [`Makefile`](Makefile)

---

## 12. Compilation Commands

### Pthreads
```bash
gcc -Wall -O2 -pthread src/thread1.c -o thread1
gcc -Wall -O2 -pthread src/thread2.c -o thread2
gcc -Wall -O2 -pthread src/thread_sum.c -o thread_sum
gcc -Wall -O2 -pthread src/race.c -o race
gcc -Wall -O2 -pthread src/mutex.c -o mutex
gcc -Wall -O2 -pthread src/pthread_perf.c -o pthread_perf
```

### OpenMP
```bash
gcc -Wall -O2 -fopenmp src/omp1.c -o omp1
gcc -Wall -O2 -fopenmp src/omp_sum.c -o omp_sum
gcc -Wall -O2 -fopenmp src/omp_race.c -o omp_race
gcc -Wall -O2 -fopenmp src/omp_critical.c -o omp_critical
gcc -Wall -O2 -fopenmp src/omp_barrier.c -o omp_barrier
gcc -Wall -O2 -fopenmp src/omp_perf.c -o omp_perf
```

---

## 13. Execution Commands
```bash
./thread1
./thread2
./thread_sum
./race
./mutex
./pthread_perf

./omp1
./omp_sum
./omp_race
./omp_critical
./omp_barrier
./omp_perf
```

---

## 14. Sample Output
```text
[thread_sum]
Input Array: 10 20 30 40 50 60 70 80
Thread 0 processed indices [0..1], partial sum = 30
Thread 1 processed indices [2..3], partial sum = 70
Thread 2 processed indices [4..5], partial sum = 110
Thread 3 processed indices [6..7], partial sum = 150
Total sum = 360 (Expected: 360)

[mutex]
=== Pthreads Mutex Protection ===
Expected counter = 400000
Actual counter   = 400000
Status           = CORRECT
```

---

## 15. Actual Execution Output & Verification

### Performance & Scalability Data
Sequential Baseline Execution Time: **1.353219 seconds**

| Threads | Pthreads Time (s) | Pthreads Speedup | Pthreads Efficiency | OpenMP Time (s) | OpenMP Speedup | OpenMP Efficiency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1.348142 | 1.004× | 100.38% | 1.409294 | 0.960× | 96.02% |
| **2** | 0.680737 | 1.988× | 99.39% | 0.715560 | 1.891× | 94.56% |
| **4** | 0.358872 | 3.771× | 94.27% | 0.360803 | 3.751× | 93.76% |
| **6** | 0.241345 | 5.608× | 93.45% | 0.241608 | 5.601× | 93.35% |
| **16** | 0.144812 | 9.345× | 58.40% | 0.140692 | 9.618× | 60.11% |

Output logs:
- [`output/pthreads_output.txt`](output/pthreads_output.txt)
- [`output/openmp_output.txt`](output/openmp_output.txt)
- [`output/performance_benchmark.txt`](output/performance_benchmark.txt)
- Terminal execution screenshot: [`screenshots/01_multithreaded_execution.png`](screenshots/01_multithreaded_execution.png)

---

## 16. Explanation of the Output
1. **Thread Creation Non-determinism**: In `thread2.c` and `omp1.c`, threads print their greetings out of numerical order because the OS kernel scheduler schedules ready threads dynamically.
2. **Race Condition Demonstration**: In `race.c` and `omp_race.c`, the actual counter falls well short of the expected value ($278,142 < 400,000$), directly confirming that simultaneous uncoordinated memory access produces lost updates.
3. **Synchronization Correctness**: Mutex locks and OpenMP critical sections enforce mutual exclusion, restoring deterministic correctness ($400,000$ and $1,000,000$).
4. **Performance Scaling**: As thread count scales from 1 to 6 threads, speedup increases near-linearly ($5.608\times$ for Pthreads, $5.601\times$ for OpenMP). At 16 threads, speedup reaches $\sim 9.6\times$ (efficiency drops to $\sim 60\%$).

---

## 17. Performance Analysis
### Why does 16 threads not yield 16x speedup?
1. **Amdahl's Sequential Fraction**: Portions of code (thread spawning, memory initialization, reduction aggregation, joining) remain inherently sequential.
2. **Memory Bus Contention**: As 16 threads read and write memory concurrently, the shared memory bus and L3 cache bandwidth become saturated.
3. **OS Scheduling and Hyper-Threading Overhead**: Simultaneous multithreading (SMT) shares execution units on physical CPU cores rather than providing completely independent physical execution pipelines.
4. **Thread Management Overhead**: Creating, synchronizing, and joining 16 threads incurs non-trivial kernel-level context switching latency.

---

## 18. Advantages
- **Pthreads**: Maximum granular control over thread attributes, scheduling policies, stack allocation, and detachment state.
- **OpenMP**: High developer productivity, cleaner readable code, seamless loop parallelization, and built-in reduction handling.

---

## 19. Limitations
- **Pthreads**: Verbose boilerplate code; error-prone manual memory and mutex management; non-portable across non-POSIX operating systems without emulation.
- **OpenMP**: Less fine-grained control over individual thread life-cycles; requires compiler support; constrained to shared-memory nodes.

---

## 20. Applications
- Multithreaded web servers (handling concurrent HTTP client requests).
- Database transaction processing engines.
- Audio and real-time signal processing pipelines.
- Multi-core gaming simulation loops (physics engine, AI pathfinding).

---

## 21. Conclusion
This experiment demonstrated multithreaded programming in Pthreads and OpenMP. Pthreads provides explicit thread and mutex control, whereas OpenMP abstracts thread management through compiler directives. Both models successfully parallelized work over arrays and achieved comparable speedups ($\sim 9.6\times$ at 16 threads). Unsynchronized shared memory access reliably manifested race conditions, which were effectively rectified using mutual exclusion primitives.

---

## 22. Viva Questions and Answers

### Q1: What is the main difference between Pthreads and OpenMP?
**A:** Pthreads is an explicit, library-based low-level API where the programmer manually creates, manages, and joins threads and locks. OpenMP is a high-level, declarative, compiler-directive model that automatically manages thread pools, loop scheduling, and work sharing via pragmas.

### Q2: What is a race condition and how is it detected?
**A:** A race condition occurs when two or more threads access a shared memory location concurrently and at least one access is a write, without synchronization. It causes lost updates and non-deterministic results. It can be detected using tools like Valgrind Helgrind or ThreadSanitizer.

### Q3: What is the difference between a mutex and a critical section?
**A:** A mutex (`pthread_mutex_t`) is a lock variable that can be explicitly locked and unlocked anywhere in the code. A critical section (`#pragma omp critical`) is a structured code block where the compiler and runtime automatically ensure only one thread executes inside the block at any time.

### Q4: What does `#pragma omp barrier` do?
**A:** It creates an explicit synchronization point. No thread in the parallel team can proceed past the barrier until all threads in that team have reached the barrier.

### Q5: How does OpenMP handle reductions without race conditions?
**A:** The `reduction(+:variable)` clause creates a private copy of the variable for each thread, initialized to 0. Each thread updates its private copy independently during the loop without lock overhead. At the end of the parallel region, the OpenMP runtime atomically sums all private copies into the shared variable.
