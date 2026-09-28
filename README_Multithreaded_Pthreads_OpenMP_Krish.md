# Multithreaded Programming Using Pthreads and OpenMP

## Experiment

**Develop Multithreaded Programs Using Parallel Programming Libraries to Understand Thread Creation, Management, and Coordination**

### Aim

To develop multithreaded programs using Pthreads and OpenMP and understand:

- Thread creation
- Thread management
- Work distribution
- Race conditions
- Synchronization
- Thread coordination
- Performance improvement using multiple threads

## Basic Idea

A thread is an execution path inside a program. A multithreaded program uses multiple threads so that different parts of a problem can be processed concurrently.

This experiment uses two technologies:

- **Pthreads** — explicit thread creation and management.
- **OpenMP** — a higher-level programming model using compiler directives.

## Software Environment

The experiment document uses:

- Windows
- WSL Ubuntu
- GCC compiler
- Pthreads
- OpenMP
- Nano editor

## Part A — Pthreads

### 1. Create One Thread

The program uses:

```c
pthread_create()
pthread_join()
```

`pthread_create()` creates an additional thread and `pthread_join()` makes the main thread wait for it to finish.

Expected output:

```text
Hello from the thread!
Main thread finished.
```

### 2. Create Multiple Threads

The experiment creates four additional threads using `pthread_create()`.

Expected output can be:

```text
Hello from Thread 1
Hello from Thread 2
Hello from Thread 4
Hello from Thread 3
All threads have finished.
```

The order of the thread messages may be different because the operating system schedules the threads.

### 3. Divide Work Among Threads

The experiment divides an array into four parts and assigns the work to different threads.

Array:

```text
10 20 30 40 50 60 70 80
```

The expected total is:

```text
360
```

### 4. Race Condition

Multiple threads modifying shared data without coordination can produce an incorrect result. This is called a **race condition**.

### 5. Fixing Race Condition Using Mutex

A mutex protects the critical section using:

```c
pthread_mutex_lock()
pthread_mutex_unlock()
```

Expected result from the document:

```text
Expected counter = 400000
Actual counter = 400000
```

## Part B — OpenMP

OpenMP provides a higher-level approach using directives such as:

```c
#pragma omp parallel
#pragma omp parallel for
#pragma omp critical
#pragma omp barrier
```

### 1. Basic Parallel Region

OpenMP can identify the current thread with:

```c
omp_get_thread_num()
```

and the total number of threads with:

```c
omp_get_num_threads()
```

The document's example shows output such as:

```text
Hello from Thread 18 of 32
Hello from Thread 13 of 32
Hello from Thread 19 of 32
...
Hello from Thread 0 of 32
```

The exact order can change.

### 2. OpenMP Work Sharing

The directive:

```c
#pragma omp parallel for
```

divides loop iterations among available threads.

The example array is:

```text
10 20 30 40 50 60 70 80
```

Expected result:

```text
Total sum = 360
```

### 3. OpenMP Race Condition

The experiment demonstrates that OpenMP does not automatically make every shared-data operation safe.

A race condition can occur when multiple threads modify:

```c
counter++;
```

### 4. OpenMP Critical Section

The race condition can be controlled using:

```c
#pragma omp critical
```

This restricts the protected section so that only one thread at a time executes it.

## Performance Analysis

The document records the following measured execution times.

| Threads | Pthreads Time (s) | OpenMP Time (s) |
|---:|---:|---:|
| 1 | 1.348142 | 1.409294 |
| 2 | 0.680737 | 0.715560 |
| 4 | 0.358872 | 0.360803 |
| 6 | 0.241345 | 0.241608 |
| 16 | 0.144812 | 0.140692 |

Sequential baseline:

```text
1.353219 seconds
```

### Speedup

The document records:

| Threads | Pthreads Speedup | OpenMP Speedup |
|---:|---:|---:|
| 1 | 1.004x | 0.960x |
| 2 | 1.988x | 1.891x |
| 4 | 3.771x | 3.751x |
| 6 | 5.608x | 5.601x |
| 16 | 9.345x | 9.618x |

### Efficiency

| Threads | Pthreads Efficiency | OpenMP Efficiency |
|---:|---:|---:|
| 1 | 100.38% | 96.02% |
| 2 | 99.39% | 94.56% |
| 4 | 94.27% | 93.76% |
| 6 | 93.45% | 93.35% |
| 16 | 58.40% | 60.11% |

## Why Does 16 Threads Not Give 16x Speedup?

The document explains that real parallel programs have overhead, including:

- Thread management
- Scheduling
- Synchronization
- Memory access
- Operating-system activity
- Non-parallel work

Therefore, adding more threads can reduce execution time, but speedup is not perfectly linear.

## Quick Difference — Pthreads vs OpenMP

| Concept | Pthreads | OpenMP |
|---|---|---|
| Create threads | `pthread_create()` | `#pragma omp parallel` |
| Wait for threads | `pthread_join()` | OpenMP runtime handles completion |
| Work distribution | Programmer explicitly divides work | `parallel for` can distribute loop iterations |
| Protect shared data | Mutex | `critical` |
| Coordination | Join / synchronization mechanisms | Barrier |
| Combine partial results | Programmer-managed | Reduction |

## Important Terms

**Thread:** A path of execution inside a program.

**Main thread:** The thread that starts executing `main()`.

**Additional thread:** A new thread created by the program.

**Multithreading:** Using multiple threads within one program.

**Parallel programming:** Dividing work so that multiple execution units can perform parts of the work concurrently.

**Work distribution:** Dividing one large task into smaller tasks and assigning them to different threads.

**Race condition:** A situation where multiple threads access or modify shared data without proper coordination.

**Mutex:** A locking mechanism used in Pthreads to protect a critical section.

**Critical section:** A section of code where simultaneous execution by multiple threads must be restricted.

**Barrier:** A synchronization point where threads wait until all required threads reach the same point.

**Speedup:** How much faster the parallel program is compared with the sequential baseline.

**Efficiency:** How effectively the available threads produce the measured speedup.

## Implemented Programs

### Pthreads

- `thread1.c` — Create one thread
- `thread2.c` — Create multiple threads
- `thread_sum.c` — Divide work among threads
- `race.c` — Demonstrate race condition
- `mutex.c` — Fix race condition using mutex
- `pthread_perf.c` — Measure performance

### OpenMP

- `omp1.c` — Parallel region and thread identification
- `omp_sum.c` — Work sharing and reduction
- `omp_race.c` — Demonstrate race condition
- `omp_critical.c` — Synchronization using critical
- `omp_barrier.c` — Thread coordination
- `omp_perf.c` — Measure performance

## Conclusion

The experiment demonstrates how multithreaded programs can be developed using Pthreads and OpenMP.

Pthreads provides explicit control over thread creation, joining, and mutex-based synchronization. OpenMP provides a higher-level programming model using parallel regions, work-sharing directives, critical sections, barriers, and reductions.

The experiments also demonstrate that multiple threads can introduce race conditions when shared data is not protected.

## Author

**Krish-119**

Parallel Computing Laboratory
