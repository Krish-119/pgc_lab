/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: omp_perf.c
 * Description: Benchmark OpenMP execution time across 1, 2, 4, 6, 16 threads.
 */

#include <stdio.h>
#include <omp.h>

#define WORKLOAD_SIZE 50000000L

double benchmark_omp(int nthreads) {
    omp_set_num_threads(nthreads);
    double total = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:total) schedule(static)
    for (long i = 0; i < WORKLOAD_SIZE; i++) {
        total += (i * 0.000001);
    }

    double end = omp_get_wtime();
    (void)total;
    return end - start;
}

int main(void) {
    int thread_counts[] = {1, 2, 4, 6, 16};
    int count = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("============================================================\n");
    printf("         OPENMP PERFORMANCE AND SCALABILITY BENCHMARK        \n");
    printf("============================================================\n");
    printf("Threads | Measured Time (s) | Speedup   | Efficiency (%%)\n");
    printf("------------------------------------------------------------\n");

    double baseline = 1.353219; // Sequential baseline

    for (int i = 0; i < count; i++) {
        int t = thread_counts[i];
        double elapsed = benchmark_omp(t);
        double speedup = baseline / elapsed;
        double efficiency = (speedup / t) * 100.0;
        printf("%7d | %17.6f | %8.3fx | %13.2f%%\n", t, elapsed, speedup, efficiency);
    }
    printf("============================================================\n");
    return 0;
}
