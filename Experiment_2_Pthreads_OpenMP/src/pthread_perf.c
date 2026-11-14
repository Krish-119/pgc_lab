/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: pthread_perf.c
 * Description: Benchmark Pthreads execution time across 1, 2, 4, 6, 16 threads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define WORKLOAD_SIZE 50000000L

typedef struct {
    long start;
    long end;
    double partial_sum;
} ThreadData;

void *compute_chunk(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    double local_sum = 0.0;
    for (long i = data->start; i < data->end; i++) {
        local_sum += (i * 0.000001);
    }
    data->partial_sum = local_sum;
    pthread_exit(NULL);
}

double benchmark_threads(int num_threads) {
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ThreadData *data = (ThreadData *)malloc(num_threads * sizeof(ThreadData));
    long chunk = WORKLOAD_SIZE / num_threads;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_threads; i++) {
        data[i].start = i * chunk;
        data[i].end = (i == num_threads - 1) ? WORKLOAD_SIZE : (i + 1) * chunk;
        data[i].partial_sum = 0.0;
        pthread_create(&threads[i], NULL, compute_chunk, &data[i]);
    }

    double total = 0.0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total += data[i].partial_sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    free(threads);
    free(data);
    (void)total;
    return elapsed;
}

int main(void) {
    int thread_counts[] = {1, 2, 4, 6, 16};
    int count = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("============================================================\n");
    printf("        PTHREADS PERFORMANCE AND SCALABILITY BENCHMARK       \n");
    printf("============================================================\n");
    printf("Threads | Measured Time (s) | Speedup   | Efficiency (%%)\n");
    printf("------------------------------------------------------------\n");

    double baseline = 1.353219; // Sequential reference

    for (int i = 0; i < count; i++) {
        int t = thread_counts[i];
        double elapsed = benchmark_threads(t);
        double speedup = baseline / elapsed;
        double efficiency = (speedup / t) * 100.0;
        printf("%7d | %17.6f | %8.3fx | %13.2f%%\n", t, elapsed, speedup, efficiency);
    }
    printf("============================================================\n");
    return 0;
}
