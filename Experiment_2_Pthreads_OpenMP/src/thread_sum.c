/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: thread_sum.c
 * Description: Divide array work among 4 threads (Array: 10,20,30,40,50,60,70,80 -> Sum = 360).
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define ARRAY_SIZE 8
#define NUM_THREADS 4

int array[ARRAY_SIZE] = {10, 20, 30, 40, 50, 60, 70, 80};
long partial_sums[NUM_THREADS] = {0};

void *sum_worker(void *arg) {
    long tid = (long)arg;
    int chunk_size = ARRAY_SIZE / NUM_THREADS;
    int start = tid * chunk_size;
    int end = start + chunk_size;

    long local_sum = 0;
    for (int i = start; i < end; i++) {
        local_sum += array[i];
    }
    partial_sums[tid] = local_sum;
    printf("Thread %ld processed indices [%d..%d], partial sum = %ld\n",
           tid, start, end - 1, local_sum);
    pthread_exit(NULL);
}

int main(void) {
    pthread_t threads[NUM_THREADS];

    printf("Input Array: ");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", array[i]);
    }
    printf("\n\n");

    for (long i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, sum_worker, (void *)i);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    long total_sum = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        total_sum += partial_sums[i];
    }

    printf("\nTotal sum = %ld (Expected: 360)\n", total_sum);
    return (total_sum == 360) ? EXIT_SUCCESS : EXIT_FAILURE;
}
