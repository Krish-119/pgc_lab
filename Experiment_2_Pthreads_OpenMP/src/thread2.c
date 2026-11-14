/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: thread2.c
 * Description: Creating and managing multiple threads (4 threads) with Pthreads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 4

void *thread_function(void *arg) {
    long tid = (long)arg;
    printf("Hello from Thread %ld\n", tid);
    pthread_exit(NULL);
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    int status;

    printf("Creating %d threads...\n", NUM_THREADS);
    for (long i = 0; i < NUM_THREADS; i++) {
        status = pthread_create(&threads[i], NULL, thread_function, (void *)(i + 1));
        if (status != 0) {
            fprintf(stderr, "Error creating thread %ld\n", i);
            return EXIT_FAILURE;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads have finished.\n");
    return EXIT_SUCCESS;
}
