/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: mutex.c
 * Description: Fixing the race condition using pthread_mutex_t locks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 4
#define ITERATIONS_PER_THREAD 100000

long counter = 0;
pthread_mutex_t lock;

void *worker(void *arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        pthread_mutex_lock(&lock);
        counter++;
        pthread_mutex_unlock(&lock);
    }
    pthread_exit(NULL);
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    long expected = (long)NUM_THREADS * ITERATIONS_PER_THREAD;

    pthread_mutex_init(&lock, NULL);

    printf("=== Pthreads Mutex Protection ===\n");
    printf("Threads: %d | Iterations per thread: %d\n", NUM_THREADS, ITERATIONS_PER_THREAD);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker, NULL);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Expected counter = %ld\n", expected);
    printf("Actual counter   = %ld\n", counter);
    printf("Status           = %s\n", (counter == expected) ? "CORRECT" : "INCORRECT");

    pthread_mutex_destroy(&lock);
    return (counter == expected) ? EXIT_SUCCESS : EXIT_FAILURE;
}
