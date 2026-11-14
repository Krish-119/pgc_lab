/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: race.c
 * Description: Demonstrates race condition on a shared counter without synchronization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 4
#define ITERATIONS_PER_THREAD 100000

long counter = 0;

void *worker(void *arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        counter++;
    }
    pthread_exit(NULL);
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    long expected = (long)NUM_THREADS * ITERATIONS_PER_THREAD;

    printf("=== Pthreads Race Condition Demonstration ===\n");
    printf("Threads: %d | Iterations per thread: %d\n", NUM_THREADS, ITERATIONS_PER_THREAD);
    printf("Expected final counter: %ld\n", expected);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker, NULL);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Actual final counter  : %ld\n", counter);
    if (counter != expected) {
        printf("Observation: Race condition occurred! %ld updates were lost.\n", expected - counter);
    } else {
        printf("Note: Race condition did not manifest in this particular run (timing coincidentally aligned).\n");
    }

    return 0;
}
