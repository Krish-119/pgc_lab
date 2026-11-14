/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: omp_race.c
 * Description: Demonstrates race condition in OpenMP without critical section.
 */

#include <stdio.h>
#include <omp.h>

#define ITERATIONS 1000000

int main(void) {
    long counter = 0;
    omp_set_num_threads(4);

    #pragma omp parallel for
    for (int i = 0; i < ITERATIONS; i++) {
        counter++;
    }

    printf("=== OpenMP Race Condition ===\n");
    printf("Expected counter: %d\n", ITERATIONS);
    printf("Actual counter  : %ld\n", counter);
    if (counter != ITERATIONS) {
        printf("Observation: Data race caused lost updates!\n");
    }
    return 0;
}
