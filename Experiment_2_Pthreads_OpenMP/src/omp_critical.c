/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: omp_critical.c
 * Description: Eliminating race condition using #pragma omp critical.
 */

#include <stdio.h>
#include <omp.h>

#define ITERATIONS 1000000

int main(void) {
    long counter = 0;
    omp_set_num_threads(4);

    #pragma omp parallel for
    for (int i = 0; i < ITERATIONS; i++) {
        #pragma omp critical
        {
            counter++;
        }
    }

    printf("=== OpenMP Critical Section ===\n");
    printf("Expected counter: %d\n", ITERATIONS);
    printf("Actual counter  : %ld\n", counter);
    printf("Status          : %s\n", (counter == ITERATIONS) ? "CORRECT" : "INCORRECT");
    return (counter == ITERATIONS) ? 0 : 1;
}
