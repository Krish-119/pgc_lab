/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: omp1.c
 * Description: OpenMP basic parallel region, thread count and thread IDs.
 */

#include <stdio.h>
#include <omp.h>

int main(void) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();
        printf("Hello from Thread %d of %d\n", tid, total);
    }
    return 0;
}
