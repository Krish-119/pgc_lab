/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: omp_barrier.c
 * Description: Demonstrating phase coordination using #pragma omp barrier.
 */

#include <stdio.h>
#include <omp.h>

int main(void) {
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("Thread %d starting phase 1...\n", tid);

        #pragma omp barrier

        printf("Thread %d passed barrier into phase 2!\n", tid);
    }

    printf("All threads completed synchronized phases.\n");
    return 0;
}
