/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: omp_sum.c
 * Description: Work sharing with OpenMP parallel for and reduction.
 * Array: 10,20,30,40,50,60,70,80 -> Sum = 360
 */

#include <stdio.h>
#include <omp.h>

#define SIZE 8

int main(void) {
    int array[SIZE] = {10, 20, 30, 40, 50, 60, 70, 80};
    long total_sum = 0;

    printf("Input Array: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%d ", array[i]);
    }
    printf("\n\n");

    #pragma omp parallel for reduction(+:total_sum)
    for (int i = 0; i < SIZE; i++) {
        printf("Thread %d processing array[%d] = %d\n", omp_get_thread_num(), i, array[i]);
        total_sum += array[i];
    }

    printf("\nTotal sum = %ld (Expected: 360)\n", total_sum);
    return 0;
}
