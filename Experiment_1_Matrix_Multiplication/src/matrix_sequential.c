/*
 * Experiment 1: Parallel Matrix Multiplication
 * File: matrix_sequential.c
 * Description: Baseline sequential matrix multiplication for 4000 x 4000 matrices.
 * Verification: C[0][0] == 4000.00
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4000

int main(void) {
    printf("============================================================\n");
    printf("   SEQUENTIAL MATRIX MULTIPLICATION (BASELINE CPU)          \n");
    printf("============================================================\n");
    printf("Matrix Dimensions : %d x %d\n", N, N);
    printf("Allocating memory for matrices A, B, and C...\n");

    // Dynamically allocate contiguous row-major memory
    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = (double *)malloc((size_t)N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    printf("Initializing matrices (A = 1.0, B = 1.0, C = 0.0)...\n");
    for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    printf("Starting sequential matrix multiplication...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\n--- RESULTS ---\n");
    printf("Sequential Execution Time : %.6f seconds\n", elapsed);
    printf("Verification C[0][0]       : %.2f (Expected: %.2f)\n", C[0], (double)N);
    printf("Status                     : %s\n", (C[0] == (double)N) ? "PASSED" : "FAILED");
    printf("============================================================\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
