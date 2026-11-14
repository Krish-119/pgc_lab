/*
 * Experiment 1: Parallel Matrix Multiplication
 * File: matrix_openmp.c
 * Description: Shared-memory parallel matrix multiplication using OpenMP.
 * Verification: C[0][0] == 4000.00
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

int main(int argc, char *argv[]) {
    int requested_threads = 12;
    if (argc > 1) {
        requested_threads = atoi(argv[1]);
    }
    omp_set_num_threads(requested_threads);

    printf("============================================================\n");
    printf("   OPENMP SHARED-MEMORY MATRIX MULTIPLICATION               \n");
    printf("============================================================\n");
    printf("Matrix Dimensions   : %d x %d\n", N, N);
    printf("OpenMP Threads Used : %d\n", requested_threads);

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = (double *)malloc((size_t)N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    printf("Starting parallel matrix multiplication with OpenMP...\n");
    double start_time = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;

    printf("\n--- RESULTS ---\n");
    printf("OpenMP Execution Time : %.6f seconds\n", elapsed);
    printf("Verification C[0][0]   : %.2f (Expected: %.2f)\n", C[0], (double)N);
    printf("Status                 : %s\n", (C[0] == (double)N) ? "PASSED" : "FAILED");
    printf("============================================================\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
