/*
 * Experiment 1: Parallel Matrix Multiplication
 * File: matrix_mpi.c
 * Description: Distributed-memory parallel matrix multiplication using MPI.
 * Verification: C[0][0] == 4000.00
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4000

int main(int argc, char *argv[]) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (N % num_procs != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Matrix size %d must be divisible by num_procs %d\n", N, num_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int rows_per_proc = N / num_procs;

    double *A = NULL;
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = NULL;

    double *local_A = (double *)malloc((size_t)rows_per_proc * N * sizeof(double));
    double *local_C = (double *)malloc((size_t)rows_per_proc * N * sizeof(double));

    if (rank == 0) {
        printf("============================================================\n");
        printf("   MPI DISTRIBUTED-MEMORY MATRIX MULTIPLICATION             \n");
        printf("============================================================\n");
        printf("Matrix Dimensions   : %d x %d\n", N, N);
        printf("MPI Processes       : %d\n", num_procs);
        printf("Rows per Process    : %d\n", rows_per_proc);

        A = (double *)malloc((size_t)N * N * sizeof(double));
        C = (double *)malloc((size_t)N * N * sizeof(double));

        for (size_t i = 0; i < (size_t)N * N; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Broadcast entire matrix B to all worker ranks
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter chunks of matrix A (rows_per_proc rows) to all ranks
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Compute local matrix multiplication
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    // Gather computed rows into final Matrix C on rank 0
    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double elapsed = end_time - start_time;
        printf("\n--- RESULTS ---\n");
        printf("MPI Execution Time    : %.6f seconds\n", elapsed);
        printf("Verification C[0][0]  : %.2f (Expected: %.2f)\n", C[0], (double)N);
        printf("Status                : %s\n", (C[0] == (double)N) ? "PASSED" : "FAILED");
        printf("============================================================\n");

        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
