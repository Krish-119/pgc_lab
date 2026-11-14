CC = gcc
CFLAGS = -O2 -Wall
OMP_FLAGS = -fopenmp
MPICC = mpicc
NVCC = nvcc
CUDA_ARCH = -arch=sm_86

all: matrix_sequential matrix_openmp

matrix_sequential: src/matrix_sequential.c
	$(CC) $(CFLAGS) -o $@ $<

matrix_openmp: src/matrix_openmp.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

matrix_mpi: src/matrix_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $<

matrix_cuda: src/matrix_cuda.cu
	$(NVCC) -O2 $(CUDA_ARCH) -o $@ $<

clean:
	rm -f matrix_sequential matrix_openmp matrix_mpi matrix_cuda matrix_cuda.exe

.PHONY: all clean
