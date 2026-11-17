#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32 // transpose block = 32 x 32 threads
#define BLK 16  // matmul block = 16 x 16 threads

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e),       \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---------- Kernel 1: UNCOALESCED transpose (one element per thread) ----------
__global__ void transposeNaive(const float *in, float *out, int N) {
  int x = blockIdx.x * TILE + threadIdx.x; // column index
  int y = blockIdx.y * TILE + threadIdx.y; // row index
  if (x < N && y < N)
    out[x * N + y] = in[y * N + x]; // read: row-wise (good)
                                    // write: column-wise, stride N (bad)
}

// ---------- Kernel 2: COALESCED transpose (shared-memory tile) ----------
__global__ void transposeCoalesced(const float *in, float *out, int N) {
  __shared__ float tile[TILE][TILE + 1]; // +1 padding avoids bank conflicts
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < N && y < N)
    tile[threadIdx.y][threadIdx.x] = in[y * N + x]; // coalesced read

  __syncthreads(); // wait until the tile is loaded

  x = blockIdx.y * TILE + threadIdx.x; // swap block coordinates
  y = blockIdx.x * TILE + threadIdx.y;
  if (x < N && y < N)
    out[y * N + x] = tile[threadIdx.x][threadIdx.y]; // coalesced write
}

// ---------- Kernel 3: NAIVE matrix multiplication C = A x B ----------
__global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
  int row = blockIdx.y * BLK + threadIdx.y;
  int col = blockIdx.x * BLK + threadIdx.x;
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) // 2 global reads + 1 multiply-add
      sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

// ---------- Host helpers: timing with CUDA events, and result checking ----------
typedef void (*TKernel)(const float *, float *, int);

float timeTranspose(TKernel k, dim3 g, dim3 b, const float *in, float *out,
                    int N, int reps) {
  cudaEvent_t s, e;
  CHECK(cudaEventCreate(&s));
  CHECK(cudaEventCreate(&e));
  CHECK(cudaMemset(out, 0, (size_t)N * N * sizeof(float)));
  k<<<g, b>>>(in, out, N); // warm-up launch (not timed)
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaEventRecord(s));
  for (int r = 0; r < reps; r++)
    k<<<g, b>>>(in, out, N);
  CHECK(cudaEventRecord(e));
  CHECK(cudaEventSynchronize(e));

  float ms;
  CHECK(cudaEventElapsedTime(&ms, s, e));
  CHECK(cudaEventDestroy(s));
  CHECK(cudaEventDestroy(e));
  return ms / reps; // average time per launch
}

bool checkTranspose(const float *in, const float *out, int N) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      if (out[j * N + i] != in[i * N + j])
        return false;
  return true;
}

bool checkMatMul(const float *A, const float *B, const float *C, int N) {
  for (int t = 0; t < 20; t++) { // spot-check 20 random elements
    int i = rand() % N, j = rand() % N;
    double ref = 0.0;
    for (int k = 0; k < N; k++)
      ref += (double)A[i * N + k] * B[k * N + j];
    if (fabs(ref - C[i * N + j]) > 1e-3 * fabs(ref) + 1e-3)
      return false;
  }
  return true;
}

int main(int argc, char **argv) {
  int NT = 4096; // transpose size (NT x NT)
  int NM = 1024; // matmul size (NM x NM)
  if (argc > 1)
    NT = atoi(argv[1]);
  if (argc > 2)
    NM = atoi(argv[2]);

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major,
         prop.minor);

  // ================= PART A : TRANSPOSE =================
  size_t bytesT = (size_t)NT * NT * sizeof(float);
  float *h_in = (float *)malloc(bytesT);
  float *h_out = (float *)malloc(bytesT);
  for (size_t i = 0; i < (size_t)NT * NT; i++)
    h_in[i] = (float)(i % 1000);

  float *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, bytesT));
  CHECK(cudaMalloc(&d_out, bytesT));
  CHECK(cudaMemcpy(d_in, h_in, bytesT, cudaMemcpyHostToDevice));

  dim3 blockT(TILE, TILE);
  dim3 gridT((NT + TILE - 1) / TILE, (NT + TILE - 1) / TILE);

  float tN = timeTranspose(transposeNaive, gridT, blockT, d_in, d_out, NT, 20);
  CHECK(cudaMemcpy(h_out, d_out, bytesT, cudaMemcpyDeviceToHost));
  bool okN = checkTranspose(h_in, h_out, NT);

  float tC =
      timeTranspose(transposeCoalesced, gridT, blockT, d_in, d_out, NT, 20);
  CHECK(cudaMemcpy(h_out, d_out, bytesT, cudaMemcpyDeviceToHost));
  bool okC = checkTranspose(h_in, h_out, NT);

  double gbN = 2.0 * bytesT / (tN * 1e6); // read + write, GB/s
  double gbC = 2.0 * bytesT / (tC * 1e6);

  printf("\n=== PART A: Matrix transpose (N = %d) ===\n", NT);
  printf("Uncoalesced : %8.3f ms | %7.1f GB/s | %s\n", tN, gbN,
         okN ? "PASS" : "FAIL");
  printf("Coalesced   : %8.3f ms | %7.1f GB/s | %s\n", tC, gbC,
         okC ? "PASS" : "FAIL");
  printf("Speedup (uncoalesced / coalesced) = %.2fx\n", tN / tC);

  // ================= PART B : NAIVE MATMUL =================
  size_t bytesM = (size_t)NM * NM * sizeof(float);
  float *hA = (float *)malloc(bytesM);
  float *hB = (float *)malloc(bytesM);
  float *hC = (float *)malloc(bytesM);
  for (size_t i = 0; i < (size_t)NM * NM; i++) {
    hA[i] = rand() / (float)RAND_MAX;
    hB[i] = rand() / (float)RAND_MAX;
  }

  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, bytesM));
  CHECK(cudaMalloc(&dB, bytesM));
  CHECK(cudaMalloc(&dC, bytesM));
  CHECK(cudaMemcpy(dA, hA, bytesM, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, bytesM, cudaMemcpyHostToDevice));

  dim3 block(BLK, BLK);
  dim3 grid((NM + BLK - 1) / BLK, (NM + BLK - 1) / BLK);

  cudaEvent_t s, e;
  CHECK(cudaEventCreate(&s));
  CHECK(cudaEventCreate(&e));
  matMulNaive<<<grid, block>>>(dA, dB, dC, NM); // warm-up
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  int reps = 5;
  CHECK(cudaEventRecord(s));
  for (int r = 0; r < reps; r++)
    matMulNaive<<<grid, block>>>(dA, dB, dC, NM);
  CHECK(cudaEventRecord(e));
  CHECK(cudaEventSynchronize(e));

  float ms;
  CHECK(cudaEventElapsedTime(&ms, s, e));
  float tM = ms / reps;

  CHECK(cudaMemcpy(hC, dC, bytesM, cudaMemcpyDeviceToHost));
  bool okM = checkMatMul(hA, hB, hC, NM);
  double gflops = 2.0 * NM * NM * (double)NM / (tM * 1e6);

  printf("\n=== PART B: Naive matrix multiplication (N = %d) ===\n", NM);
  printf("Naive matmul: %8.3f ms | %7.1f GFLOPS | %s\n", tM, gflops,
         okM ? "PASS" : "FAIL");

  // ================= SUMMARY =================
  printf("\n=== SUMMARY ===\n");
  printf("Transpose uncoalesced : %.3f ms\n", tN);
  printf("Transpose coalesced   : %.3f ms\n", tC);
  printf("Naive matmul          : %.3f ms\n", tM);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(h_in);
  free(h_out);
  free(hA);
  free(hB);
  free(hC);
  return 0;
}
