#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

#define TILE_SIZE     32

static int mpi_rank, mpi_world_size;

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int local_row = threadIdx.y;
  int local_col = threadIdx.x;

  int global_row = TILE_SIZE * blockIdx.y + local_row;
  int global_col = TILE_SIZE * blockIdx.x + local_col;

  __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;
  for (int k = 0; k < K; k += TILE_SIZE) {
    A_tile[local_row][local_col] = A[global_row * K + local_col + k];
    B_tile[local_row][local_col] = B[(local_row + k) * N + global_col];

    __syncthreads();

    for (int t = 0; t < TILE_SIZE; t++) {
      sum += A_tile[local_row][t] * B_tile[t][local_col];
    }

    __syncthreads();
  }

  C[global_row * N + global_col] = sum;
}

#define NGPU 4

static int Mbegin, Mend;
static int ngpu;
static cudaStream_t stream;
static float *A_gpu, *B_gpu, *C_gpu;


void matmul(float *A, float *B, float *C, int M, int N, int K) {

  int M_per_node = M / mpi_world_size;

  MPI_Scatter(
    A, M_per_node * K, MPI_FLOAT,
    A, M_per_node * K, MPI_FLOAT,
    0, MPI_COMM_WORLD);
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  CHECK_CUDA(cudaMemcpyAsync(A_gpu, &A[Mbegin * K],
                              (Mend - Mbegin) * K * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(B_gpu, B, K * N * sizeof(float),
                              cudaMemcpyHostToDevice, stream));

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (Mend - Mbegin + TILE_SIZE - 1) / TILE_SIZE);
  matmul_kernel<<<gridDim, blockDim, 0, stream>>>(
      A_gpu, B_gpu, C_gpu, Mend - Mbegin, N, K);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpyAsync(&C[Mbegin * N], C_gpu,
                              (Mend - Mbegin) * N * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));

  cudaStreamSynchronize(stream);

  MPI_Gather(
    C, M_per_node * N, MPI_FLOAT,
    C, M_per_node * N, MPI_FLOAT,
    0, MPI_COMM_WORLD);
}


void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  CHECK_CUDA(cudaGetDeviceCount(&ngpu));

  int gpu_id = mpi_rank % ngpu;
  printf("[rank %d] Number of devices: %d\n", mpi_rank, ngpu);
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, gpu_id));
  printf("[rank %d] device %d: %s\n", mpi_rank, gpu_id, prop.name);

  int M_per_node = M / mpi_world_size;

  Mbegin = 0;
  Mend = M_per_node;

  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaStreamCreate(&stream));

  CHECK_CUDA(
      cudaMalloc(&A_gpu, (Mend - Mbegin) * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc(&C_gpu, (Mend - Mbegin) * N * sizeof(float)));
}


void matmul_finalize() {
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUDA(cudaStreamDestroy(stream));
}
