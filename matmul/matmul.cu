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

#define GPU_THREADS   32
#define NUM_CYCLES    16

static int mpi_rank, mpi_world_size;

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int local_row = threadIdx.y;
  int local_col = threadIdx.x;

  int global_row = GPU_THREADS * blockIdx.y + local_row;
  int global_col = GPU_THREADS * blockIdx.x + local_col;

  __shared__ float A_tile[GPU_THREADS][GPU_THREADS];
  __shared__ float B_tile[GPU_THREADS][GPU_THREADS];

  float sum = 0.0f;
  for (int k = 0; k < K; k += GPU_THREADS) {
    A_tile[local_row][local_col] = A[global_row * K + local_col + k];
    B_tile[local_row][local_col] = B[(local_row + k) * N + global_col];

    __syncthreads();

    for (int t = 0; t < GPU_THREADS; t++) {
      sum += A_tile[local_row][t] * B_tile[t][local_col];
    }

    __syncthreads();
  }

  C[global_row * N + global_col] = sum;
}

#define NGPU 4

static int Mbegin[NUM_CYCLES][NGPU], Mend[NUM_CYCLES][NGPU];
static int ngpu;
static cudaStream_t streams[NGPU];
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];

void matmul(float *A, float *B, float *C, int M, int N, int K) {

  int M_per_cycle = M / NUM_CYCLES;
  int M_per_cycle_node = M_per_cycle / mpi_world_size;

  MPI_Request scatter_req, gather_req;
  
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], B, K * N * sizeof(float),
                              cudaMemcpyHostToDevice, streams[i]));
  }

  MPI_Iscatter(
    A, M_per_cycle_node * K, MPI_FLOAT,
    A, M_per_cycle_node * K, MPI_FLOAT,
    0, MPI_COMM_WORLD, &scatter_req);

  for (int cycle = 0; cycle < NUM_CYCLES; cycle++) {
    MPI_Wait(&scatter_req, MPI_STATUS_IGNORE);

    // Async memcpy H->D on each GPU
    for (int i = 0; i < ngpu; i++) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &A[Mbegin[cycle][i] * K],
                                (Mend[cycle][i] - Mbegin[cycle][i]) * K * sizeof(float),
                                cudaMemcpyHostToDevice, streams[i]));
    }

    if (cycle + 1 != NUM_CYCLES) {
      MPI_Iscatter(
        A + (cycle + 1) * M_per_cycle * K, M_per_cycle_node * K, MPI_FLOAT,
        A + (cycle + 1) * M_per_cycle * K, M_per_cycle_node * K, MPI_FLOAT,
        0, MPI_COMM_WORLD, &scatter_req);
    }

    // Run kernels asynchronously on each GPU
    for (int i = 0; i < ngpu; i++) {
      CHECK_CUDA(cudaSetDevice(i));
      dim3 blockDim(GPU_THREADS, GPU_THREADS);
      dim3 gridDim((N + GPU_THREADS - 1) / GPU_THREADS, (Mend[cycle][i] - Mbegin[cycle][i] + GPU_THREADS - 1) / GPU_THREADS);
      matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(
          A_gpu[i], B_gpu[i], C_gpu[i], Mend[cycle][i] - Mbegin[cycle][i], N, K);
      CHECK_CUDA(cudaGetLastError());
    }

    if (cycle != 0) {
      MPI_Wait(&gather_req, MPI_STATUS_IGNORE);
    }

    // Async memcpy D->H on each GPU
    for (int i = 0; i < ngpu; i++) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaMemcpyAsync(&C[Mbegin[cycle][i] * N], C_gpu[i],
                                (Mend[cycle][i] - Mbegin[cycle][i]) * N * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all async jobs to finish
    for (int i = 0; i < ngpu; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(streams[i]);
    }

    MPI_Igather(
      C + cycle * M_per_cycle * N, M_per_cycle_node * N, MPI_FLOAT,
      C + cycle * M_per_cycle * N, M_per_cycle_node * N, MPI_FLOAT,
      0, MPI_COMM_WORLD, &gather_req);
  }

  MPI_Wait(&gather_req, MPI_STATUS_IGNORE);
  // MPI_Request_free(&scatter_req);
  // MPI_Request_free(&gather_req);
}


void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  CHECK_CUDA(cudaGetDeviceCount(&ngpu));

  printf("[rank %d] Number of devices: %d\n", mpi_rank, ngpu);
  cudaDeviceProp props[NGPU];
  for (int i = 0; i < ngpu; ++i) {
    CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));
    printf("[rank %d] device %d: %s\n", mpi_rank, i, props[i].name);
  }

  int M_per_cycle = M / NUM_CYCLES;
  int M_per_node = M_per_cycle / mpi_world_size;
  int M_per_gpu = M_per_node / ngpu;

  for (int cycle = 0; cycle < NUM_CYCLES; cycle++) {
    for (int i = 0; i < ngpu; i++) {
      Mbegin[cycle][i] = cycle * M_per_cycle + M_per_gpu * i;
      Mend[cycle][i] = Mbegin[cycle][i] + M_per_gpu;
      // if (i == ngpu - 1) Mend[cycle][i] = M_per_node;
    }
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(
        cudaMalloc(&A_gpu[i], M_per_gpu * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
    CHECK_CUDA(
        cudaMalloc(&C_gpu[i], M_per_gpu * N * sizeof(float)));
  }
}


void matmul_finalize() {
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(B_gpu[i]));
    CHECK_CUDA(cudaFree(C_gpu[i]));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
  }
}
