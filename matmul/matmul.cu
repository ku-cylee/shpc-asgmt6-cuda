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
#define ITERATION     16

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

static int Mbegin[ITERATION][NGPU], Mend[ITERATION][NGPU];
static int ngpu;
static cudaStream_t streams[NGPU];
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];

void matmul(float *A, float *B, float *C, int M, int N, int K) {

  int M_per_iter = M / ITERATION;
  int M_per_iter_node = M_per_iter / mpi_world_size;

  MPI_Request scatter_req, gather_req;
  MPI_Iscatter(
    A, M_per_iter_node * K, MPI_FLOAT,
    A, M_per_iter_node * K, MPI_FLOAT,
    0, MPI_COMM_WORLD, &scatter_req);
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  for (int iter = 0; iter < ITERATION; iter++) {
    MPI_Wait(&scatter_req, MPI_STATUS_IGNORE);

    // Async memcpy H->D on each GPU
    for (int i = 0; i < ngpu; i++) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &A[Mbegin[iter][i] * K],
                                (Mend[iter][i] - Mbegin[iter][i]) * K * sizeof(float),
                                cudaMemcpyHostToDevice, streams[i]));
      if (iter == 0) {
        CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], B, K * N * sizeof(float),
                                  cudaMemcpyHostToDevice, streams[i]));
      }
    }

    if (iter + 1 != ITERATION) {
      MPI_Iscatter(
        A + (iter + 1) * M_per_iter * K, M_per_iter_node * K, MPI_FLOAT,
        A + (iter + 1) * M_per_iter * K, M_per_iter_node * K, MPI_FLOAT,
        0, MPI_COMM_WORLD, &scatter_req);
    }

    // Run kernels asynchronously on each GPU
    for (int i = 0; i < ngpu; i++) {
      CHECK_CUDA(cudaSetDevice(i));
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (Mend[iter][i] - Mbegin[iter][i] + TILE_SIZE - 1) / TILE_SIZE);
      matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(
          A_gpu[i], B_gpu[i], C_gpu[i], Mend[iter][i] - Mbegin[iter][i], N, K);
      CHECK_CUDA(cudaGetLastError());
    }

    if (iter != 0) {
      MPI_Wait(&gather_req, MPI_STATUS_IGNORE);
    }

    // Async memcpy D->H on each GPU
    for (int i = 0; i < ngpu; i++) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaMemcpyAsync(&C[Mbegin[iter][i] * N], C_gpu[i],
                                (Mend[iter][i] - Mbegin[iter][i]) * N * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all async jobs to finish
    for (int i = 0; i < ngpu; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(streams[i]);
    }

    MPI_Igather(
      C + iter * M_per_iter * N, M_per_iter_node * N, MPI_FLOAT,
      C + iter * M_per_iter * N, M_per_iter_node * N, MPI_FLOAT,
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

  int M_per_iter = M / ITERATION;
  int M_per_node = M_per_iter / mpi_world_size;
  int M_per_gpu = M_per_node / ngpu;

  for (int iter = 0; iter < ITERATION; iter++) {
    for (int i = 0; i < ngpu; i++) {
      Mbegin[iter][i] = iter * M_per_iter + M_per_gpu * i;
      Mend[iter][i] = Mbegin[iter][i] + M_per_gpu;
      // if (i == ngpu - 1) Mend[iter][i] = M_per_node;
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
