#include <cuda_runtime.h>
#include "utils.h"

#define BLOCK_SIZE 16
#define SHMEM_SIZE (16 * 16)

__global__ void sharedDgemm(
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double* const A,
    const double* const B,
    const double beta,
    double* const C)
{
//   TODO
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int row = blockIdx.x * blockDim.x + tx;
    const int col = blockIdx.y * blockDim.y + ty;

    __shared__ double a[SHMEM_SIZE];
    __shared__ double b[SHMEM_SIZE];

    double tmp = 0;
    
#pragma unroll
    for (int i = 0; i < (k / BLOCK_SIZE); i++){

      a[tx + ty * BLOCK_SIZE] = A[row + (i*BLOCK_SIZE + ty) * m];

      b[tx + ty * BLOCK_SIZE] = B[col*k + i*BLOCK_SIZE + tx];
      __syncthreads();

#pragma unroll
      for (int j = 0; j < BLOCK_SIZE; j++) {
        tmp += a[j*BLOCK_SIZE + tx] * b[ty*BLOCK_SIZE + j];
      }
      __syncthreads();

    }
    C[col*m + row] = alpha*tmp + beta*C[col*m + row];

}

void myDgemm(
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double* const A,
    const double* const B,
    const double beta,
    double* const C)
{
//  TODO
  dim3 dimGrid(k/BLOCK_SIZE, m/BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  sharedDgemm<<<dimGrid, dimBlock>>>(m, n, k, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}
