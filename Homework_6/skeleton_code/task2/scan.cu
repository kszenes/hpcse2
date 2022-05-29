#include "scan.h"
#include "../include/utils.h"
#include <algorithm>
#include <cstdio>

#define FULL_MASK 0xFFFFFFFF

__device__ int inscanWarp(int a) {
    int laneIdx = threadIdx.x & 31;
    int b;
    b = __shfl_up_sync(FULL_MASK, a, 1);
    if (laneIdx > 0) {
        a += b;
    } 
    b = __shfl_up_sync(FULL_MASK, a, 2);
    if (laneIdx > 1) {
        a += b;
    } 
    b = __shfl_up_sync(FULL_MASK, a, 4);
    if (laneIdx > 3) {
        a += b;
    } 
    b = __shfl_up_sync(FULL_MASK, a, 8);
    if (laneIdx > 7) {
        a += b;
    } 
    b = __shfl_up_sync(FULL_MASK, a, 16);
    if (laneIdx > 15) {
        a += b;
    } 
    return a;

}

__global__ void inscanBlock(const int *inDev, int *outDev, int* tmp, int N) {
    const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int laneIdx  = threadIdx.x & 31;
    const int warpIdx  = threadIdx.x >> 5;
    /* int a = idx < N ? inDev[idx] : 0.0; */
    if (idx < N) {
        int a = inDev[idx];
        a = inscanWarp(a);

        __shared__ int warpInscans[32];
        // Last thread of each warp loads value into shared memory
        if (laneIdx == 31) {
            warpInscans[warpIdx] = a;
        }
        __syncthreads();

        // First warp computes inscan on loaded values
        if (threadIdx.x < 32) {
            warpInscans[threadIdx.x] = inscanWarp(warpInscans[threadIdx.x]);
        }
        __syncthreads();

        // Broadcast inscanned values to warps > 0
        if (warpIdx > 0) {
            a += warpInscans[warpIdx-1];
        }

        if (threadIdx.x == 1023 && tmp != nullptr) {
            tmp[blockIdx.x] = a;
            /* printf("\ntmp[%d] = %d\n", blockIdx.x, a); */
        }

        outDev[idx] = a;
    }
}

__global__ void addBlocks(const int *inDev, int *out, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* if (threadIdx.x == 0) { */
    /*     printf("\ntmp[%d] = %d\n", blockIdx.x, inDev[blockIdx.x]); */
    /* } */
    if (idx < N) {
        out[idx] +=  inDev[blockIdx.x];
    }
}

Scan::Scan() { }

Scan::~Scan() {
    // TODO: Deallocate any temporary buffers.
    CUDA_CHECK(cudaFree(tmp));
}

void Scan::inclusiveSum(const int *inDev, int *outDev, int N) {
    (void)inDev;  // do not complain about unused variables...
    (void)outDev;
    (void)N;

    int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;


    if (N <= 1024) {
        inscanBlock<<<numBlocks, numThreads>>>(inDev, outDev, tmp, N);
    } else {
        CUDA_CHECK(cudaMalloc(&tmp, numBlocks * sizeof(int)));
        inscanBlock<<<numBlocks, numThreads>>>(inDev, outDev, tmp, N);
        inscanBlock<<<1, numThreads>>>(tmp, tmp, nullptr, numBlocks);
        addBlocks<<<numBlocks-1, numThreads>>>(tmp, outDev + numThreads, N - numThreads);
    }


}

