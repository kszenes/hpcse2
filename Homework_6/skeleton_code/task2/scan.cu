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

__global__ void inscanBlock(const int *inDev, int *outDev, int N) {
    int idx  = threadIdx.x;
    int a = idx < N ? inDev[idx] : 0.0;
    a = inscanWarp(a);
    outDev[idx] = a;
}

Scan::Scan() { }

Scan::~Scan() {
    // TODO: Deallocate any temporary buffers.
}

void Scan::inclusiveSum(const int *inDev, int *outDev, int N) {
    (void)inDev;  // do not complain about unused variables...
    (void)outDev;
    (void)N;



    // TODO: Implement kernels and launching the kernels.
    inscanBlock<<<1, N>>>(inDev, outDev, N);

}

