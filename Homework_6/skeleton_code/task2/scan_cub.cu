#include "scan_cub.h"
#include "../include/utils.h"
#include <cub/device/device_scan.cuh>

ScanCub::ScanCub() { }

ScanCub::~ScanCub() {
    CUDA_CHECK(cudaFree(bufferDev_));
}

void ScanCub::inclusiveSum(const int *inDev, int *outDev, int N) {
    size_t size = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, size, inDev, outDev, N));

    if (bufferSize_ < size) {
        CUDA_CHECK(cudaFree(bufferDev_));
        CUDA_CHECK(cudaMalloc(&bufferDev_, size));
        bufferSize_ = size;
    }

    CUDA_CHECK(cub::DeviceScan::InclusiveSum(bufferDev_, size, inDev, outDev, N));
}
