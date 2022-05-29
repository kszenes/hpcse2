#include "cell_list.h"
#include "interaction.h"
#include "../include/utils.h"

CellList::CellList(double2 domainSize, double cellSize) {
    numCells_.x = (int)std::ceil(domainSize.x / cellSize);
    numCells_.y = (int)std::ceil(domainSize.y / cellSize);
    invCellSize_.x = numCells_.x / domainSize.x;
    invCellSize_.y = numCells_.y / domainSize.y;

    const int totCells = numCells_.x * numCells_.y;
    // TODO: Allocate buffers. Note that offsets array needs 1 element more
    // than the counts array!
    CUDA_CHECK(cudaMalloc(&countsDev_, totCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&offsetsDev_, (totCells + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(countsDev_, 0, totCells * sizeof(int)));
    CUDA_CHECK(cudaMemset(offsetsDev_, 0, 1 * sizeof(int)));
}

CellList::~CellList() {
    // TODO: Deallocate buffers.
    CUDA_CHECK(cudaFree(countsDev_));
    CUDA_CHECK(cudaFree(offsetsDev_));
}


__global__ void computeCellSizes(
    const double2* pDev, int* countsDev,
    CellListInfo info, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        int cell = info.computeIndex(pDev[idx]);
        atomicAdd(&countsDev[cell], 1);
    }
} 

__global__ void computeSortedList(
   const double2* pDev, double2* pSortedDev,
   int* offsetsDev, int* countsDev, CellListInfo info, int numParticles) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   __syncthreads();
   if (idx < numParticles) {
        int cell = info.computeIndex(pDev[idx]);
        int pos = offsetsDev[cell] + atomicAdd(&countsDev[cell], 1);
        pSortedDev[pos] = pDev[idx];
   }

}

/* __global__ void printOffset(int* offset) { */
/*     printf("Offset[%d] = %d\n", threadIdx.x, offset[threadIdx.x]); */
/* } */

void CellList::build(const double2 *pDev, double2 *pSortedDev, int numParticles) {
    // Do not complain about unused variable. Erase once the code has been implemented.

    const CellListInfo info = getInfo();

    const int numThreads = 256;
    const int numBlocks = (numParticles + numThreads - 1) / numThreads;

    // Stage 1: compute cell sizes
    // TODO: Compute cell sizes (number of particles per cell).
    computeCellSizes<<<numBlocks, numThreads>>>(pDev, countsDev_, info, numParticles);


    // Stage 2: compute offsets
    // TODO: Compute offsets using the Scan class (use the scan_ member variable).
    scan_.inclusiveSum(countsDev_, offsetsDev_ + 1, numCells_.x * numCells_.y);

    /* printOffset<<<1, 2>>>(offsetsDev_); */

    CUDA_CHECK(cudaMemset(countsDev_, 0, numCells_.x * numCells_.y * sizeof(int)));
    // Stage 3: reorder particles into cells
    // TODO: Reorder the particles in the sorted order and save them to pSortedDev.
    computeSortedList<<<numBlocks, numThreads>>>(pDev, pSortedDev, offsetsDev_, countsDev_, info, numParticles);
}
