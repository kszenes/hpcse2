#include "cell_list.h"
#include "interaction.h"
#include <stdio.h>

__global__ void computeForceKernel(Interaction interaction, CellListInfo info, const double2* p, double2 *f, int N) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < N;
            idx += gridDim.x * blockDim.x) {
        double2 fTotal{0.0, 0.0};
        int2 cell = info.computeIndices(p[idx]);
        int lower_x = max(0, cell.x - 1);
        int lower_y = max(0, cell.y - 1);
        int upper_x = min(info.numCells.x - 1, cell.x + 1);
        int upper_y = min(info.numCells.y - 1, cell.y + 1);
        for (int y_idx = lower_y; y_idx <= upper_y; y_idx++) {
            int first_cell = y_idx * info.numCells.x + lower_x;
            int last_cell = y_idx * info.numCells.x + upper_x;

            int first_particle = info.offsets[first_cell];
            int last_particle = info.offsets[last_cell+1];
            if (idx == 4) {
                printf("%d; %d\n", first_particle, last_particle);
            }
            for (int particle = first_particle; particle < last_particle; particle++){
                double2 fCurrent = interaction(p[idx], p[particle]);
                fTotal.x += fCurrent.x;
                fTotal.y += fCurrent.y;
            }
        }
        f[idx] = fTotal;
    }
}

void computeForces(
        const CellListInfo info,
        Interaction interaction,
        const double2 *pSortedDev,
        double2 *f,
        int numParticles) {

    // Do not complain about unused variable. Erase once the code has been implemented.

    // TODO: Implement the optimized version of computeForcesSlow (see
    // brute_force.cu) that utilized cell lists.
    const int numThreads = 256;
    const int numBlocks = (numParticles + numThreads - 1) / numThreads; 
    computeForceKernel<<<numBlocks, numThreads>>>(interaction, info, pSortedDev, f, numParticles);
}
