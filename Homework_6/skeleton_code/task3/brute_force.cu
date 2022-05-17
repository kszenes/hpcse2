#include "brute_force.h"
#include "../include/utils.h"

#include <cstdio>

__global__ void computeForcesSlowKernel(Interaction interaction, const double2 *p, double2 *f, int N) {
    // Not the fastest N^2 implementation, but good enough.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += gridDim.x * blockDim.x) {
        double2 fTotal{0.0, 0.0};
#pragma unroll 8
        for (int i = 0; i < N; ++i) {
            const double2 fCurrent = interaction(p[idx], p[i]);
            fTotal.x += fCurrent.x;
            fTotal.y += fCurrent.y;
        }
        f[idx] = fTotal;
    }
}

void computeForcesSlow(
        Interaction interaction,
        const double2 *p,
        double2 *f,
        int numParticles) {
    const int threads = 256;
    const int blocks = (numParticles + threads - 1) / threads;
    CUDA_LAUNCH(computeForcesSlowKernel, blocks, threads, interaction, p, f, numParticles);
}
