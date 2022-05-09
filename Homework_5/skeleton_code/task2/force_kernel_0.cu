#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 *f) {

    for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < N;
            idx += gridDim.x * blockDim.x){

        f[idx] = double3{0.0, 0.0, 0.0};
        for (int i = 0; i < N; ++i) {
            if(i != idx){
                double dx = p[i].x - p[idx].x;
                double dy = p[i].y - p[idx].y;
                double dz = p[i].z - p[idx].z;
                double r = sqrt(dx * dx + dy * dy + dz * dz);
                double inv_r = 1.0 / r;
                f[idx].x += dx * inv_r * inv_r * inv_r;
                f[idx].y += dy * inv_r * inv_r * inv_r;
                f[idx].z += dz * inv_r * inv_r * inv_r;
            }
        }
    }
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;
    computeForcesKernel<<<numBlocks, numThreads>>>(N, p, f);
}
