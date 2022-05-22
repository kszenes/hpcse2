#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 *f) {
  
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < N;
            idx += gridDim.x * blockDim.x){


        f[idx] = double3{0.0, 0.0, 0.0};

        double px = p[idx].x; 
        double py = p[idx].y;
        double pz = p[idx].z;

        double fx = f[idx].x; 
        double fy = f[idx].y;
        double fz = f[idx].z;

        for (int i = 0; i < idx; ++i) {
                double dx = p[i].x - px;
                double dy = p[i].y - py;
                double dz = p[i].z - pz;

                /* double r = sqrt(dx * dx + dy * dy + dz * dz); */
                /* double inv_r = 1.0 / r; */

                double inv_r = rsqrt(dx * dx + dy * dy + dz * dz);
                double inv_r_3 = inv_r * inv_r * inv_r;
                fx += dx * inv_r_3;
                fy += dy * inv_r_3;
                fz += dz * inv_r_3;
        }

        for (int i = idx+1; i < N; ++i) {
                double dx = p[i].x - px;
                double dy = p[i].y - py;
                double dz = p[i].z - pz;

                /* double r = sqrt(dx * dx + dy * dy + dz * dz); */
                /* double inv_r = 1.0 / r; */

                double inv_r = rsqrt(dx * dx + dy * dy + dz * dz);
                double inv_r_3 = inv_r * inv_r * inv_r;

                fx += dx * inv_r_3;
                fy += dy * inv_r_3;
                fz += dz * inv_r_3;
        }
        f[idx] = double3{fx, fy, fz};
    }
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;
    computeForcesKernel<<<numBlocks, numThreads>>>(N, p, f);
}
