#include "brute_force.h"
#include "cell_list.h"
#include "../include/utils.h"
#include <algorithm>
#include <chrono>
#include <random>

class Test {
public:
    Test(double2 domainSize, double cutoff, int N, unsigned seed) :
        cellList_{domainSize, cutoff},
        interaction_{cutoff, /* alpha */ 0.2},
        N_{N}
    {
        CUDA_CHECK(cudaMalloc(&pDev_, N * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&pSortedDev_, N * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&fComputedDev_, N * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&fExpectedDev_, N * sizeof(double2)));
        CUDA_CHECK(cudaMallocHost(&pHost_, N * sizeof(double2)));
        CUDA_CHECK(cudaMallocHost(&pSortedHost_, N * sizeof(double2)));
        CUDA_CHECK(cudaMallocHost(&fComputedHost_, N * sizeof(double2)));
        CUDA_CHECK(cudaMallocHost(&fExpectedHost_, N * sizeof(double2)));

        // Generate N positions uniformly randomly distributed throughout the
        // domain. Note that this might produce clusters of high density and
        // clusters of low density, which is not entirely representative of typical
        // physical systems such as those appearing in e.g. Molecular Dynamics.
        std::mt19937 gen{seed};
        std::uniform_real_distribution<double> distr{0.0, 1.0};
        for (int i = 0; i < N; ++i) {
            pHost_[i].x = distr(gen) * domainSize.x;
            pHost_[i].y = distr(gen) * domainSize.y;
        }

        // Upload initial state to the GPU.
        CUDA_CHECK(cudaMemcpy(pDev_, pHost_, N * sizeof(double2), cudaMemcpyHostToDevice));
    }

    ~Test() {
        CUDA_CHECK(cudaFreeHost(fExpectedHost_));
        CUDA_CHECK(cudaFreeHost(fComputedHost_));
        CUDA_CHECK(cudaFreeHost(pSortedHost_));
        CUDA_CHECK(cudaFreeHost(pHost_));
        CUDA_CHECK(cudaFree(fExpectedDev_));
        CUDA_CHECK(cudaFree(fComputedDev_));
        CUDA_CHECK(cudaFree(pSortedDev_));
        CUDA_CHECK(cudaFree(pDev_));
    }

    void buildAndTestCellList() {
        cellList_.build(pDev_, pSortedDev_, N_);

        CUDA_CHECK(cudaMemcpy(pSortedHost_, pSortedDev_, N_ * sizeof(double2), cudaMemcpyDeviceToHost));
        {
            std::vector<double2> p(pHost_, pHost_ + N_);
            std::vector<double2> pSorted(pSortedHost_, pSortedHost_ + N_);
            auto cmp = [](double2 p, double2 q) {
                return p.x < q.x || (p.x == q.x && p.y < q.y);
            };
            std::sort(p.begin(), p.end(), cmp);
            std::sort(pSorted.begin(), pSorted.end(), cmp);

            bool equal = true;
            for (int i = 0; i < N_; ++i) {
                if (p[i].x != pSorted[i].x || p[i].y != pSorted[i].y) {
                    equal = false;
                    break;
                }
            }
            if (!equal) {
                printf("Cell list did not produce correct set of positions.\n");
                printf("Expected (lexicographically sorted):\n");
                for (int i = 0; i < N_; ++i)
                    printf("    %g %g\n", p[i].x, p[i].y);
                printf("Computed (lexicographically sorted):\n");
                for (int i = 0; i < N_; ++i)
                    printf("    %g %g\n", pSorted[i].x, pSorted[i].y);
                exit(1);
            }
        }
    }

    void runAndCompare() {
        computeForces(cellList_.getInfo(), interaction_, pSortedDev_, fComputedDev_, N_);
        // Brute force must be evaluated for sorted particles as well, to have
        // expected and computed forces in the same order.
        computeForcesSlow(interaction_, pSortedDev_, fExpectedDev_, N_);
        CUDA_CHECK(cudaMemcpy(fComputedHost_, fComputedDev_, N_ * sizeof(double2), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(fExpectedHost_, fExpectedDev_, N_ * sizeof(double2), cudaMemcpyDeviceToHost));
        bool correct = true;
        for (int i = 0; i < N_; ++i) {
            const double dfx = fComputedHost_[i].x - fExpectedHost_[i].x;
            const double dfy = fComputedHost_[i].y - fExpectedHost_[i].y;
            if (dfx * dfx + dfy * dfy > 1e-6) {
                correct = false;
                printf("incorrect force for i=%d:  expected=(%g %g)  computed=(%g %g)\n",
                       i,
                       fExpectedHost_[i].x,
                       fExpectedHost_[i].y,
                       fComputedHost_[i].x,
                       fComputedHost_[i].y);
            }
        }
        if (!correct)
            exit(1);
    }

    double benchmark(int fast, int repeat) {
        auto compute = [this, fast] {
            if (fast) {
                computeForces(cellList_.getInfo(), interaction_, pSortedDev_, fComputedDev_, N_);
            } else {
                computeForcesSlow(interaction_, pSortedDev_, fExpectedDev_, N_);
            }
        };

        // Warmup.
        for (int step = 0; step < 1; ++step)
            compute();

        // Benchmark.
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now();
        for (int step = 0; step < repeat; ++step)
            compute();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        double seconds_per_ts = 1e-9 * ns / repeat;
        return seconds_per_ts;
    }

private:
    CellList cellList_;
    const Interaction interaction_;

    const int N_;
    double2 *pDev_;
    double2 *pSortedDev_;
    double2 *fComputedDev_;
    double2 *fExpectedDev_;
    double2 *pHost_;
    double2 *pSortedHost_;
    double2 *fComputedHost_;
    double2 *fExpectedHost_;
};

enum {
    NoBenchmark,
    BenchmarkFast,
    BenchmarkBoth,
};

static void test(double2 domainSize, double cutoff, int N, unsigned seed,
                 bool shouldTest, int benchmark) {
    const double density = N / (domainSize.x * domainSize.y);
    printf("Running %gx%g domain with cutoff %g and %d particles (%.2f per cell)\n",
           domainSize.x, domainSize.y, cutoff, N, density * cutoff * cutoff);
    Test test{domainSize, cutoff, N, seed};
    test.buildAndTestCellList();
    if (shouldTest) {
        test.runAndCompare();
        printf("    OK!\n");
    }
    if (benchmark == BenchmarkFast || benchmark == BenchmarkBoth) {
        printf("    Benchmarking...\n");
        const double tFast = test.benchmark(true, 10);
        const double usefulInteractions = density * M_PI * cutoff * cutoff * N;
        printf("    fast=%.1f ms  iterations/s=%.2gG (counting pairs within cutoff only)\n",
               tFast * 1e3, usefulInteractions / tFast / 1e9);
        if (benchmark == BenchmarkBoth) {
            const double tSlow = test.benchmark(false, 3);
            const double complexitySlow = (double)N * N;
            printf("    slow=%.1f ms  speedup=%g\n",
                   tSlow * 1e3, tSlow / tFast);
        }
    }
}


static void testAll() {
    test({10.0, 10.0}, 10.0, 5, 12345, true, NoBenchmark);
    test({30.0, 30.0}, 10.0, 5, 12346, true, NoBenchmark);
    test({30.0, 30.0}, 10.0, 50, 12347, true, NoBenchmark);
    test({100.0, 50.0}, 10.0, 50000, 12348, true, BenchmarkBoth);
    test({1000.0, 100.0}, 10.0, 10000, 12349, true, BenchmarkBoth);
    test({1000.0, 1000.0}, 10.0, 100000, 12350, true, BenchmarkFast);
    test({1000.0, 1000.0}, 10.0, 1000000, 12351, false, BenchmarkFast);
    test({1000.0, 10000.0}, 10.0, 1000000, 12352, false, BenchmarkFast);
}

int main() {
    testAll();
}
