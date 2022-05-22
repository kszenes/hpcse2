#include "scan.h"
#include "scan_cub.h"
#include "../include/utils.h"
#include <chrono>
#include <random>

struct CmdlineArgs {
    bool verbose = false;
    bool warp = false;
    bool block = false;
    bool medium = false;
    bool large = false;
    bool profile = false;
};

static void test(int N, bool verbose) {
    printf("Testing with N=%d...\n", N);
    fflush(stdout);

    int *inDev;
    int *outDev;
    int *inHost;
    int *outHost;
    CUDA_CHECK(cudaMalloc(&inDev, N * sizeof(inDev[0])));
    CUDA_CHECK(cudaMalloc(&outDev, N * sizeof(outDev[0])));
    CUDA_CHECK(cudaMallocHost(&inHost, N * sizeof(inHost[0])));
    CUDA_CHECK(cudaMallocHost(&outHost, N * sizeof(outHost[0])));

    const int maxValue = 200;
    if ((long long)maxValue * N > (1LL << 31)) {
        fprintf(stderr, "maxValue * N too large\n");
        exit(1);
    }
    std::mt19937 gen;
    std::uniform_int_distribution<int> distr(1, maxValue);
    for (int i = 0; i < N; ++i) {
        inHost[i] = distr(gen);
        inHost[i] = 100 + i;
    }

    CUDA_CHECK(cudaMemcpy(inDev, inHost, N * sizeof(inDev[0]), cudaMemcpyHostToDevice));

    Scan scan;
    scan.inclusiveSum(inDev, outDev, N);
    CUDA_CHECK(cudaMemcpy(outHost, outDev, N * sizeof(outDev[0]), cudaMemcpyDeviceToHost));

    int sum = 0;
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        sum += inHost[i];
        if (outHost[i] != sum) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        printf("Incorrect result for N=%d\n", N);
        if (verbose) {
            printf("  Index  Input  Expected   Computed\n");
            int sum = 0;
            for (int i = 0; i < N; ++i) {
                sum += inHost[i];
                printf("%7d  %5d  %8d   %8d%s\n",
                       i, inHost[i], sum, outHost[i],
                       outHost[i] != sum ? "   <------" : "");
            }
        } else {
            printf("Run with --verbose for more detailed output.\n");
        }
        exit(1);
    }

    CUDA_CHECK(cudaFreeHost(outHost));
    CUDA_CHECK(cudaFreeHost(inHost));
    CUDA_CHECK(cudaFree(outDev));
    CUDA_CHECK(cudaFree(inDev));
    printf("OK!\n");
}

__global__ void fillWithDummyValues(int *out, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        out[idx] = 2 * idx + 1;  // Some random value.
}

static void benchmark(int N, int repeat) {
    char str[48];
    snprintf(str, 48, "Benchmark for N=%d with %d repeats... ", N, repeat);
    printf("%-45s", str);
    fflush(stdout);

    int *inDev;
    int *outDev;
    CUDA_CHECK(cudaMalloc(&inDev, N * sizeof(inDev[0])));
    CUDA_CHECK(cudaMalloc(&outDev, N * sizeof(outDev[0])));
    CUDA_LAUNCH(fillWithDummyValues, (N + 1024 - 1) / 1024, 1024, inDev, N);

    auto measure = [=](auto &scan) -> double {
        // Warm-up.
        scan.inclusiveSum(inDev, outDev, N);

        CUDA_CHECK(cudaDeviceSynchronize());

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i)
            scan.inclusiveSum(inDev, outDev, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        const double dt = 1e-9 * (double)std::chrono::duration_cast<
                std::chrono::nanoseconds>(t1 - t0).count() / repeat;
        const double bandwidth = N * sizeof(int) / dt;
        return bandwidth;
    };

    Scan scan;
    ScanCub scanCub;
    const double bandwidth = measure(scan);
    const double bandwidthCub = measure(scanCub);
    printf("Scan: %6.2fGB/s  cub: %6.2fGB/s\n",
           1e-9 * bandwidth,
           1e-9 * bandwidthCub);

    CUDA_CHECK(cudaFree(outDev));
    CUDA_CHECK(cudaFree(inDev));
}

static void testAll(CmdlineArgs args) {
    auto test = [verbose = args.verbose](int N) {
        ::test(N, verbose);
    };
    if (args.warp && !args.profile) {
        test(1);
        test(32);
    }
    if (args.block && !args.profile) {
        test(35);
        test(100);
        test(128+32);
        test(300);
        test(1024);
    }
    if (args.medium) {
        if (args.profile) {
            benchmark(1011151, 0);
        } else {
            test(1026);
            test(17000);
            test(1024 * 2);
            test(1023001);
            benchmark(10135, 1000);
            benchmark(101345, 100);
            benchmark(1011151, 10);
        }
    }
    if (args.large) {
        if (args.profile) {
            benchmark(52341211, 0);
        } else {
            test(1200141);
            test(10211511);
            benchmark(12341211, 5);
            benchmark(52341211, 5);
        }
    }
}

static void printUsage(const char *path) {
    printf("Usage: %s [--warp] [--block] [--medium] [--large] [--profile] [--verbose]\n", path);
}

static CmdlineArgs parseCmdlineArgs(int argc, const char * const *argv) {
    CmdlineArgs args{};
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            exit(0);
        }
        if (strcmp(argv[i], "--warp") == 0) {
            args.warp = true;
            continue;
        }
        if (strcmp(argv[i], "--block") == 0) {
            args.block = true;
            continue;
        }
        if (strcmp(argv[i], "--medium") == 0) {
            args.medium = true;
            continue;
        }
        if (strcmp(argv[i], "--large") == 0) {
            args.large = true;
            continue;
        }
        if (strcmp(argv[i], "--profile") == 0) {
            args.profile = true;
            continue;
        }
        if (strcmp(argv[i], "--verbose") == 0) {
            args.verbose = true;
            continue;
        }
        printUsage(argv[0]);
        exit(1);
    }
    if (!args.warp && !args.block && !args.medium && !args.large)
        args.warp = args.block = args.medium = args.large = true;
    return args;
}

int main(int argc, char **argv) {
    CmdlineArgs args = parseCmdlineArgs(argc, argv);
    testAll(args);
}
