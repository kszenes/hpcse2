#include "LaplacianSmoother.h"

using namespace std::chrono;

#define MEASURE(func, timer)                                                   \
    do {                                                                       \
        const auto t_start = steady_clock::now();                              \
        func;                                                                  \
        const auto t_end = steady_clock::now();                                \
        timer += duration_cast<microseconds>(t_end - t_start).count();         \
    } while (0)

void LaplacianSmoother::sweep()
{
    const auto ts_start = steady_clock::now();

    // smoothing logic:
    MEASURE(comm()           , t_async);
    MEASURE(smooth_inner()   , t_inner);
    MEASURE(sync()           , t_ghosts);
    MEASURE(smooth_boundary(), t_boundary);

    // swap references
    std::swap(temp, data);

    sweep_count += 1;
    const auto ts_end = steady_clock::now();
    t_sweep += duration_cast<microseconds>(ts_end - ts_start).count();
}

void LaplacianSmoother::report()
{
    const int Nx = N[0] - 2;
    const int Ny = N[1] - 2;
    const int Nz = N[2] - 2;
    t_sweep    /= sweep_count;
    t_async    /= sweep_count;
    t_inner    /= sweep_count;
    t_ghosts   /= sweep_count;
    t_boundary /= sweep_count;
    std::printf("Measurement report for dimension %d x %d x %d:\n", Nx, Ny, Nz);
    std::printf("\tSweep count:           %zu\n", sweep_count);
    std::printf("\tAvg. time sweep:       %.3e microseconds\n", t_sweep);
    std::printf("\tAvg. time async comm:  %.3e microseconds (%.2f%%)\n", t_async   , 100*t_async   /t_sweep);
    std::printf("\tAvg. time inner:       %.3e microseconds (%.2f%%)\n", t_inner   , 100*t_inner   /t_sweep);
    std::printf("\tAvg. time sync ghosts: %.3e microseconds (%.2f%%)\n", t_ghosts  , 100*t_ghosts  /t_sweep);
    std::printf("\tAvg. time boundary:    %.3e microseconds (%.2f%%)\n", t_boundary, 100*t_boundary/t_sweep);

    double checksum = 0.0;
    for (int k = 0; k < Nz; ++k)
    for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i)
        checksum += operator()(i,j,k);
    std::printf("\tChecksum:    %10.8e \n", checksum);
}

void LaplacianSmoother::sync()
{
    // no synchronization needed here, only loading (periodic) ghost cells is needed
    const int Nx = N[0] - 2;
    const int Ny = N[1] - 2;
    const int Nz = N[2] - 2;

    // j-k faces
    for (int k = 0; k < Nz; ++k)
    for (int j = 0; j < Ny; ++j)
    {
        operator()(-1,j,k) = operator()(Nx-1,j,k);
        operator()(Nx,j,k) = operator()(0   ,j,k);
    }

    // i-k faces
    for (int k = 0; k < Nz; ++k)
    for (int i = 0; i < Nx; ++i)
    {
        operator()(i,-1,k) = operator()(i,Ny-1,k);
        operator()(i,Ny,k) = operator()(i,0   ,k);
    }

    // i-j faces
    for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i)
    {
        operator()(i,j,-1) = operator()(i,j,Nz-1);
        operator()(i,j,Nz) = operator()(i,j,0   );
    }
}

void LaplacianSmoother::smooth_inner()
{
    // smooth inner domain
    const int Nx = N[0] - 2;
    const int Ny = N[1] - 2;
    const int Nz = N[2] - 2;
    #pragma omp parallel
    {
        #pragma omp for
        for (int k = 1; k < Nz - 1; ++k)
        for (int j = 1; j < Ny - 1; ++j)
        for (int i = 1; i < Nx - 1; ++i)
            update(i, j, k);
    }
}

void LaplacianSmoother::smooth_boundary()
{
    // smooth boundary domain
    const int Nx = N[0] - 2;
    const int Ny = N[1] - 2;
    const int Nz = N[2] - 2;

    // j-k faces
    const int i0 = 0;
    const int iN = Nx - 1;
    #pragma omp parallel for
    for (int k = 0; k < Nz; ++k)
    for (int j = 0; j < Ny; ++j)
    {
        update(i0, j, k);
        update(iN, j, k);
    }

    // i-k faces
    const int j0 = 0;
    const int jN = Ny - 1;
    #pragma omp parallel for
    for (int k = 0; k < Nz; ++k)
    for (int i = 1; i < Nx - 1; ++i)
    {
       update(i, j0, k);
       update(i, jN, k);
    }

    // i-j faces
    const int k0 = 0;
    const int kN = Nz - 1;
    #pragma omp parallel for
    for (int j = 1; j < Ny - 1; ++j)
    for (int i = 1; i < Nx - 1; ++i)
    {
       update(i, j, k0);
       update(i, j, kN);
    }
}