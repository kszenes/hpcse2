#pragma once

#include <cassert>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <omp.h>
#include <iostream>
#include <cstring>

class LaplacianSmoother
{
public:
    LaplacianSmoother(const int Nx = 128, const int Ny = 128, const int Nz = 128): N{Nx+2,Ny+2,Nz+2}// +2 for ghost cells
    {
        sweep_count = 0;
        t_sweep     = 0.0;
        t_async     = 0.0;
        t_inner     = 0.0;
        t_ghosts    = 0.0;
        t_boundary  = 0.0;

        data = new double [N[0] * N[1] * N[2]];
        temp = new double [N[0] * N[1] * N[2]];

        // initialize data
        for (int k = 0; k < Nz; ++k)
        for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i)
            operator()(i, j, k) = 123*i*i + 456*j*j + 789*k*k;
    }

    ~LaplacianSmoother()
    {
        delete [] data;
        delete [] temp;
    }

    // perform a smoothing sweep
    void sweep();

    // report measurements
    virtual void report();

    // data accessor
    double &operator()(const int ix, const int iy, const int iz)
    {
        // stencil is +/- 1:
        const int I = ix + 1;
        const int J = iy + 1;
        const int K = iz + 1;
        return data[I + N[0] * (J + N[1] * K)];
    }

protected:
    // profiling
    size_t sweep_count;
    double t_sweep, t_async, t_inner, t_ghosts, t_boundary;

    // data dimensions (including ghost cells)
    const int N[3];

    // main data buffers
    double *data; // data reference
    double *temp; // temporary reference

    // communication methods
    virtual void comm() {}
    virtual void sync();

private:
    void smooth_inner();
    void smooth_boundary();
    void update(const int ix, const int iy, const int iz)
    {
        constexpr double fac = 1.0/12.0;
        const int I = ix + 1;
        const int J = iy + 1;
        const int K = iz + 1;
        temp[I + N[0] * (J + N[1] * K)] = fac * ( operator()(ix - 1, iy    , iz    ) + operator()(ix + 1, iy    , iz    ) +
                                                  operator()(ix    , iy - 1, iz    ) + operator()(ix    , iy + 1, iz    ) +
                                                  operator()(ix    , iy    , iz - 1) + operator()(ix    , iy    , iz + 1) +
                                              6 * operator()(ix, iy, iz));
    }
};
