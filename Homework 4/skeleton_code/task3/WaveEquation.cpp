#include "WaveEquation.h"

double WaveEquation::initialCondition(const double x, const double y)
{
    const double r = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
    return 1 - std::sin(M_PI * r) * std::exp(-r);
}

void WaveEquation::applyStencil(const int i, const int j)
{
    const int center = i * N_halo + j;
    u_new[center] = 2.0 * u[center] - u_old[center] + c_aug *(u[(i+1)*N_halo+ j   ] 
                                                            + u[(i-1)*N_halo+ j   ] 
                                                            + u[ i   *N_halo+(j+1)]
                                                            + u[ i   *N_halo+(j-1)] - 4.0*u[center]);
}

void WaveEquation::derivedFunctionCalls()
{
    // TODO : parallelize this function with OPENMP. 
    //        The OPENMP-parallel version will be called by all threads 
    //        from inside a parallel region. 

    //the energy norm is computed here
    energy_norm_m = 0;
    energy_norm_a = 0;

    for (int i = 1; i < N + 1; ++i)
    for (int j = 1; j < N + 1; ++j)
    {
        energy_norm_m += (u[i * N_halo + j] - u_old[i * N_halo + j] ) * (u[i * N_halo + j] - u_old[i * N_halo + j] );
        energy_norm_a += (u[(i + 1) * N_halo + j] - u[i * N_halo + j]) * (u[(i + 1) * N_halo + j] - u[i * N_halo + j])
                      +  (u[(i - 1) * N_halo + j] - u[i * N_halo + j]) * (u[(i - 1) * N_halo + j] - u[i * N_halo + j])
                      +  (u[i * N_halo + (j + 1)] - u[i * N_halo + j]) * (u[i * N_halo + (j + 1)] - u[i * N_halo + j])
                      +  (u[i * N_halo + (j - 1)] - u[i * N_halo + j]) * (u[i * N_halo + (j - 1)] - u[i * N_halo + j]);
    }    

    double energy_norm = 0.5 *(energy_norm_m *h/dt *h/dt + energy_norm_a);

    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &energy_norm, &energy_norm, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0)
        std::cout << " --> E(t) = "  << energy_norm << std::endl;
}

void WaveEquation::computeTimestep()
{
    dt = h/3;
    c_aug = dt * dt / (h * h);
}