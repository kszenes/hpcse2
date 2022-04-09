#include "wave.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            std::cerr << "==============================================================" << std::endl
                      << "Wrong arguments! Correct usage:" << std::endl
                      << "mpirun -n np wave {gridpoints_per_dim} {t_end}" << std::endl
                      << "==============================================================" << std::endl;        
        return 1;
    }

    int gridpoints_per_dim = std::stoi(argv[1]);
    int procs_per_dim = static_cast<int>(std::sqrt(size));
    double t_end = std::stod(argv[2]);
    // std::cout << size << " " << procs_per_dim << '\n';
    assert(gridpoints_per_dim % procs_per_dim == 0);
    assert(procs_per_dim * procs_per_dim == size);

    {
        WaveEquation simulation = WaveEquation(gridpoints_per_dim, procs_per_dim, t_end, MPI_COMM_WORLD);
        simulation.run();
    }

    MPI_Finalize();

    return 0;
}
