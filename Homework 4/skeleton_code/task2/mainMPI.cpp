#include "LaplacianSmootherMPI.h"
#include <cstdlib>

int main(int argc, char *argv[])
{
    // We are using MPI_Init here although we use OpenMP in some parts of the
    // program.  If and only if any call to the MPI API is performed with the
    // master thread, then this code is correct from an MPI point of view.  In
    // any other multi-threaded program or if you want to be more explicit, you
    // must use MPI_Init_thread() here instead.  Note that this may impose a
    // performance penalty because the guarantee of thread-safety imposes
    // restrictions on possible optimizations.
    MPI_Init(&argc, &argv);

    {
        int Nx, Ny, Nz, Px, Py, Pz;
        Nx = Ny = Nz = 128;
        Px = Py = Pz = 1;
        if (7 == argc) {
            Nx = std::atoi(argv[1]);
            Ny = std::atoi(argv[2]);
            Nz = std::atoi(argv[3]);
            Px = std::atoi(argv[4]);
            Py = std::atoi(argv[5]);
            Pz = std::atoi(argv[6]);
        }

        LaplacianSmootherMPI ls_data(Nx, Ny, Nz, Px, Py, Pz, MPI_COMM_WORLD);
        for (int i = 0; i < 100; ++i) {
            ls_data.sweep();
        }
        ls_data.report();
    }

    MPI_Finalize();
    return 0;
}
