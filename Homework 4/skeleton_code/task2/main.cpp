#include "LaplacianSmoother.h"
#include <cstdlib>

int main(int argc, char *argv[])
{
    int Nx, Ny, Nz;
    Nx = Ny = Nz = 128;
    if (argc == 4)
    {
        Nx = std::atoi(argv[1]);
        Ny = std::atoi(argv[2]);
        Nz = std::atoi(argv[3]);
    }

    LaplacianSmoother ls_data(Nx, Ny, Nz);
    for (int i = 0; i < 100; ++i)
    {
        ls_data.sweep();
    }
    ls_data.report();

    return 0;
}
