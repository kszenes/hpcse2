#pragma once

#include "LaplacianSmoother.h"
#include <mpi.h>

class LaplacianSmootherMPI : public LaplacianSmoother
{
public:
    LaplacianSmootherMPI(const int Nx = 128, // per process
                         const int Ny = 128, // per process
                         const int Nz = 128, // per process
                         const int Px = 1,   // processes in x
                         const int Py = 1,   // processes in y
                         const int Pz = 1,   // processes in z
                         const MPI_Comm comm_root = MPI_COMM_WORLD);

    ~LaplacianSmootherMPI();

    // report measurements (distributed)
    void report() override;

protected:
    // communication methods
    void comm() override;
    void sync() override;

private:
    enum { X0, X1, Y0, Y1, Z0, Z1, NFaces }; // neighbor id's for convenience

    // MPI topology
    MPI_Comm comm_world, comm_cart;
    const int procs[3];
    int rank_cart, nbr[NFaces];

    // MPI datatypes
    MPI_Datatype StripeX, FaceX, FaceY, FaceZ;
    std::vector<MPI_Request> recv_req;
    std::vector<MPI_Request> send_req;
};
