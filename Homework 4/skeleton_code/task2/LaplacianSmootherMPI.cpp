#include "LaplacianSmootherMPI.h"
using namespace std::chrono;

LaplacianSmootherMPI::LaplacianSmootherMPI(const int Nx,
                                           const int Ny,
                                           const int Nz,
                                           const int Px,
                                           const int Py,
                                           const int Pz,
                                           const MPI_Comm comm_root)
    : LaplacianSmoother(Nx, Ny, Nz), comm_world(comm_root),
      comm_cart(MPI_COMM_NULL), procs{Px, Py, Pz},
      recv_req(6, MPI_REQUEST_NULL), send_req(6, MPI_REQUEST_NULL)
{
    int size;
    MPI_Comm_size(comm_world, &size);
    if (size != Px * Py * Pz)
    {
        MPI_Abort(comm_world, 1);
    }
    const int periodic[3] = {true, true, true};


    // TODO: define cartesian topology and neighboring ranks
    comm_cart = comm_world;
    const int procs[] = {Px, Py, Pz};
    MPI_Cart_create(comm_root, 3, procs, periodic, true, &comm_cart);

    MPI_Comm_rank(comm_cart, &rank_cart);
    MPI_Cart_shift(comm_cart, 0, 1, &nbr[X0], &nbr[X1]); // x 
    MPI_Cart_shift(comm_cart, 1, 1, &nbr[Y0], &nbr[Y1]); // y
    MPI_Cart_shift(comm_cart, 2, 1, &nbr[Z0], &nbr[Z1]); // z

    // TODO: define custom MPI datatypes for x,y,z faces
    MPI_Datatype y_vector;
    MPI_Type_vector(Ny, 1, Nx, MPI_DOUBLE, &y_vector);

    MPI_Type_vector(Nz, 1, 1, y_vector, &FaceX);
    MPI_Type_vector(Nz, Nx, Ny, MPI_DOUBLE, &FaceY);
    MPI_Type_vector(Nx*Ny, 1, 1, MPI_DOUBLE, &FaceZ);

    MPI_Type_commit(&FaceX);
    MPI_Type_commit(&FaceY);
    MPI_Type_commit(&FaceZ);

    // TODO: initialize data.
    // This is only correct for one rank. For multiple ranks coords corresponds to coordinates in
    // Cartesian communicator
    int coords [3] = {0,0,0};
    MPI_Cart_coords(comm_cart, rank_cart, 3, coords);
    for (int k = 0; k < Nz; ++k)
    for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i)
    {
        const int ig = Nx * coords[0] + i;
        const int jg = Ny * coords[1] + j;
        const int kg = Nz * coords[2] + k;
        operator()(i, j, k) = 123*ig*ig + 456*jg*jg + 789*kg*kg;
    }
}

LaplacianSmootherMPI::~LaplacianSmootherMPI()
{
    //TODO: free MPI datatypes and communicator
    MPI_Type_free(&FaceX);
    MPI_Type_free(&FaceY);
    MPI_Type_free(&FaceZ);
    MPI_Comm_free(&comm_cart);
}

void LaplacianSmootherMPI::report()
{
    const int Nx = N[0] - 2;
    const int Ny = N[1] - 2;
    const int Nz = N[2] - 2;
    double checksum_global = 0.0;
    double checksum = 0.0;
    for (int k = 0; k < Nz; ++k)
    for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i)
        checksum += operator()(i,j,k);
    MPI_Reduce(&checksum,&checksum_global,1,MPI_DOUBLE,MPI_SUM,0,comm_cart);
    std::printf("\tChecksum:    %10.8e \n", checksum_global);


    //TODO : implement a distributed version of report() from LaplacianSmoother
}

void LaplacianSmootherMPI::comm()
{
    //TODO : send and receive the six faces of ghost cells


}

void LaplacianSmootherMPI::sync()
{
    //TODO : uncomment the following to enable waiting for communication to complete

    // wait for pending sends (because we will write to it)
    //MPI_Waitall(6, send_req.data(), MPI_STATUSES_IGNORE);

    // wait for pending recvs (because we will read from it)
    //MPI_Waitall(6, recv_req.data(), MPI_STATUSES_IGNORE);
}