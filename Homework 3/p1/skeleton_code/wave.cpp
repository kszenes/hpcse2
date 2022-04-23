#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include "wave.hpp"

//
// Function defining the initial displacement on the grid
//
double f(double x, double y)
{
    double r = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
    return 1 - sin(M_PI * r) * exp(-r);
}

WaveEquation::WaveEquation(int a_gridpoints_per_dim,
                           int a_procs_per_dim,
                           double a_t_end,
                           MPI_Comm a_comm)
    : t_end(a_t_end), Ntot(a_gridpoints_per_dim), 
    procs_per_dim(a_procs_per_dim), cart_comm(a_comm)
{
    h = L / Ntot;
    N = Ntot / procs_per_dim;
    N_halo = N + 2;

    dt = h / 3;
    c_aug = dt * dt / (h * h);

    u.resize(N_halo * N_halo);
    u_old.resize(N_halo * N_halo);
    u_new.resize(N_halo * N_halo);

    // TODO Question a)
    // MPI related initializations (create new communicator and initialize
    // member variables) (see 'wave.hpp' for present member variables:
    // cart_comm, nums, rank_plus, rank_minus)
    //
    // Do not forget to free the communicator in the destructor of this class!!
    // (bottom of this file)
    MPI_Comm_size(a_comm, &size);
    
    nums[0] = procs_per_dim;
    nums[1] = procs_per_dim;
    MPI_Dims_create(size, 2, nums);

    int periodic[2] = {1, 1};
    MPI_Cart_create(a_comm, 2, nums, periodic, true, &cart_comm);
    MPI_Comm_rank(cart_comm, &rank);

    MPI_Cart_shift(cart_comm, 0, 1, &rank_minus[0], &rank_plus[0]);
    MPI_Cart_shift(cart_comm, 1, 1, &rank_minus[1], &rank_plus[1]);

    if (rank == 0) {
        std::cout << "(" << nums[0] << ", " << nums[1] << ") processes mapping to a (" << N << ", " << N << ") grid\n";
    } 

    //
    // Find its location in the simulation space
    // TODO Uncomment this section and the one at the beginning when finished
    // with subquestion a))
    //

    MPI_Cart_coords(cart_comm, rank, 2, &coords[0]);
    origin[0] = N * h * coords[0];
    origin[1] = N * h * coords[1];

    //
    // Set initial conditions (inclusive first time derivative)
    //
    initializeGrid();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = (i + 1) * N_halo + (j + 1);
            u_old[idx] = u[idx];
            u_new[idx] = u[idx];
        }
    }
}

void WaveEquation::run()
{
    double t = 0.0;
    unsigned int count = 0;

    // TODO Question b)
    // Create datatypes to communicate halo boundaries (before the loop while-loop)
    //
    // You can use the buffers below to create the necessary array datatypes in a for-loop.
    // Feel free to initialize and store them in a different format
    // (this might be less error prone but also results in more lines of code).
    //
    // The buffers' lengths represent the number of dimensions of our domain (we are solving a 2D problem).
    // i.e. in SEND_HALO_PLUS[0] we store the datatype for sending the right-most column
    //          (of the inner N x N grid) to the right neighboring rank.
    //      RECV_HALO_MINUS[1] stores the datatype for receiving the halo row sent from the rank below.
    //  Contact the slides of the tutorial where we introduced the exercise if anything is unclear.

    MPI_Datatype SEND_HALO_PLUS[2];
    MPI_Datatype SEND_HALO_MINUS[2];

    MPI_Datatype RECV_HALO_PLUS[2];
    MPI_Datatype RECV_HALO_MINUS[2];

    int sizes[] = {N_halo, N_halo};
    int subsizesH[] = {N, 1};
    int subsizesV[] = {1, N};

    int startL_send[] = {1, 1};
    int startR_send[] = {N, 1};

    int startL_recv[] = {0, 1};
    int startR_recv[] = {N+1, 1};

    int startB_send[] = {1, 1};
    int startT_send[] = {1, N};

    int startT_recv[] = {1, N+1};
    int startB_recv[] = {1, 0};

    MPI_Type_create_subarray(2, sizes, subsizesV, startL_send, MPI_ORDER_C, MPI_DOUBLE, &SEND_HALO_MINUS[0]);
    MPI_Type_create_subarray(2, sizes, subsizesV, startR_send, MPI_ORDER_C, MPI_DOUBLE, &SEND_HALO_PLUS[0]);

    MPI_Type_create_subarray(2, sizes, subsizesV, startL_recv, MPI_ORDER_C, MPI_DOUBLE, &RECV_HALO_MINUS[0]);
    MPI_Type_create_subarray(2, sizes, subsizesV, startR_recv, MPI_ORDER_C, MPI_DOUBLE, &RECV_HALO_PLUS[0]);

    MPI_Type_create_subarray(2, sizes, subsizesH, startB_send, MPI_ORDER_C, MPI_DOUBLE, &SEND_HALO_MINUS[1]);
    MPI_Type_create_subarray(2, sizes, subsizesH, startT_send, MPI_ORDER_C, MPI_DOUBLE, &SEND_HALO_PLUS[1]);

    MPI_Type_create_subarray(2, sizes, subsizesH, startT_recv, MPI_ORDER_C, MPI_DOUBLE, &RECV_HALO_PLUS[1]);
    MPI_Type_create_subarray(2, sizes, subsizesH, startB_recv, MPI_ORDER_C, MPI_DOUBLE, &RECV_HALO_MINUS[1]);


    MPI_Type_commit(&SEND_HALO_MINUS[0]);
    MPI_Type_commit(&SEND_HALO_MINUS[1]);
    MPI_Type_commit(&SEND_HALO_PLUS[0]);
    MPI_Type_commit(&SEND_HALO_PLUS[1]);

    MPI_Type_commit(&RECV_HALO_MINUS[0]);
    MPI_Type_commit(&RECV_HALO_MINUS[1]);
    MPI_Type_commit(&RECV_HALO_PLUS[0]);
    MPI_Type_commit(&RECV_HALO_PLUS[1]);

    //
    // Main loop propagating the solution forward in time
    //
    while (t < t_end) {

        // TODO Question c)
        // Send and receive halo boundaries
        //
<<<<<<< HEAD
        MPI_Request request[4];
        // MPI_Irecv(&u, 1, RECV_HALO_MINUS[0], rank_minus[0], 0, cart_comm, &request[0]);
        // MPI_Irecv(&u, 1, RECV_HALO_MINUS[1], rank_minus[1], 1, cart_comm, &request[1]);
        // MPI_Irecv(&u, 1, RECV_HALO_PLUS[0], rank_plus[0], 2, cart_comm, &request[2]);
        // MPI_Irecv(&u, 1, RECV_HALO_PLUS[1], rank_plus[1], 3, cart_comm, &request[3]);

        // MPI_Send(&u, 1, SEND_HALO_MINUS[0], rank_minus[0], 0, cart_comm);
        // MPI_Send(&u, 1, SEND_HALO_MINUS[1], rank_minus[1], 1, cart_comm);
        MPI_Send(&u, 1, SEND_HALO_PLUS[0], rank_plus[0], 2, cart_comm);
        // MPI_Send(&u, 1, SEND_HALO_PLUS[1], rank_plus[1], 3, cart_comm);

=======
        MPI_Request request[8];
        MPI_Irecv(u.data(), 1, RECV_HALO_MINUS[0], rank_minus[0], 0, cart_comm, &request[0]);
        MPI_Irecv(u.data(), 1, RECV_HALO_PLUS[0], rank_plus[0], 1, cart_comm, &request[1]);
        MPI_Irecv(u.data(), 1, RECV_HALO_MINUS[1], rank_minus[1], 2, cart_comm, &request[2]);
        MPI_Irecv(u.data(), 1, RECV_HALO_PLUS[1], rank_plus[1], 3, cart_comm, &request[3]);

        MPI_Isend(u.data(), 1, SEND_HALO_PLUS[0], rank_plus[0], 0, cart_comm, &request[4]);
        MPI_Isend(u.data(), 1, SEND_HALO_MINUS[0], rank_minus[0], 1, cart_comm, &request[5]);
        MPI_Isend(u.data(), 1, SEND_HALO_PLUS[1], rank_plus[1], 2, cart_comm, &request[6]);
        MPI_Isend(u.data(), 1, SEND_HALO_MINUS[1], rank_minus[1], 3, cart_comm, &request[7]);
        MPI_Waitall(8, request, MPI_STATUS_IGNORE);
>>>>>>> 359211339c8df43a63719f18b8c874cea3535a86


        //
        // TODO Uncomment following when finished with subquestion a)
        //
        if (count % 10 == 9) {
            saveGrid(count);
            double energy_norm = computeSquaredEnergyNorm();
            if (rank == 0)
                std::cout << "t=" << count << " : E(t) = "  << energy_norm << std::endl;
        }

        //
        // Update the cells with FD stencil
        //
        for (int i = 1; i < N + 1; ++i) {
            for (int j = 1; j < N + 1; ++j) {
                applyStencil(i, j);
            }
        }

        //
        // Swap vectors
        //
        u_old.swap(u);
        u.swap(u_new);

        //
        // Update time
        //
        t += dt;
        count++;
    }

    // TODO Question c)
    // Free communication datatypes
    //
    MPI_Type_free(&SEND_HALO_MINUS[0]);
    MPI_Type_free(&SEND_HALO_MINUS[1]);
    MPI_Type_free(&SEND_HALO_PLUS[0]);
    MPI_Type_free(&SEND_HALO_PLUS[1]);

    MPI_Type_free(&RECV_HALO_MINUS[0]);
    MPI_Type_free(&RECV_HALO_MINUS[1]);
    MPI_Type_free(&RECV_HALO_PLUS[0]);
    MPI_Type_free(&RECV_HALO_PLUS[1]);
}

void WaveEquation::initializeGrid()
{
    double x_pos, y_pos;
    for (int i = 0; i < N; ++i) {
        x_pos = origin[0] + i * h + 0.5 * h;
        for (int j = 0; j < N; ++j) {
            y_pos = origin[1] + j * h + 0.5 * h;
            u[(i + 1) * N_halo + (j + 1)] = f(x_pos, y_pos);
        }
    }
}

double WaveEquation::computeSquaredEnergyNorm() const
{
    double energy_norm_m = 0.0;
    double energy_norm_a = 0.0;
    
    for (int i = 1; i < N + 1; ++i)
        for (int j = 1; j < N + 1; ++j) {
              energy_norm_m += (u[i * N_halo + j] - u_old[i * N_halo + j] ) * (u[i * N_halo + j] - u_old[i * N_halo + j] );
              energy_norm_a += (u[(i + 1) * N_halo + j] - u[(i - 1) * N_halo + j]) * (u[(i + 1) * N_halo + j] - u[(i - 1) * N_halo + j])
                            +  (u[i * N_halo + (j + 1)] - u[i * N_halo + (j - 1)]) * (u[i * N_halo + (j + 1)] - u[i * N_halo + (j - 1)]);
        }

    double energy_norm = 0.5 *(energy_norm_m / c_aug + 0.25 * energy_norm_a);

    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &energy_norm, &energy_norm, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    return energy_norm;
}

void WaveEquation::applyStencil(int i, int j)
{
    int center = i * N_halo + j;
    u_new[center] =
        2.0 * u[center] - u_old[center] +
        c_aug * (u[(i + 1) * N_halo + j] + u[(i - 1) * N_halo + j] +
                 u[i * N_halo + (j + 1)] + u[i * N_halo + (j - 1)] -
                 4.0 * u[center]);
}

void WaveEquation::saveGrid(int timestep) const
{
    std::stringstream ss;
    ss << "./output/wave_" << std::setfill('0') << std::setw(3) << timestep
       << ".bin";
    std::string fname = ss.str();

    // Create derived datatype for interior grid (output grid)
    MPI_Datatype grid;
    const int start[2] = {1, 1};
    const int arrsize[2] = {N_halo, N_halo};
    const int gridsize[2] = {N, N};

    MPI_Type_create_subarray(
        2, arrsize, gridsize, start, MPI_ORDER_C, MPI_DOUBLE, &grid);
    MPI_Type_commit(&grid);

    // Create derived type for file view
    MPI_Datatype view;
    const int startV[2] = {coords[0] * N, coords[1] * N};
    const int arrsizeV[2] = {nums[0] * N, nums[1] * N};
    const int gridsizeV[2] = {N, N};

    MPI_Type_create_subarray(
        2, arrsizeV, gridsizeV, startV, MPI_ORDER_C, MPI_DOUBLE, &view);
    MPI_Type_commit(&view);

    MPI_File fh;

    MPI_File_open(cart_comm,
                  fname.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL,
                  &fh);

    MPI_File_set_view(fh, 0, MPI_DOUBLE, view, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, u.data(), 1, grid, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&grid);
}

WaveEquation::~WaveEquation() {
    // TODO Question a)
    // Free the Cartesian communicator
    MPI_Comm_free(&cart_comm);
}
