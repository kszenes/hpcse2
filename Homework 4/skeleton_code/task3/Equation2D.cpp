#include "Equation2D.h"

Equation2D::Equation2D(const int a_N, const MPI_Comm comm) : Ntot(a_N)
{
    MPI_Comm_size(comm, &size);
    const int ranks_per_dim = sqrt(size);
    assert (size == ranks_per_dim * ranks_per_dim && "Number of processes must be a square number.\n");

    N      = Ntot / ranks_per_dim; //grid points per dimension for each rank
    N_halo = N + 2;                //add two points for halo cells
    h      = L / Ntot;             //grid spacing

    //Allocation of solution and auxiliary arrays
    u     = new double[N_halo * N_halo];
    u_old = new double[N_halo * N_halo];
    u_new = new double[N_halo * N_halo];

    //Create Cartesian topology
    const int periodic[2] = {true, true}; //periodic boundaries
    int nums[2] = {0,0};
    MPI_Dims_create(size, 2, nums);                             //get ranks along each dimension
    MPI_Cart_create(comm, 2, nums, periodic, true, &cart_comm); //get Cartesian communicator

    //Get neighboring ranks
    MPI_Comm_rank (cart_comm, &rank);
    MPI_Cart_shift(cart_comm, 0, 1, &rank_minus[0], &rank_plus[0]);
    MPI_Cart_shift(cart_comm, 1, 1, &rank_minus[1], &rank_plus[1]);

    //Get indices (I,J) of this rank's location in the Cartesian grid
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    //Convert indices to location in space
    origin[0] = N * h * coords[0];
    origin[1] = N * h * coords[1];

    //Define custom datatypes to send/receive an Nx1 or an 1xN array of halo cells at each boundary
    const int ndims = 2;
    const int sizes[2] = {N_halo, N_halo};
    int subsizes   [2] = {N     , N     };
    int starts     [2] = {1     , 1     };
    for (int i = 0; i < ndims; ++i)
    {
        // Dimension i has now subsize 1, while the rest have subsize N
        subsizes[i] = 1;
        
        // Topmost halo boundary
        starts[i] = 0;
        MPI_Type_create_subarray(ndims, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &RECV_HALO_MINUS[i]);

        // Topmost inner boundary
        starts[i] = 1;
        MPI_Type_create_subarray(ndims, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &SEND_HALO_MINUS[i]);

        // Lowermost inner boundary
        starts[i] = N;
        MPI_Type_create_subarray(ndims, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &SEND_HALO_PLUS[i]);

        // // Lowermost halo boundary
        starts[i] = N + 1;
        MPI_Type_create_subarray(ndims, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &RECV_HALO_PLUS[i]);

        // Reset before next dimension
        starts[i] = 1;
        subsizes[i] = N;

        // Commit types for dimension i
        MPI_Type_commit(&SEND_HALO_PLUS[i]);
        MPI_Type_commit(&RECV_HALO_PLUS[i]);
        MPI_Type_commit(&SEND_HALO_MINUS[i]);
        MPI_Type_commit(&RECV_HALO_MINUS[i]);
    }

    //Datatype for interior grid (output grid), used when saving solution to file
    const int start   [2] = {1     , 1     };
    const int arrsize [2] = {N_halo, N_halo};
    const int gridsize[2] = {N     , N     };
    MPI_Type_create_subarray(2, arrsize, gridsize, start, MPI_ORDER_C, MPI_DOUBLE, &grid);
    MPI_Type_commit(&grid);
 
    // Datatype for output file view
    const int startV   [2] = {coords[0] * N, coords[1] * N};
    const int arrsizeV [2] = {nums  [0] * N, nums  [1] * N};
    const int gridsizeV[2] = {N            , N            };
    MPI_Type_create_subarray(2, arrsizeV, gridsizeV, startV, MPI_ORDER_C, MPI_DOUBLE, &view);
    MPI_Type_commit(&view);

    if (rank == 0)
        std::cout << "(" << nums[0] << ", " << nums[1] << ") processes mapping to a (" << N << ", " << N << ") grid\n";
}

void Equation2D::run(const double t_end)
{
    //TODO (1): overlap computation and communication for pure MPI version of this function.
    //TODO (2): parallelize this function with OPENMP.
    //          You are only allowed to open only one parallel region

    MPI_Request request[8];
    
    //OPENMP parallel section for TODO (2) should start here
    #pragma omp parallel
    {
        //Set initial conditions (also set du/dt = 0 -> u_old = u at t=0)
        #pragma omp for collapse(2)
        for (int i = -1; i < N+1; ++i)
        for (int j = -1; j < N+1; ++j)
        {
            const int idx = (i + 1) * N_halo + (j + 1);
            const double x = origin[0] + i * h + 0.5 * h;
            const double y = origin[1] + j * h + 0.5 * h;
            u    [idx] = initialCondition(x, y);
            u_old[idx] = u[idx];
            u_new[idx] = u[idx];
        }

        double t = 0.0;
        unsigned int count = 0;

        while (t < t_end)
        {
            computeTimestep();

            if (count % 100 == 0)
                #pragma omp master
                {
                if (rank == 0) std::cout << "t = " << t << std::endl;
                    derivedFunctionCalls();
                    saveGrid(count);
                }

            // Send and receive halo cells
            int d;
            d = 0;
            #pragma omp master
            {
              MPI_Irecv(u, 1, RECV_HALO_MINUS[d], rank_minus[d], d    , cart_comm, &request[2 * d        ]);
              MPI_Irecv(u, 1, RECV_HALO_PLUS [d], rank_plus [d], d + 1, cart_comm, &request[2 * d + 1    ]);
              MPI_Isend(u, 1, SEND_HALO_PLUS [d], rank_plus [d], d    , cart_comm, &request[4 + 2 * d    ]);
              MPI_Isend(u, 1, SEND_HALO_MINUS[d], rank_minus[d], d + 1, cart_comm, &request[4 + 2 * d + 1]);
              d = 1;
              MPI_Irecv(u, 1, RECV_HALO_MINUS[d], rank_minus[d], d    , cart_comm, &request[2 * d        ]);
              MPI_Irecv(u, 1, RECV_HALO_PLUS [d], rank_plus [d], d + 1, cart_comm, &request[2 * d + 1    ]);
              MPI_Isend(u, 1, SEND_HALO_PLUS [d], rank_plus [d], d    , cart_comm, &request[4 + 2 * d    ]);
              MPI_Isend(u, 1, SEND_HALO_MINUS[d], rank_minus[d], d + 1, cart_comm, &request[4 + 2 * d + 1]);
            }
            #pragma omp barrier

            // Apply stencil and update solution
            #pragma omp for collapse(2)
            for (int i = 2; i < N; ++i)
            for (int j = 2; j < N; ++j)
            {
                applyStencil(i, j);
            }
            #pragma omp master
            { 
              // Wait for communication to complete
              MPI_Waitall(8, &request[0], MPI_STATUSES_IGNORE);
            }
            #pragma omp barrier

            #pragma omp for
            for (int i = 1; i < N+1; ++i)
            {
                applyStencil(i, 1);
                applyStencil(i, N);

                applyStencil(1, i);
                applyStencil(N, i);
            }

            #pragma omp master
            {
              // Swap vectors
              std::swap(u_old,u);
              std::swap(u_new,u);

              t += dt;
              count++;
            }
        }//while (t < t_end)

    }//OPENMP parallel section for TODO (2) should end here
}

void Equation2D::saveGrid(int timestep) const
{
    //TODO: make this function thread-safe. You do not need to parallelize it with OPENMP but
    //      you need to make sure it is only executed by only one thread.

    std::stringstream ss;
    ss << "./output/wave_" << std::setfill('0') << std::setw(3) << timestep << ".bin";
    std::string fname = ss.str();
    MPI_File fh;
    MPI_File_open(cart_comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, view, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, u, 1, grid, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

Equation2D::~Equation2D()
{
  delete [] u; 
  delete [] u_old; 
  delete [] u_new; 
  MPI_Type_free(&RECV_HALO_PLUS [0]);
  MPI_Type_free(&RECV_HALO_PLUS [1]);
  MPI_Type_free(&RECV_HALO_MINUS[0]);
  MPI_Type_free(&RECV_HALO_MINUS[1]);
  MPI_Type_free(&SEND_HALO_PLUS [0]);
  MPI_Type_free(&SEND_HALO_PLUS [1]);
  MPI_Type_free(&SEND_HALO_MINUS[0]);
  MPI_Type_free(&SEND_HALO_MINUS[1]);
  MPI_Type_free(&grid);
  MPI_Type_free(&view);
  MPI_Comm_free(&cart_comm); 
}
