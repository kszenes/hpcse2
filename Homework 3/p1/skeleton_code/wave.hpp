#include <mpi.h>
#include <vector>

class WaveEquation
{
public:
    ~WaveEquation();

    WaveEquation(int a_N,
                 int a_procs_per_dim,
                 double a_t_end,
                 MPI_Comm a_comm);

    void run();

    void initializeGrid();
    double computeSquaredEnergyNorm() const;
    void applyStencil(int i, int j);

    void saveGrid(int timestep) const;

private:
    static constexpr double L = 1.0; // size of box
    int N;              // grid points per direction for this rank
    int N_halo;            // for readability is N + 2
    double h;              // grid spacing (dx = dy = dz = h)
    double dt;             // timestep
    double t;              // current time
    double c_aug;          // augmented wavespeed term
    const double t_end;    // Total simulation time
    double origin[2];      // Spatial origin on local grid
    int coords[2];         // Coordinates withing the cartesian grid
    int nums[2];           // nprocs along each dimension
    std::vector<double> u; // solution vector
    std::vector<double> u_old;
    std::vector<double> u_new;

    int Ntot;
    int procs_per_dim;

    int size; // total number of MPI ranks = procs_per_dim*procs_per_dim
    int rank;

    int rank_plus[2]; // neighboring ranks
    int rank_minus[2];

    MPI_Comm cart_comm; // cartesian topology distributed over the grid
};
