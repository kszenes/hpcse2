#include "WaveEquation.h"
#include <cassert>
#include <iostream>

int main(int argc, char **argv)
{
  //TODO: initialize MPI & OPENMP correctly with appropriate thread safety
  // MPI_Init(&argc, &argv);
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 3)
  {
    if (rank == 0)
      std::cerr << "================================================" << std::endl
                << "Wrong arguments! Correct usage:"                  << std::endl
                << "mpirun -n np wave {gridpoints_per_dim} {t_end}"   << std::endl
                << "================================================" << std::endl;        
    return 1;
  }

  const int gridpoints_per_dim = std::stoi(argv[1]);
  const double t_end           = std::stod(argv[2]);


  MPI_Barrier(MPI_COMM_WORLD);//put a barrier here to measure execution time correctly
  double time = -MPI_Wtime();
  {
    WaveEquation simulation = WaveEquation(gridpoints_per_dim, MPI_COMM_WORLD);
    simulation.run(t_end);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  time += MPI_Wtime();

  if (rank == 0)
    std::cout << "Runtime = " << time << std::endl;

  MPI_Finalize();

  return 0;
}