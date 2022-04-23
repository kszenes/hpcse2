#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <cassert>

class Equation2D
{
  /*
   * Base class to solve a 2D Partial Differential Equation for one scalar quantity.
   * The equation is solved in the [0,1]x[0,1] square with periodic boundary conditions.
   *
   * The user needs to provide an MPI communicator and the number of grid points per dimension
   * in the class constructor. The equation is then solved from t=0 up to t=t_end,
   * by calling the "run(t_end)" method.
   *
   * ALL derived classes MUST define the following functions:
   * 1. initialCondition(x,y): specify a function of (x,y) to be used as an initial condition
   * 2. computeTimestep(): compute the timestep dt. Depending on the equation solved, the timestep
   *                       may or may not have a constant value. 
   * 3. applyStencil(i,j): this defines an "update rule" for grid point (i,j). It should return the
   *                       value of the grid point at the next timestep n+1, as a function of the 
   *                       neighboring points (i+1,j),(i-1,j),(i,j+1),(i,j-1) and/or the values at
   *                       steps n and n-1.
   * 4. derivedFunctionCalls(): this can be anything else that is specific to the equation solved. 
   *
   */
  protected:
    virtual double initialCondition(const double x, const double y) =0;

    virtual void computeTimestep()=0;

    virtual void applyStencil(const int i, const int j)=0;

    virtual void derivedFunctionCalls()=0;

    void saveGrid(const int timestep) const;

    const double L{1.0};  // size of the square domain (0<x<L, 0<y<L)
    int Ntot;             // grid points per direction (global quantity)
    int N;                // grid points per direction for this rank
    int N_halo;           // = N + 2
    double h;             // grid spacing (dx = dy = h)
    double dt;            // timestep
    double * u;           // solution vector at timestep n
    double * u_old;       // solution vector at timestep n-1
    double * u_new;       // solution vector at timestep n+1
    MPI_Comm cart_comm;   // cartesian topology communicator
    int size;             // total number of MPI ranks
    int rank;             // ID of this rank in the cartesian communicator
    int rank_plus [2];    // Neighboring ranks in Cartesian grid
    int rank_minus[2];    // Neighboring ranks in Cartesian grid
    int coords[2];        // Indices (I,J) of this rank's location in the Cartesian grid
    double origin[2];     // Indices (I,J) converted to location in space
    MPI_Datatype SEND_HALO_PLUS [2]; // MPI datatype to send halo cells
    MPI_Datatype SEND_HALO_MINUS[2]; // MPI datatype to send halo cells
    MPI_Datatype RECV_HALO_PLUS [2]; // MPI datatype to receive halo cells
    MPI_Datatype RECV_HALO_MINUS[2]; // MPI datatype to receive halo cells
    MPI_Datatype grid;               // MPI datatype to facilitate saving solution to files
    MPI_Datatype view;               // MPI datatype to facilitate saving solution to files

  public:
    virtual ~Equation2D();

    Equation2D(const int a_N, const MPI_Comm a_comm);

    void run(const double t_end);
};