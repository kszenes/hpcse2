#include <fstream>
#include <mpi.h>
#include <stdexcept>
#include <vector>
#include <iostream>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size != 2) 
    {
        MPI_Finalize();
        throw std::runtime_error("Need exactly two ranks for this benchmark");
    }

    const size_t iterations = 1000;

    //TODO: Define correct min and max message sizes
    const size_t min_size = 1;        // this should be 1 byte
    const size_t max_size = 1 << 24;        // this should be 16 MB

    std::vector<MPI_Request> requests(2*iterations);

    std::ofstream fout;
    if (rank == 0)
    {
        std::string fname("results.dat");
        if (argc == 2)
        {
            fname = argv[1];
        }
        fout.open(fname);
    }
    for (size_t msg_size = min_size; msg_size <= max_size; msg_size *= 2)
    {

        //TODO: perform send/receive of message with size msg_size and measure how long 
        //      it takes for communication to complete
      std::vector<char> msg(msg_size);
	    double time = -MPI_Wtime();
        if (rank == 0) {
            // Get reliable estimate of bandwidth with averaging over iterations
            for (size_t i = 0; i < iterations; i++) {
              // Send non blocking message
              MPI_Isend(msg.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &requests[i]);

              // MPI_Send(msg.data(), 1, MPI_INT, 1, i, MPI_COMM_WORLD);
            }
        } else {
            for (size_t i = 0; i < iterations; i++) {
              // Non blocking message
              MPI_Irecv(msg.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &requests[i]);
              // MPI_Recv(msg.data(), 1, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Wait for messages to be sent and received
        MPI_Waitall(iterations, requests.data(), MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
        time += MPI_Wtime();

        if (rank == 0) 
        {
            //TODO: Compute bandwidth
            const double bandwidth = msg_size * iterations / time;
            fout << msg_size << '\t' << bandwidth << '\n';
        }
    }

    MPI_Finalize();
    return 0;
}
