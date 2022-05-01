// File       : benchmark.cpp
// Created    : Tue Apr 13 2021 05:31:24 PM (+0200)
// Description: Benchmark for MPI point-to-point bandwidth
// Copyright 2021 ETH Zurich. All Rights Reserved.

#include <fstream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (2 != size) {
        MPI_Finalize();
        throw std::runtime_error("Need exactly two ranks for this benchmark");
    }

    double t_elapsed = 0;
    size_t iterations = 1000;
    constexpr size_t msg_sequence = 64;
    constexpr size_t min_buf_size = 1;        // 1 byte
    constexpr size_t max_buf_size = 1 << 24;  // 16 Mbyte
    std::vector<char> send_buf(max_buf_size);
    std::vector<char> recv_buf(max_buf_size);
    std::vector<MPI_Request> requests(msg_sequence);

    std::ofstream fout;
    if (0 == rank) {
        std::string fname("results.dat");
        if (2 == argc) {
            fname = argv[1];
        }
        fout.open(fname);
    }
    for (size_t msg_size = min_buf_size; msg_size <= max_buf_size;
         msg_size *= 2) {
        t_elapsed = 0;

        if (msg_size > (1 << 20)) {
            iterations = 50; // measurements are less noisy at this message size
        }

        if (0 == rank) {
            const double t_start = MPI_Wtime();
            for (size_t i = 0; i < iterations; ++i) {
                for (size_t j = 0; j < msg_sequence; ++j) {
                    // see MPI standard section 3.5 for message order and
                    // non-overtaking messages.  We send msg_sequence
                    // consecutive messages with non-blocking semantics to
                    // emulate sustained bandwidth at the given message size.
                    MPI_Isend(send_buf.data(),
                              msg_size,
                              MPI_CHAR,
                              1,
                              100,
                              MPI_COMM_WORLD,
                              &requests[j]);
                }
                // await the pending requests
                MPI_Waitall(msg_sequence, requests.data(), MPI_STATUSES_IGNORE);

                // wait for a handshake from the receiver (blocking)
                MPI_Recv(recv_buf.data(),
                         4,
                         MPI_CHAR,
                         1,
                         200,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
            t_elapsed = MPI_Wtime() - t_start;
        } else if (1 == rank) {
            for (size_t i = 0; i < iterations; ++i) {
                for (size_t j = 0; j < msg_sequence; ++j) {
                    MPI_Irecv(recv_buf.data(),
                              msg_size,
                              MPI_CHAR,
                              0,
                              100,
                              MPI_COMM_WORLD,
                              &requests[j]);
                }
                // await the pending requests
                MPI_Waitall(msg_sequence, requests.data(), MPI_STATUSES_IGNORE);

                // notify sender that messages have been received (blocking)
                MPI_Send(send_buf.data(), 4, MPI_CHAR, 0, 200, MPI_COMM_WORLD);
            }
        }

        if (0 == rank) {
            const double total_Mbyte =
                msg_size * iterations * msg_sequence / 1.0e6;
            fout << msg_size << '\t' << total_Mbyte / t_elapsed << '\n';
        }
    }

    MPI_Finalize();
    return 0;
}

