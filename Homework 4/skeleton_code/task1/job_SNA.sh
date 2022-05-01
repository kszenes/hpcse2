#!/usr/bin/env bash
#BSUB -W 00:15
#BSUB -J p2p_SAN_benchmark
#BSUB -o p2p_SAN_benchmark_%J.out
#BSUB -n 48


mpirun -np 2 --map-by ppr:1:node --report-bindings \
    ./bandwidth_sol results_SAN_${LSB_JOBID}.dat

