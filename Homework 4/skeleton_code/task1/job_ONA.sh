#!/usr/bin/env bash
#BSUB -W 00:15
#BSUB -J p2p_OCN_benchmark
#BSUB -o p2p_OCN_benchmark_%J.out
#BSUB -n 24


mpirun -np 2  --map-by ppr:1:core --report-bindings \
    ./bandwidth_sol results_OCN_${LSB_JOBID}.dat

