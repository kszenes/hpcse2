#!/usr/bin/env bash
module unload gcc
module load gcc/9.3.0
module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
module load cdt-cuda
module load craype-accel-nvidia60
