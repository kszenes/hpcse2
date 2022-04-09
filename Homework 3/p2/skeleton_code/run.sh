#!/bin/bash

make -j4 && mpirun --use-hwthread-cpus -n 2 messages
