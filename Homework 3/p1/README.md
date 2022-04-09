### Exercise 3 -- Task 1

The total number of grid points per side length must be divisible by the number of procs in each dimension:
> Good: Ntot = 256, nProcs = 4, procs_per_dim = 2 \
> Bad:  Ntot = 512, nProcs = 9, procs_per_dim = 3

During the timestepping the squared L2-norm is printed to the console. If the program is correct, it should remain in the same order of magnitude over time.
In this section of the code one can also alter the frequency with which the grid is being stored to './output'.

Use the Makefile to compile (`make`) and run the program on the current node with (`make run`).
Or you can directly submit a job with (`make submit`).

By running `python3 plot.py` you can generate an animation '.gif' with the data dumped in the 'output/' directory by your program.

Load following modules on euler:

```
. /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0  python/3.6.4  cmake/3.9.4 openmpi/4.0.2
export MPICXX=`which mpicxx`
```
