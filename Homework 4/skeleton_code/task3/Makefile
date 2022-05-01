CXX = mpicxx
CXXFLAGS = -g -O3 -std=c++11 -Wall -Wextra -Wno-cast-function-type -fopenmp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main: main.o Equation2D.o WaveEquation.o
	$(CXX) -o $@ $^ $(CXXFLAGS)
run:
	export OMP_NUM_THREADS=6 ; unset LSB_AFFINITY_HOSTFILE ; mpirun -n 4 --use-hwthread-cpus --map-by node:PE=6 ./main 256 1.0
submit:
	export OMP_NUM_THREADS=1 ; unset LSB_AFFINITY_HOSTFILE ; bsub -I -n 48 -W 0:10 mpirun ./main 256 1.0
clean:
	rm -rf *.o output/* main *.gif
