# CXX = CC mpicxx
CXX = mpicxx
CXXFLAGS = -g -O3 -std=c++11 -Wall -Wextra -fopenmp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
all: main mainMPI
main: main.o LaplacianSmoother.o
	$(CXX) -o $@ $^ $(CXXFLAGS)
mainMPI: mainMPI.o LaplacianSmoother.o LaplacianSmootherMPI.o
	$(CXX) -o $@ $^ $(CXXFLAGS)
clean:
	rm -rf *.o main mainMPI mainMPI-pack *.bin
run:
	mpirun -n 4 ./mainMPI
