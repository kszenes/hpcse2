BINARY = wave
CXX = mpicxx
CXXFLAGS = -g -std=c++11 -Wno-narrowing -Wall -Wextra
CXXFLAGS += -O0

.PHONY: all
all: $(BINARY)

$(BINARY) : main.o wave.o
	$(CXX) -o $@ $^ $(CXXFLAGS)

main.o: main.cpp
	$(CXX) -c $< $(CXXFLAGS)

wave.o: wave.cpp
	$(CXX) -c $< $(CXXFLAGS)

run:
	mpirun -n 4 $(BINARY) --use-hwthread-cpus 128 1.0

submit:
	bsub -n 4 -J waveSim -o waveSim.out -W 0:05 mpirun -n 4 $(BINARY) 128 1.0

.PHONY: clean
clean:
	rm $(BINARY) ./*.o ./output/*.bin ./wave.gif

.PHONY: all clean
