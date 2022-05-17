CUFLAGS=-O3 -std=c++14 --compiler-options "-Wall -Wextra"

.PHONY: all clean

all: cell_lists
	@true

clean:
	rm -f cell_lists *.o

%.o: %.cu %.h ../include/utils.h
	nvcc $(CUFLAGS) $< -c -o $@

main.o: main.cu brute_force.h cell_list.h interaction.h scan.h
	nvcc $(CUFLAGS) $< -c -o $@

cell_lists: brute_force.o cell_list.o interaction.o main.o scan.o
	nvcc $^ -o $@

