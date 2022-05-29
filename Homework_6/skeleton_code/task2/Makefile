CUFLAGS=-O3 -std=c++14 -lineinfo --compiler-options "-Wall -Wextra"

.PHONY: all clean

all: inscan
	@true

clean:
	rm -f inscan *.o

%.o: %.cu %.h ../include/utils.h
	nvcc $(CUFLAGS) $< -c -o $@

main.o: main.cu scan.h scan_cub.h ../include/utils.h
	nvcc $(CUFLAGS) $< -c -o $@

inscan: main.o scan.o scan_cub.o
	nvcc $^ -o $@
