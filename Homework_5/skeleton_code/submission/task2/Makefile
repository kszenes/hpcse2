CUFLAGS=-O3 -std=c++11 --compiler-options "-Wall -Wextra" -I../include -lineinfo

.PHONY: all clean

all: force_kernel_0 force_kernel_1 force_kernel_2 force_kernel_3
	@true

clean:
	rm -f *.o force_kernel_0 force_kernel_1

%.o: %.cu
	nvcc $(CUFLAGS) $< -c -o $@

%: %.o main.o
	nvcc $(CUFLAGS) $^ -o $@
