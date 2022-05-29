CUFLAGS=-O3 -std=c++14 -lineinfo --compiler-options "-Wall -Wextra"

.PHONY: all clean

all: reduction_sum reduction_argmax
	@true

clean:
	rm -f reduction_sum reduction_argmax

reduction_sum: reduction_sum.cu ../include/utils.h _reduction_sum.h
	nvcc $(CUFLAGS) $< -o $@

reduction_argmax: reduction_argmax.cu ../include/utils.h _reduction_argmax.h
	nvcc $(CUFLAGS) $< -o $@
