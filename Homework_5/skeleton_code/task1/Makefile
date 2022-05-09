CUFLAGS = -O3 -std=c++11 --compiler-options "-Wall -Wextra" -I../include \
					-use_fast_math -lcublas -gencode=arch=compute_60,code=sm_60 \
					-lineinfo
.PHONY: all clean

all: my_dgemm_1.exe my_dgemm_2.exe my_dgemm_3.exe
	@true

ptx: my_dgemm_1.ptx my_dgemm_2.ptx my_dgemm_3.ptx
	@true

clean:
	rm -f *.o *.ptx *.exe

%.o: %.cu
	nvcc $(CUFLAGS) $< -c -o $@

%.exe: %.o main.o
	nvcc $(CUFLAGS) $^ -o $@

%.ptx: %.cu
	nvcc $(CUFLAGS) -ptx $^ -o $@
