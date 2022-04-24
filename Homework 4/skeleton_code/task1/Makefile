bw_bench: bandwidth.cpp
	# ${CC} -O3 -o $@ $<
	mpic++ -O3 -o $@ $<

clean:
	rm -f bw_bench result.dat
submit_1:
	bsub -n 24 -I -R fullnode -R "span[ptile=24]" -W 0:10 mpirun -n 2 -bind-to hwthread --report-bindings --map-by ppr:2:socket ./bw_bench

submit_2:
	bsub -n 24 -I -R "span[ptile=12]" -W 0:10 mpirun -n 2 --report-bindings -bind-to hwthread --map-by ppr:1:node ./bw_bench
