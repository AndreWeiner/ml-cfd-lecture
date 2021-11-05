bubble: bubble_axis_symmetric.c
	qcc -Wall -O2 -D_MPI=1 -I/usr/include/openmpi/ bubble_axis_symmetric.c -o bubble -lm -lmpi

native: bubble_axis_symmetric.c
	qcc -source -D_MPI=1 bubble_axis_symmetric.c

clean:
	rm -f bubble
