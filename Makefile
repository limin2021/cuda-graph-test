# Makefile
TARGET	= graphs_bench
CC	= g++
CFLAGS	=  -L /usr/local/cuda/lib64 -I /usr/local/cuda/include

all:
	$(CC) -O3 -c -o timer.o timer.cpp 
	nvcc -O3 -o $(TARGET) $(CFLAGS) graphs_bench.cu timer.o 

clean:
	rm -f $(TARGET) *.o

