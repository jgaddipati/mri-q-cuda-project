NVCC    = nvcc
CC	= gcc
CPP	= g++
EXE	= mri-q-cuda
OBJ	= file.o parboil.o args.o main.o

APP_CXXFLAGS = -O3 -funroll-all-loops -ffast-math
APP_CFLAGS   = -O3 -funroll-all-loops -ffast-math
APP_LDFLAGS  = -lm -lstdc++
NVCC_FLAGS   = -O3 -I/usr/local/cuda/include
LD_FLAGS     = -lcudart -L/usr/local/cuda/lib64
PTXAS_FLAGS  = --ptxas-options=--verbose

LANGUAGE=cuda

default: $(EXE)

main.o: main.cu computeQ.cu parboil.h
	$(NVCC) $(PTXAS_FLAGS) -c -o $@ main.cu $(NVCC_FLAGS)

parboil.o: parboil.c parboil.h
	$(CC) $(APP_CFLAGS) -c parboil.c

file.o: file.cc file.h
	$(CPP) $(APP_CXXFLAGS) -c file.cc

args.o: args.c
	$(CC) $(APP_CFLAGS) -c args.c

mri-q-cuda: main.o file.o args.o parboil.o
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -f *.o $(EXE)
