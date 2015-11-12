CC = g++ #alternative: nvcc for cuda
ALG = Top

CFLAGS += -I./tools/eigen/ -I./tools
CFLAGS += -std=c++11 
CFLAGS += -Wall
CFLAGS += -O3 
CFLAGS += -fopenmp

all: train

$(ALG).o: $(ALG).cc
	$(CC) $(CFLAGS) -c $(ALG).cc
main.o: main.cc
	$(CC) $(CFLAGS) -c main.cc
problem.o: problem.cc
	$(CC) $(CFLAGS) -c problem.cc
train: main.o problem.o $(ALG).o
	$(CC) $(CFLAGS) main.o problem.o $(ALG).o -o train -lboost_system
clean:
	rm -f *.o
