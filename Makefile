CC = g++ #alternative: nvcc for cuda
ALG = Top

CFLAGS += -I./tools/eigen/ -I./tools
CFLAGS += -std=c++11 
CFLAGS += -Wall
CFLAGS += -O3 
CFLAGS += -fopenmp

all: top

$(ALG).o: $(ALG).cc
	$(CC) $(CFLAGS) -c $(ALG).cc
main.o: main.cc OptionParser.cpp
	$(CC) $(CFLAGS) -c main.cc
problem.o: problem.cc
	$(CC) $(CFLAGS) -c problem.cc
top: main.o problem.o $(ALG).o OptionParser.o
	$(CC) $(CFLAGS) main.o problem.o $(ALG).o OptionParser.o -o top
clean:
	rm -f *.o
