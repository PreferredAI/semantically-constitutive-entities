# the compiler: gcc for C program, define as g++ for C++
CC = gcc

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS  = -g -Wall

INCLUDES = gurobi1001/linux64/include/
CLIB = -Lgurobi1001/linux64/lib -lgurobi100


all: relaxed_partial relaxed_solver

relaxed_partial: relaxed_partial.c
	$(CC) $(CFLAGS) -I$(INCLUDES) $(CLIB) -o relaxed_partial relaxed_partial.c

relaxed_solver: relaxed_solver.c
	$(CC) $(CFLAGS) -I$(INCLUDES) $(CLIB) -o relaxed_solver relaxed_solver.c

clean:
	$(RM) relaxed_partial relaxed_solver
