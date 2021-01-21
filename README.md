# message_passing_project

This is a simple relief algorithm using message passing model (MPI)

It runs simple relief algorithm concurrently in several processors.

I used OpenMPI
How to compile and run example1:
mpic++ -o relief mpi_project.cpp
mpirun --oversubscribe -np 6 relief input1.tsv
