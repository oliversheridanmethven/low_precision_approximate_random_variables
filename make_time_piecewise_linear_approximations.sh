gcc -I. -fopenmp -Wall -march=armv8.2-a+fp16 -Ofast -c piecewise_linear_approximation.c
ar rcs libpiecewise_approximations.a piecewise_linear_approximation.o
gcc -I. -O0 -c time_piecewise_linear_approximations.c
gcc -I. -L. -o time_piecewise_linear_approximation time_piecewise_linear_approximations.o -lgsl -lgslcblas -lpiecewise_approximations
