gcc -O0 -Wall -fopenmp -march=armv8.2-a+fp16 -I. generate_sample_paths.c piecewise_linear_approximation.c -o sample_paths -lm -lgsl -lgslcblas
