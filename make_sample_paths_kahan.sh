gcc -O0 -Wall -march=armv8.2-a+fp16 -I. qmlmc_kahan.c -o kahan_paths -lm -lgsl -lgslcblas
