// Author:
//
//      Oliver Sheridan-Methven, November 2020.
//
// Description:
//
//      Timing the performance for approximations to the
//      Gaussian's inverse cumulative distribution function.

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_cdf.h>
#include "piecewise_linear_approximation.h"

int main(int argc, char **argv)
{
    /* We don't want to hold to many numbers in memory at once, so we break them into batches. */
    unsigned int samples_in_batch = 512 * 128;
    unsigned int n_batches = 1000;
    unsigned int total_number_of_samples = samples_in_batch * n_batches;
    double input_64[samples_in_batch];
    double output_64[samples_in_batch];
    float32 input_32[samples_in_batch];
    float32 output_32[samples_in_batch];
    half16 input_16[samples_in_batch];
    half16 output_16[samples_in_batch];
    for (unsigned int i = 0; i < samples_in_batch; i++)
    {   /* Random numbers in the range (0, 1), which are non-inclusive. */
        input_64[i] = input_32[i] = input_16[i] = (double) ((unsigned long int) rand() + 1) / (double) ((unsigned long int) RAND_MAX + 2);
    }
    clock_t run_time;
    double elapsed_time;


    run_time = clock();
    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
        piecewise_linear_approximation_single(samples_in_batch, input_32, output_32);
    }
    elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
    printf("Average time for the approximate function (single): %g s.\n", elapsed_time / total_number_of_samples);

    run_time = clock();
    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
        piecewise_linear_approximation_half(samples_in_batch, input_16, output_16);
    }
    elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
    printf("Average time for the approximate function (half): %g s.\n", elapsed_time / total_number_of_samples);

    run_time = clock();
    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
        for (unsigned int sample = 0; sample < samples_in_batch; sample++)
        {
            output_64[sample] = gsl_cdf_ugaussian_Pinv(input_64[sample]);
        }
    }
    elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
    printf("Average time for the exact (GSL) function (double): %g s.\n", elapsed_time / total_number_of_samples);

    run_time = clock();
    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
        memcpy(output_16, input_16, sizeof(half16) * samples_in_batch);
    }
    elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
    printf("Average time for the memcpy function (half): %g s.\n", elapsed_time / total_number_of_samples);
}
