/*******************************************************************************
 * This file is part of the "high performance low precision vectorised         *
 * arithmetic" project, created by Oliver Sheridan-Methven 2017.               *
 *                                                                             *
 * Copyright (C) 2017 Oliver Sheridan-Methven, University of Oxford.           *
 *                                                                             *
 * Commercial users wanting to use this software should contact the author:    *
 * oliver.sheridan-methven@maths.ox.ac.uk                                      *
 *                                                                             *
 ******************************************************************************/

/*
 * Author:
 *
 *      Oliver Sheridan-Methven, November 2020.
 *
 * Description:
 *
 *      Experiments on using reduced precision formats for modified
 *      Euler-Maruyama schemes. 
 *     
 * Comments:
 *
 *      This has been built to be run on a Nvidia Xavier chip containing and
 *      8-Core ARM v8.2 64-Bit CPU, 8 MB L2 + 4 MB L3 CPU attached. Both the
 *      CPU and GPU support half-precision. 
 *      
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <gsl/gsl_rng.h>  // For the RNG.
#include <gsl/gsl_cdf.h>  // For the Gaussian inverse CDF.


#include <arm_fp16.h>  // GCC has access to this.


// Assuming double precision.
#define SDE_INITIAL_VALUE 1.0
#define SDE_DRIFT 0.05
#define SDE_VOLATILITY 0.2
#define SDE_TENURE 1.0

typedef _Float16 half;

double gbm_64(double X, const double drift, const double volatility, const double time_increment, const double normal_increment);

float gbm_32(float X, const float drift, const float volatility, const float time_increment, const float normal_increment);

half gbm_16(half X, const half drift, const half volatility, const half time_increment, const half normal_increment);

half gbm_16_k(half X, const half drift, const half volatility, const half time_increment, const half normal_increment, half * c);

double uniform_to_gaussian_64(double u);

float uniform_to_approx_gaussian_32(float u);

half uniform_to_approx_gaussian_16(half u);

int main(int argc, char ** argv)
{
    unsigned int N = atoi(argv[1]); 
    unsigned int L_min = atoi(argv[2]);
    unsigned int L_max = atoi(argv[3]);

    double recip_root_two = 1.0/sqrt(2.0);


    /* Building the RNG */
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937); // We use the Mersenne Twister.
    gsl_rng_set(rng, (unsigned long int) time(NULL)); // Seeding the RNG.

        for (unsigned int l = L_min; l <= L_max; l++)
        {
            unsigned int m = 1 << l; // The number of time steps, which increase in powers of 2.
            double xf_64, xc_64, z1_64, z2_64, zc_64;
            float xf_q32, xc_q32, z1_q32, z2_q32, zc_q32;
            half xf_q16, xc_q16, xf_q16_k, xc_q16_k, z1_q16, z2_q16, zc_q16;
            half u1, u2;
            double time_increment = SDE_TENURE / ((double) m);

            for (unsigned int n = 0; n < N; n++)
            {
                xf_64 = SDE_INITIAL_VALUE;
                xf_q32 = SDE_INITIAL_VALUE;
                xf_q16 = SDE_INITIAL_VALUE;
                xf_q16_k = SDE_INITIAL_VALUE;
                xc_64 = SDE_INITIAL_VALUE;
                xc_q32 = SDE_INITIAL_VALUE;
                xc_q16 = SDE_INITIAL_VALUE;
                xc_q16_k = SDE_INITIAL_VALUE;
                half c_f = 0.0, c_c = 0.0;  // Kahan compensation term for fine and coarse levels. 
                for (unsigned int i = 0; i < m; i++)
                {
                    do
                    {
                        u1=gsl_rng_uniform(rng);
                        u2=gsl_rng_uniform(rng);
                    }
                    while (((u1 == 0) || (u1 == 1)) || ((u2 == 0) || (u2 == 1)));
                    
                    z1_64 = uniform_to_gaussian_64(u1);
                    z1_q32 = uniform_to_approx_gaussian_32(u1);
                    z1_q16 = uniform_to_approx_gaussian_16(u1);
                    z2_64 = uniform_to_gaussian_64(u2);
                    z2_q32 = uniform_to_approx_gaussian_32(u2);
                    z2_q16 = uniform_to_approx_gaussian_16(u2);
                    xf_64 = gbm_64(xf_64, SDE_DRIFT, SDE_VOLATILITY, time_increment, z1_64);
                    xf_q32 = gbm_32(xf_q32, SDE_DRIFT, SDE_VOLATILITY, time_increment, z1_q32);
                    xf_q16 = gbm_16(xf_q16, SDE_DRIFT, SDE_VOLATILITY, time_increment, z1_q16);
                    xf_q16_k = gbm_16_k(xf_q16_k, SDE_DRIFT, SDE_VOLATILITY, time_increment, z1_q16, &c_f);
                    xf_64 = gbm_64(xf_64, SDE_DRIFT, SDE_VOLATILITY, time_increment, z2_64);
                    xf_q32 = gbm_32(xf_q32, SDE_DRIFT, SDE_VOLATILITY, time_increment, z2_q32);
                    xf_q16 = gbm_16(xf_q16, SDE_DRIFT, SDE_VOLATILITY, time_increment, z2_q16);
                    xf_q16_k = gbm_16_k(xf_q16_k, SDE_DRIFT, SDE_VOLATILITY, time_increment, z2_q16, &c_f);
                    zc_64 = (z1_64 + z2_64) * recip_root_two; // The addition is done in the appropriate precision. 
                    zc_q32 = (z1_q32 + z2_q32) * recip_root_two;
                    zc_q16 = (z1_q16 + z2_q16) * recip_root_two;
                    xc_64 = gbm_64(xc_64, SDE_DRIFT, SDE_VOLATILITY, 2.0 * time_increment, zc_64);
                    xc_q32 = gbm_32(xc_q32, SDE_DRIFT, SDE_VOLATILITY, 2.0 * time_increment, zc_q32);
                    xc_q16 = gbm_16(xc_q16, SDE_DRIFT, SDE_VOLATILITY, 2.0 * time_increment, zc_q16);
                    xc_q16_k = gbm_16_k(xc_q16_k, SDE_DRIFT, SDE_VOLATILITY, 2.0 * time_increment, zc_q16, &c_c);
                }
                //            printf("dt,x_fine_exact_64,x_coarse_exact_64,x_fine_approx_32,x_coarse_approx_32,x_fine_approx_16,x_coarse_approx_16,x_fine_approx_16_kahan,x_coarse_approx_16_kahan\n");
                //printf("%f,%f,%f,%f,%f,%f,%f,%f,%f\n",time_increment,xf_64,xc_64,(double)xf_q32,(double)xc_q32,(double)xf_q16,(double)xc_q16,(double)xf_q16_k,(double)xc_q16_k);
                printf("%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",time_increment,xf_64,xc_64,(double)xf_q32,(double)xc_q32,(double)xf_q16,(double)xc_q16,(double)xf_q16_k,(double)xc_q16_k);
            }
        }

    gsl_rng_free(rng);
}


/* Note on the GBM implementations.
 *
 *      To avoid systematic rounding errors, we first add the stochastic element, and then after this we
 *      try to add the deterministic amount. While this might not be deemed optimal to a compiler or
 *      mathematician, we do this to ensure that each arithmetic operation is acting on near random data which are
 *      i.i.d. This will ensure that the round off-errors are independent.
 *      Hence we will perform
 *      X *= (1 + b*dW) + a*dt      <--- YES
 *      and not
 *      X *= (1 + a*dt) + b*dW      <--- NO
 * */
double gbm_64(double X, const double drift, const double volatility, const double time_increment, const double normal_increment)
{
    double time_increment_sqrt = sqrt(time_increment);
    double stochastic_term = normal_increment * volatility;
    stochastic_term = fma(drift, time_increment_sqrt, stochastic_term);
    X += X * time_increment_sqrt * stochastic_term;
    return X;
}

float gbm_32(float X, const float drift, const float volatility, const float time_increment, const float normal_increment)
{
    float time_increment_sqrt = sqrtf(time_increment);
    float stochastic_term = normal_increment * volatility;
    stochastic_term = fmaf(drift, time_increment_sqrt, stochastic_term);
    X += X * time_increment_sqrt * stochastic_term;
    return X;
}

half gbm_16(half X, const half drift, const half volatility, const half time_increment, const half normal_increment)
{
    half time_increment_sqrt = vsqrth_f16(time_increment);
    half stochastic_term = normal_increment * volatility;
    stochastic_term = vfmah_f16(stochastic_term, drift, time_increment_sqrt);
    X += X * time_increment_sqrt * stochastic_term;
    return X;

}

half gbm_16_k(half X, const half drift, const half volatility, const half time_increment, const half normal_increment, half * c)
{
    /* Performing a Kahan summation. */
    half time_increment_sqrt = vsqrth_f16(time_increment);
    half stochastic_term = normal_increment * volatility;
    stochastic_term = vfmah_f16(stochastic_term, drift, time_increment_sqrt);
    half dX = X * time_increment_sqrt * stochastic_term;
    half y = dX - (*c); // Kahan compensation.
    half t = X + y;
    *c = (t - X) - y;
    return t;
}

unsigned int get_table_index_from_float_format(float u);
unsigned int get_table_index_from_half_format(half u);
float linear_approximation_32(float u, unsigned int b);
half linear_approximation_16(half u, unsigned int b);


double uniform_to_gaussian_64(const double u)
{
    return gsl_cdf_ugaussian_Pinv(u);
}


float uniform_to_approx_gaussian_32(float u)
{
    float z;
    bool predicate = u < 0.5f;
    u = predicate ? u : 1.0f - u;
    unsigned int b = get_table_index_from_float_format(u);
    z = linear_approximation_32(u, b);
    z = predicate ? z : -z;
    return z;
}

half uniform_to_approx_gaussian_16(half u)
{
    half z;
    bool predicate = u < 0.5f16;
    u = predicate ? u : 1.0f16 - u;
    unsigned int b = get_table_index_from_half_format(u);
    z = linear_approximation_16(u, b);
    z = predicate ? z : -z;
    return z;
}

#define N_MANTISSA_32 23
#define N_MANTISSA_16 10
#define FLOAT32_EXPONENT_BIAS 127
#define FLOAT16_EXPONENT_BIAS 15
#define FLOAT32_EXPONENT_BIAS_TABLE_OFFSET (FLOAT32_EXPONENT_BIAS - 1)
#define FLOAT16_EXPONENT_BIAS_TABLE_OFFSET (FLOAT16_EXPONENT_BIAS - 1)

#define TABLE_SIZE 16
#define TABLE_MAX_INDEX (TABLE_SIZE - 1) // Zero indexing...

float linear_approximation_32(float u, unsigned int b)
{

    const float poly_coef_0[TABLE_SIZE] = {0.0, -1.32705468316, -1.60211363454, -1.89517898841, -2.17029169248, -2.42545425048, -2.66295208398, -2.8853675019, -3.09492212866, -3.29342637073, -3.48234293521, -3.66285965711, -3.83595036528, -4.00242198071, -4.16295020439, -4.56405881592};
const float poly_coef_1[TABLE_SIZE] = {0.0, 2.67304493943, 3.76922290369, 6.07216678027, 10.3896821759, 18.3986090944, 33.3115138442, 61.2513770816, 113.914016285, 213.708456508, 403.695425144, 766.837343251, 1463.3480102, 2803.27422972, 5387.74589832, 21632.6362366};
    return fmaf(u, poly_coef_1[b], poly_coef_0[b]);
}


unsigned int get_table_index_from_float_format(float u)
{
    union
    {   
        uint32_t as_integer;
        float as_float;
    } u_pun;
    
    u_pun.as_float = u;
    uint32_t b;
    b = u_pun.as_integer >> N_MANTISSA_32; // Keeping only the exponent, and removing the mantissa.
    b = FLOAT32_EXPONENT_BIAS_TABLE_OFFSET - b; // Getting the table index.
    b = b > TABLE_MAX_INDEX ? TABLE_MAX_INDEX : b;  // Ensuring we don't overflow out of the table.
    return b;
}

half linear_approximation_16(half u, unsigned int b)
{

    const half poly_coef_0[TABLE_SIZE] = {0.0, -1.32705468316, -1.60211363454, -1.89517898841, -2.17029169248, -2.42545425048, -2.66295208398, -2.8853675019, -3.09492212866, -3.29342637073, -3.48234293521, -3.66285965711, -3.83595036528, -4.00242198071, -4.16295020439, -4.56405881592};
const half poly_coef_1[TABLE_SIZE] = {0.0, 2.67304493943, 3.76922290369, 6.07216678027, 10.3896821759, 18.3986090944, 33.3115138442, 61.2513770816, 113.914016285, 213.708456508, 403.695425144, 766.837343251, 1463.3480102, 2803.27422972, 5387.74589832, 21632.6362366};
    return vfmah_f16(poly_coef_0[b], u, poly_coef_1[b]);
}


unsigned int get_table_index_from_half_format(half u)
{
        union
    {
        uint16_t as_integer;
        half as_float;
    } u_pun;

    u_pun.as_float = u;
    uint16_t b;
    b = u_pun.as_integer >> N_MANTISSA_16; // Keeping only the exponent, and removing the mantissa.
    b = FLOAT16_EXPONENT_BIAS_TABLE_OFFSET - b; // Getting the table index.
    b = b > TABLE_MAX_INDEX ? TABLE_MAX_INDEX : b;  // Ensuring we don't overflow out of the table.
    return b;
}
