// Author:
//
//      Oliver Sheridan-Methven, November 2020.
//
// Description:
//
//      A piecewise linear approximation to
//      the Gaussian's inverse cumulative distribution function.
// 


#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "piecewise_linear_approximation.h"
#include "piecewise_linear_approximation_coefficients.h"

// For IEEE 754
#define N_MANTISSA_32 23
#define N_MANTISSA_16 10
#define FLOAT32_EXPONENT_BIAS 127
#define FLOAT16_EXPONENT_BIAS 15
#define FLOAT32_EXPONENT_BIAS_TABLE_OFFSET (FLOAT32_EXPONENT_BIAS - 1)
#define FLOAT16_EXPONENT_BIAS_TABLE_OFFSET (FLOAT16_EXPONENT_BIAS - 1)

#define TABLE_MAX_INDEX (TABLE_SIZE - 1) // Zero indexing...

#pragma omp declare simd
static inline float32 polynomial_approximation_32(float32 u, uint32 b)
{
    return poly_coef_32_0[b] + poly_coef_32_1[b] * u;
}

#pragma omp declare simd
static inline half16 polynomial_approximation_16(half16 u, uint16 b)
{
    return poly_coef_16_0[b] + poly_coef_16_1[b] * u;
}


#pragma omp declare simd
static inline uint32 get_table_index_from_float_32_format(float32 u)
{
    union
    {
        uint32 as_integer;
        float32 as_float;
    } u_pun;

    u_pun.as_float = u;
    uint32 b;
    b = u_pun.as_integer >> N_MANTISSA_32; // Keeping only the exponent, and removing the mantissa.
    b = FLOAT32_EXPONENT_BIAS_TABLE_OFFSET - b; // Getting the table index.
    b = b > TABLE_MAX_INDEX ? TABLE_MAX_INDEX : b;  // Ensuring we don't overflow out of the table.
    return b;
}


#pragma omp declare simd
static inline uint16 get_table_index_from_float_16_format(half16 u)
{
    union
    {
        uint16 as_integer;
        half16 as_float;
    } u_pun;

    u_pun.as_float = u;
    uint16 b;
    b = u_pun.as_integer >> N_MANTISSA_16; // Keeping only the exponent, and removing the mantissa.
    b = FLOAT16_EXPONENT_BIAS_TABLE_OFFSET - b; // Getting the table index.
    b = b > TABLE_MAX_INDEX ? TABLE_MAX_INDEX : b;  // Ensuring we don't overflow out of the table.
    return b;
}


void piecewise_linear_approximation_single(unsigned int n_samples, const float32 *restrict input, float32 *restrict output)
{
    #pragma omp simd
    for (unsigned int i = 0; i < n_samples; i++)
    {
        float32 u, z;
        u = input[i];
        bool predicate = u < 0.5f;
        u = predicate ? u : 1.0f - u;
        uint32 b = get_table_index_from_float_32_format(u);
        z = polynomial_approximation_32(u, b);
        z = predicate ? z : -z;
        output[i] = z;
    }
}

void piecewise_linear_approximation_half(unsigned int n_samples, const half16 *restrict input, half16 *restrict output)
{
    #pragma omp simd
    for (unsigned int i = 0; i < n_samples; i++)
    {
        half16 u, z;
        u = input[i];
        bool predicate = u < 0.5f;
        u = predicate ? u : 1.0f - u;
        uint16 b = get_table_index_from_float_16_format(u);
        z = polynomial_approximation_16(u, b);
        z = predicate ? z : -z;
        output[i] = z;
    }
}



