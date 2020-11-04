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

#include <arm_fp16.h>
#include <stdint.h>

#ifndef APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_LINEAR_APPROXIMATION_H

// Assuming IEEE 754 that integers and floats are 32 bits.
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef float float32;
typedef _Float16 half16;

void piecewise_linear_approximation_single(unsigned int n_samples, const float32 *restrict input, float32 *restrict output);
void piecewise_linear_approximation_half(unsigned int n_samples, const half16 *restrict input, half16 *restrict output);

#define APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_LINEAR_APPROXIMATION_H

#endif //APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_LINEAR_APPROXIMATION_H
