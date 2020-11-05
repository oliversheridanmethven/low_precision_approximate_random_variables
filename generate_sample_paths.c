/*
Author:

Oliver Sheridan-Methven, November 2020.

Description:

Produces sample paths for exact and approximate path simulations
in various precisions.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <gsl/gsl_cdf.h>

#include "piecewise_linear_approximation.h"

int main(void)
{
    unsigned int n_paths_per_level = 100;
    unsigned int max_level = 17;

    for (unsigned int level = 1; level <= max_level; level++)
    {
        unsigned int n_increments = 1 << level;
        double dt = 1.0 / n_increments;
        double dt_64 = dt;
        float32 dt_32 = dt;
        half16 dt_16 = dt;
        
        
       double x_fine_exact_64, x_coarse_exact_64;
       double weiner_increments_exact[n_increments];
       float32 weiner_increments_approx_32[n_increments];
       half16 weiner_increments_approx_16[n_increments];

        float32 x_fine_approx_32, x_coarse_approx_32;
        half16 x_fine_approx_16, x_coarse_approx_16, x_fine_approx_16_kahan, x_coarse_approx_16_kahan;
        half16 dx_fine, dx_coarse;
        half16 dx_fine_compensated, dx_coarse_compensated;
        half16 accumulation_fine, accumulation_coarse;
        half16 compensation_fine, compensation_coarse;

        for (unsigned int simulation = 0; simulation < n_paths_per_level; simulation++)
        {
            x_fine_exact_64 = x_coarse_exact_64 = x_fine_approx_32 = x_coarse_approx_32 = x_fine_approx_16 = x_coarse_approx_16 = x_fine_approx_16_kahan = x_coarse_approx_16_kahan = 1.0;
            //        Generate the Gaussian increments.
            for(unsigned int i = 0; i < n_increments; i++)
            {
                half16 uniform = 0;
                while (!uniform)
                {
                    uniform =  ((half16) (((double) (rand() + 1)) / ((double) (((unsigned long int) RAND_MAX) + 2)) / 2.0)); // We divide by 2 so all numbers are in 0.5 so we don't need the complementary inverse CDF for uniforms ro    unded up towards 1 by typecasting.
                }
                double uniforms_64 = uniform;
                float32 uniforms_32 = uniform;
                half16 uniforms_16 = uniform;
                weiner_increments_exact[i] = gsl_cdf_ugaussian_Pinv(uniforms_64);
                piecewise_linear_approximation_single(1, &uniforms_32, weiner_increments_approx_32 + i);
                piecewise_linear_approximation_half(1, &uniforms_16, weiner_increments_approx_16 + i);
            }

            //        Generate the Weiner increments.
            for(unsigned int i = 0; i < n_increments; i++)
            {
                //            We correct for the sign of the Weiner increments as the uniforms were in (0, 0.5) and not (0, 1) by randomly negating.
                bool flip_sign = (((double) (rand() + 1)) / ((double) (((unsigned long int)RAND_MAX) + 2))) > 0.5 ? true: false;
                if (flip_sign)
                {
                    weiner_increments_exact[i] = -weiner_increments_exact[i];
                    weiner_increments_approx_32[i] = -weiner_increments_approx_32[i];
                    weiner_increments_approx_16[i] = -weiner_increments_approx_16[i];
                }
                double sqrt_dt_64 = sqrt(dt);
                float32 sqrt_dt_32 = sqrt_dt_64;
                half16 sqrt_dt_16 = sqrt_dt_64;
                weiner_increments_exact[i] *= sqrt_dt_64;
                weiner_increments_approx_32[i] *= sqrt_dt_32;
                weiner_increments_approx_16[i] *= sqrt_dt_16;
            }


            compensation_fine = compensation_coarse = 0.0;
            for (unsigned int i = 0; i < n_increments; i+=2)
            {
                double mu_64 = 0.05;
                float32 mu_32 = mu_64;
                half16 mu_16 = mu_64;
                double sigma_64 = 0.2;
                float32 sigma_32 = sigma_64;
                half16 sigma_16 = sigma_64;


                //                Double precision
                x_fine_exact_64 += mu_64 * x_fine_exact_64 * dt_64 + sigma_64 * x_fine_exact_64 * weiner_increments_exact[i];
                x_fine_exact_64 += mu_64 * x_fine_exact_64 * dt_64 + sigma_64 * x_fine_exact_64 * weiner_increments_exact[i+1];
                double weiner_coarse_exact_64 = weiner_increments_exact[i] + weiner_increments_exact[i+1];
                double dt_coarse_64 = dt_64 + dt_64;
                x_coarse_exact_64 += mu_64 * x_coarse_exact_64 * dt_coarse_64 + sigma_64 * x_coarse_exact_64 * weiner_coarse_exact_64;



                //                Single precision
                x_fine_approx_32 += mu_32 * x_fine_approx_32 * dt_32 + sigma_32 * x_fine_approx_32 * weiner_increments_approx_32[i];
                x_fine_approx_32 += mu_32 * x_fine_approx_32 * dt_32 + sigma_32 * x_fine_approx_32 * weiner_increments_approx_32[i+1];
                float32 weiner_coarse_approx_32 = weiner_increments_approx_32[i] + weiner_increments_approx_32[i+1];
                float32 dt_coarse_32 = dt_32 + dt_32;
                x_coarse_approx_32 += mu_32 * x_coarse_approx_32 * dt_coarse_32 + sigma_32 * x_coarse_approx_32 * weiner_coarse_approx_32;



                //                Half precision
                x_fine_approx_16 += mu_16 * x_fine_approx_16 * dt_16 + sigma_16 * x_fine_approx_16 * weiner_increments_approx_16[i];
                x_fine_approx_16 += mu_16 * x_fine_approx_16 * dt_16 + sigma_16 * x_fine_approx_16 * weiner_increments_approx_16[i+1];
                half16 weiner_coarse_approx_16 = weiner_increments_approx_16[i] + weiner_increments_approx_16[i+1];
                half16 dt_coarse_16 = dt_16 + dt_16;
                x_coarse_approx_16 += mu_16 * x_coarse_approx_16 * dt_coarse_16 + sigma_16 * x_coarse_approx_16 * weiner_coarse_approx_16;



                //                Half precision with Kahan compensation
                dx_fine = mu_16 * x_fine_approx_16_kahan * dt_16 + sigma_16 * x_fine_approx_16_kahan * weiner_increments_approx_16[i];
                dx_fine_compensated = dx_fine - compensation_fine;
                accumulation_fine = x_fine_approx_16_kahan + dx_fine_compensated;
                compensation_fine = (accumulation_fine - x_fine_approx_16_kahan) - dx_fine;
                x_fine_approx_16_kahan = accumulation_fine;

                dx_fine = mu_16 * x_fine_approx_16_kahan * dt_16 + sigma_16 * x_fine_approx_16_kahan * weiner_increments_approx_16[i+1];
                dx_fine_compensated = dx_fine - compensation_fine;
                accumulation_fine = x_fine_approx_16_kahan + dx_fine_compensated;
                compensation_fine = (accumulation_fine - x_fine_approx_16_kahan) - dx_fine;
                x_fine_approx_16_kahan = accumulation_fine;

                dx_coarse = mu_16 * x_coarse_approx_16_kahan * dt_coarse_16 + sigma_16 * x_coarse_approx_16_kahan * weiner_coarse_approx_16;
                dx_coarse_compensated = dx_coarse - compensation_coarse;
                accumulation_coarse = x_coarse_approx_16_kahan + dx_coarse_compensated;
                compensation_coarse = (accumulation_coarse - x_coarse_approx_16_kahan) - dx_coarse;
                x_coarse_approx_16_kahan = accumulation_coarse;

            }
            //            printf("dt,x_fine_exact_64,x_coarse_exact_64,x_fine_approx_32,x_coarse_approx_32,x_fine_approx_16,x_coarse_approx_16,x_fine_approx_16_kahan,x_coarse_approx_16_kahan\n");
            printf("%f,%f,%f,%f,%f,%f,%f,%f,%f\n",dt,x_fine_exact_64,x_coarse_exact_64,(double)x_fine_approx_32,(double)x_coarse_approx_32,(double)x_fine_approx_16,(double)x_coarse_approx_16,(double)x_fine_approx_16_kahan,(double)x_coarse_approx_16_kahan);
        }

    }

}
