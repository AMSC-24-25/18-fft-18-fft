#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>
#include "../include/Cooley-Tukey-parallel.hpp"

std::vector<std::complex<double>> ParallelIterativeFFT::findFFT(std::vector<std::complex<double>> input){
    int n = input.size();
    int m = log2(n);
    std::vector<std::complex<double>> y(n);

    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int k = 0; k < m; k++) {
            if (i & (1 << k)) {
                j |= (1 << (m - 1 - k));
            }
        }
        y[j] = input[i];
    }

    // Iterative FFT
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            for (int j = 1; j <= m; j++) {
                int d = 1 << j;           
                std::complex<double> wd(std::cos(2 * M_PI / d), std::sin(2 * M_PI / d));

                #pragma omp parallel for schedule(static)
                for (int k = 0; k < d / 2; k++) {
                    std::complex<double> w = std::pow(wd, k); // Precompute the value of w for this task
                    for (int i = k; i < n; i += d) {
                        std::complex<double> t = w * y[i + d / 2];      
                        std::complex<double> x = y[i];
                        y[i] = x + t;
                        y[i + d / 2] = x - t;
                    }
                }
            }
        }
        //#pragma omp taskwait
    }
    return y;
};