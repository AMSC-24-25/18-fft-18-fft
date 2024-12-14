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
        for (int j = 1; j <= m; j++) {
            int d = 1 << j;           
            std::complex<double> w(1, 0);
            std::complex<double> wd(std::cos(2 * M_PI / d), std::sin(2 * M_PI / d));
            for (int k = 0; k < d/2; k ++) {
                for (int m = k; m < n; m += d) {
                    std::complex<double> t = w * y[m + d/2];      
                    std::complex<double> x = y[m];
                    y[m] = x + t;
                    y[m + d/2] = x - t;
                    
                }         
            w = w * wd;
            }
        }
    
    return y;
};