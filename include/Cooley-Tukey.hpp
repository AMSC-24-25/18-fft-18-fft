#ifndef HH_COOLEY_TURKEY_HH
#define HH_COOLEY_TURKEY_HH

#include <vector>
#include <complex>

class SequentialFFT{
    public:
        std::vector<std::complex<double>> recursive_FFT(std::vector<std::complex<double>> x);
        std::vector<std::complex<double>> iterative_FFT(std::vector<std::complex<double>> x);
        std::vector<std::complex<double>> iterative_inverse_FFT(std::vector<std::complex<double>> x);
};

#endif