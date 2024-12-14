#ifndef COOLEY_TUKEY_PARALLEL_HPP
#define COOLEY_TUKEY_PARALLEL_HPP

#include <vector>
#include <complex>

class ParallelIterativeFFT {
    public:
        std::vector<std::complex<double>> findFFT(std::vector<std::complex<double>> input);
};

#endif // COOLEY_TUKEY_PARALLEL_HPP