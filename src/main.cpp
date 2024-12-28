#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include "../include/Cooley-Tukey-parallel.hpp"
#include "../include/Cooley-Tukey.hpp"

#define N std::pow(2, 15) // Must be a power of 2

int main() {
    struct timeval t1, t2;
    double etimePar, etimeSeq;

    // Initialize solvers
    ParallelIterativeFFT ParallelFFTSolver = ParallelIterativeFFT();
    SequentialFFT SequentialFFTSolver = SequentialFFT();

    // Creating a random input vector with normalized values
    srand(95);
    std::vector<std::complex<double>> input_vector;
    for (int i = 0; i < N; i++) {
        input_vector.push_back(std::complex<double>((rand() % 100) / 100.0, (rand() % 100) / 100.0));
    }

    // Perform sequential recursive FFT
    std::vector<std::complex<double>> recursiveResult = SequentialFFTSolver.recursive_FFT(input_vector);

    // Measure SEQUENTIAL iterative FFT
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> iterativeResult = SequentialFFTSolver.iterative_FFT(input_vector);
    gettimeofday(&t2, NULL);
    etimeSeq = std::abs(t2.tv_usec - t1.tv_usec);
    std::cout << "Sequential version done, took ->  " << etimeSeq << " usec." << std::endl;

    // Measure PARALLEL iterative FFT
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> parallelResult = ParallelFFTSolver.findFFT(input_vector);
    gettimeofday(&t2, NULL);
    etimePar = std::abs(t2.tv_usec - t1.tv_usec);
    std::cout << "Parallel version done, took ->  " << etimePar << " usec." << std::endl;

    std::cout << "The parallel version is " << etimeSeq / etimePar << " times faster. " << std::endl;

    // Measure SEQUENTIAL INVERSE iterative FFT
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> iterativeInverseResult = SequentialFFTSolver.iterative_inverse_FFT(iterativeResult);
    gettimeofday(&t2, NULL);
    etimeSeq = std::abs(t2.tv_usec - t1.tv_usec);
    std::cout << "Inverse iterative version done, took ->  " << etimeSeq << " usec." << std::endl;

    // Check if inverse FFT reconstructs the original input
    std::cout << "\nVerifying Inverse FFT reconstruction...\n";
    bool inverseCheck = true;
    for (int i = 0; i < input_vector.size(); i++) {
        if (std::abs(input_vector[i] - iterativeInverseResult[i]) > 1e-6) {
            std::cout << "Inverse FFT mismatch at index " << i
                      << " | Original: " << input_vector[i]
                      << ", Reconstructed: " << iterativeInverseResult[i] << std::endl;
            inverseCheck = false;
            break;
        }
    }
    if (inverseCheck) {
        std::cout << "Inverse FFT successfully reconstructed the original input." << std::endl;
    } else {
        std::cout << "Inverse FFT did not match the original input." << std::endl;
    }

    // Verify consistency among FFT implementations
    std::cout << "\nChecking results of FFT implementations...\n";
    bool fftCheck = true;
    for (int i = 0; i < recursiveResult.size(); i++) {
        if (std::abs(recursiveResult[i] - iterativeResult[i]) > 1e-6 ||
            std::abs(iterativeResult[i] - parallelResult[i]) > 1e-6) {
            std::cout << "FFT mismatch at index " << i
                      << " | Recursive: " << recursiveResult[i]
                      << ", Iterative: " << iterativeResult[i]
                      << ", Parallel: " << parallelResult[i] << std::endl;
            fftCheck = false;
            break;
        }
    }
    if (fftCheck) {
        std::cout << "FFT results are consistent across implementations." << std::endl;
    } else {
        std::cout << "FFT results are inconsistent across implementations." << std::endl;
    }

    return 0;
}