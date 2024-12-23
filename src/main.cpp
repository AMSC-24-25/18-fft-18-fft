#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include "../include/Cooley-Tukey-parallel.hpp"
#include "../include/Cooley-Tukey.hpp"

#define N std::pow(2, 15) // Must be a power of 2
using namespace std;
int main(){
    struct timeval t1, t2;
    double etimePar,etimeSeq;

    //Initialize solvers
    ParallelIterativeFFT ParallelFFTSolver = ParallelIterativeFFT();
    SequentialFFT SequentialFFTSolver = SequentialFFT();

    //creating a random input vector
    srand(95);
    std::vector<std::complex<double>>input_vector;
    for(int i=0; i<N; i++)
    {
       input_vector.push_back(std::complex<double>(rand() % RAND_MAX, rand() % RAND_MAX));
    }

    std::vector<std::complex<double>> recursiveResult = SequentialFFTSolver.recursive_FFT(input_vector);

    //exec and measure of SEQUENTIAL iterativeFFT
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> iterativeResult = SequentialFFTSolver.iterative_FFT(input_vector);
    gettimeofday(&t2, NULL);
	etimeSeq = std::abs(t2.tv_usec - t1.tv_usec);
	std::cout <<"Sequential version done, took ->  " << etimeSeq << " usec." << std::endl;

    //exec and measure of PARALLEL iterativeFFT    
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> parallelResult = ParallelFFTSolver.findFFT(input_vector);
	gettimeofday(&t2, NULL);
    etimePar = std::abs(t2.tv_usec - t1.tv_usec);
	std::cout <<"Parallel version done, took ->  " << etimePar << " usec." << std::endl;

    std::cout<<"The parallel version is "<< etimeSeq/etimePar <<" times faster. "<<std::endl; 

    //exec and measure SEQUENTIAL INVERSE iterativeFFT
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> iterativeInverseResult = SequentialFFTSolver.iterative_inverse_FFT(input_vector);
	gettimeofday(&t2, NULL);
    etimePar = std::abs(t2.tv_usec - t1.tv_usec);
	std::cout <<"Inverse iterative version done, took ->  " << etimePar << " usec." << std::endl;
    
/*
  double tolerance = 1e-9; // Define an acceptable tolerance
    bool inverseCheck = true;
    for (int i = 0; i < input_vector.size(); i++) {
        if (std::abs(input_vector[i] - iterativeInverseResult[i]) > tolerance) {
            cout << "Inverse Error at index: " << i 
                << " | Original: " << input_vector[i]
                << " | Inverse: " << iterativeInverseResult[i] << endl;
            inverseCheck = false;
        }
    }

    if (inverseCheck) {
        cout << "Inverse FFT validation successful!" << endl;
    } else {
        cout << "Inverse FFT validation failed!" << endl;
    }

*/
  


    //Checking if the 3 implementations give the same results 
    std::cout << "\nChecking results... " << std::endl;
    bool check = true;
    for(int i = 0; i < recursiveResult.size(); i++){
        if(recursiveResult[i]!=iterativeResult[i] && iterativeResult[i]!=parallelResult[i])
        {
            std::cout <<"Different result in line " << i << std::endl;
            check=false;
        }
    }

    if(check)
        std::cout <<"Same result for the 3 methods" << std::endl;

    return 0;
}