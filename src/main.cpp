#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include "../include/Cooley-Tukey-parallel.hpp"
#include "../include/Cooley-Tukey.hpp"

#define N 4096 // Must be a power of 2

int main(){
    struct timeval t1, t2;
    double etimePar,etimeSeq;

    //Initialize solvers
    ParallelIterativeFFT ParallelFFTSolver = ParallelIterativeFFT();
    SequentialFFT SequentialFFTSolver = SequentialFFT();

    //creating a random input vector
    srand(95);
    std::vector<std::complex<double>> x;
    for(int i=0; i<N; i++)
    {
        x.push_back(std::complex<double>(rand() % RAND_MAX, rand() % RAND_MAX));
    }

    std::vector<std::complex<double>> y1r = SequentialFFTSolver.recursive_FFT(x);

    //exec and measure of SEQUENTIAL iterativeFFT
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> y1i = SequentialFFTSolver.iterative_FFT(x);
    gettimeofday(&t2, NULL);
	etimeSeq = (t2.tv_usec - t1.tv_usec);
	std::cout <<"Not parallel version done, took ->  " << etimeSeq << " usec." << std::endl;

    //exec and measure of PARALLEL iterativeFFT    
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> y1p = ParallelFFTSolver.findFFT(x);
	gettimeofday(&t2, NULL);
	etimePar = (t2.tv_usec - t1.tv_usec);
	std::cout <<"Parallel version done, took ->  " << etimePar << " usec." << std::endl;

    std::cout<<"The parallel version is "<< etimeSeq/etimePar <<" times faster. "<<std::endl; 


    //Checking if the 3 implementations give the same results 
    std::cout << "\nChecking results... " << std::endl;
    bool check = true;
    for(int i = 0; i < y1r.size(); i++){
        if(y1r[i]!=y1i[i] && y1i[i]!=y1p[i])
        {
            std::cout <<"Different result in line " << i << std::endl;
            check=false;
        }
    }

    if(check)
        std::cout <<"Same result for the 3 methods" << std::endl;

    return 0;
}