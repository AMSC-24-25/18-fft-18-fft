//#include "../include/Cooley-Tukey.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include "../include/Cooley-Tukey-parallel.hpp"

std::vector<std::complex<double>> recursive_FF(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> iterative_FF(std::vector<std::complex<double>> x);
using namespace std;

#define N 16

int main(){
    struct timeval t1, t2;
    double etime;

    srand(95);
    std::vector<std::complex<double>> x;
    for(int i=0; i<N; i++)
    {
        x.push_back(std::complex<double>(rand() % RAND_MAX, rand() % RAND_MAX));
    }

    std::vector<std::complex<double>> y1r = recursive_FF(x);

    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> y1i = iterative_FF(x);
    gettimeofday(&t2, NULL);

	etime = (t2.tv_usec - t1.tv_usec);

	std::cout <<"not parallel done, took " << etime << " usec. Verification..." << std::endl;

    ParallelIterativeFFT parallelIterator = ParallelIterativeFFT();

    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> y1p = parallelIterator.findFFT(x);
	gettimeofday(&t2, NULL);

	etime = (t2.tv_usec - t1.tv_usec);

	std::cout <<" parallel done, took " << etime << " usec. Verification..." << std::endl;

    std::cout << "Checking results... " << std::endl;
    bool check = true;
    for(int i = 0; i < y1r.size(); i++){
        if(y1r[i]!=y1i[i] && y1i[i]!=y1p[i])
        {
            std::cout <<"Different result in line " << i << std::endl;
            check=false;
        }
    }

    if(check)
    {
        std::cout <<"Same result for the 3 methods" << std::endl;
    }

    return 0;
}

std::vector<std::complex<double>> recursive_FF(std::vector<std::complex<double>> x){
    if(x.size() == 1){ 
        return x;
        }
    else{
        int n = x.size();
        std::complex<double> wn(std::cos(2 * M_PI / n), std::sin(2 * M_PI / n)) ;
        std::complex<double> w(1,0);
        
        std::vector<std::complex<double>> x_even;
        std::vector<std::complex<double>> x_odd;
        for(int i=0; i < n; i++){
            if(i % 2 == 0){
                x_even.push_back(x[i]);
            }
            else{
                x_odd.push_back(x[i]);
            }
        }

        std::vector<std::complex<double>> y_even = recursive_FF(x_even);
        std::vector<std::complex<double>> y_odd = recursive_FF(x_odd);

        std::vector<std::complex<double>> y(n);
        for(int i = 0; i < n/2; i++){
            y[i] = y_even[i] + w * y_odd[i];
            y[i + n/2] = y_even[i] - w * y_odd[i];
            w = w * wn;
        }
        return y;
    }
}


std::vector<std::complex<double>> iterative_FF(std::vector<std::complex<double>> input) {
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
}