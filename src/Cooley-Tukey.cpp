//#include "../include/Cooley-Tukey.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

std::vector<std::complex<double>> recursive_FF(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> iterative_FF(std::vector<std::complex<double>> x);
using namespace std;
int main(){

    std::vector<std::complex<double>> x = {std::complex<double>(1,0), std::complex<double>(2,0), std::complex<double>(4,0), std::complex<double>(3,0)};

    std::vector<std::complex<double>> y1r = recursive_FF(x);
    std::vector<std::complex<double>> y1i = iterative_FF(x);

    std::cout << "RECURSIVE-------Vector 1:" << std::endl;
    for(int i = 0; i < y1r.size(); i++){
        std::cout << y1r[i] << std::endl;
    }

    std::cout << "ITERATIVE-------Vector 1:" << std::endl;
    for(int i = 0; i < y1i.size(); i++){
        std::cout << y1i[i] << std::endl;
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