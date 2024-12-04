//#include "../include/Cooley-Tukey.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

std::vector<std::complex<double>> recursive_FF(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> iterative_FF(std::vector<std::complex<double>> x);

int main(){

    std::vector<std::complex<double>> x = {std::complex<double>(1,0), std::complex<double>(2,0), std::complex<double>(3,0), std::complex<double>(4,0)};
    std::vector<std::complex<double>> x2 = {std::complex<double>(1,0), std::complex<double>(4,0), std::complex<double>(237,0), std::complex<double>(3,0)};
    

    std::vector<std::complex<double>> y = recursive_FF(x2);
    std::vector<std::complex<double>> y2 = iterative_FF(x2);

    for(int i = 0; i < y.size(); i++){
        std::cout << y[i] << std::endl;
    }
    std::cout << "----------------Vector 2:" << std::endl;
    for(int i = 0; i < y2.size(); i++){
        std::cout << y2[i] << std::endl;
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

std::vector<std::complex<double>> iterative_FF(std::vector<std::complex<double>> x) {
    int n = x.size();
    int m = log2(n);
    std::vector<std::complex<double>> y(n);

    // Bit-reversal permutation
    for (int i = 0; i < n; ++i) {
        int j = 0;
        for (int k = 0; k < m; ++k) {
            if (i & (1 << k)) {
                j |= (1 << (m - 1 - k));
            }
        }
        y[j] = x[i];
    }

    // Iterative FFT
    for (int s = 1; s <= m; ++s) {
        int m = 1 << s;
        std::complex<double> wm(std::cos(2 * M_PI / m), std::sin(2 * M_PI / m));
        for (int k = 0; k < n; k += m) {
            std::complex<double> w(1, 0);
            for (int j = 0; j < m / 2; ++j) {
                std::complex<double> t = w * y[k + j + m / 2];
                std::complex<double> u = y[k + j];
                y[k + j] = u + t;
                y[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }

    return y;
}