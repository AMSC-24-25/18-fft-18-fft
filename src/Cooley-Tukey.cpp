//#include "../include/Cooley-Tukey.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

std::vector<std::complex<double>> recursive_FF(std::vector<std::complex<double>> x);

int main(){

    std::vector<std::complex<double>> x = {std::complex<double>(1,0), std::complex<double>(2,0), std::complex<double>(3,0), std::complex<double>(4,0)};
    std::vector<std::complex<double>> x2 = {std::complex<double>(1,0), std::complex<double>(2,0), std::complex<double>(4,0), std::complex<double>(3,0)};
    

    std::vector<std::complex<double>> y = recursive_FF(x);
    std::vector<std::complex<double>> y2 = recursive_FF(x2);

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
        //std::cout<<"Caso Base:"<< x[0] <<std::endl;
        return x;
        }
    else{
        int n = x.size();
        std::complex<double> wn(std::cos(2 * M_PI / n), std::sin(2 * M_PI / n)) ;
        std::complex<double> w(1,0);
        
        std::vector<std::complex<double>> x_even(n/2);
        std::vector<std::complex<double>> x_odd(n/2);
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
        for(int i = 0; i < n/2-1; i++){
            y[i] = y_even[i] + w * y_odd[i];
            y[i + n/2] = y_even[i] - w * y_odd[i];
            w = w * wn;
        }
        return y;
    }
}
