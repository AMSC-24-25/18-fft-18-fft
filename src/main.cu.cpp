//#include "../include/Cooley-Tukey.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuComplex.h>
#define N 4
std::vector<std::complex<double>> recursive_FF(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> iterative_FF(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> permutation(std::vector<std::complex<double>> input);
using namespace std;


// Function to compute wd^h where wd is e^(2πi/d) and h is a complex number
__host__ __device__ cuDoubleComplex compute_wd_pow_h(int d, cuDoubleComplex h) {
    // Extract real and imaginary parts of h
    double real_h = cuCreal(h);
    double imag_h = cuCimag(h);

    // Calculate the argument of wd, which is 2π/d
    double angle = 2 * M_PI / d;

    // Compute wd^h = exp(h * ln(wd)), where ln(wd) = i * 2π/d
    double real_part = imag_h * angle;  // This is the imaginary part of the exponent
    double imag_part = real_h * angle;  // This is the real part of the exponent

    // Compute the result as e^(i * (real_part + imag_part)), which is cos(imag_part) + i * sin(imag_part)
    return make_cuDoubleComplex(cos(imag_part), sin(imag_part));
}

__global__ void parallel_fft(cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int log_n ){
    int d;
    unsigned int t_y = blockIdx.y * blockDim.y + threadIdx.y;
    bool up_down; // up = 1 down = 0 ( if down it will write on x otherwise it will write on t)
    int tmp_index;
    int p;
    for(int j= 1; j<=log_n; j++){
        d = 1 << j;
        if( (t_y % d ) < d/2 ) {up_down = 0;}
        else up_down = 1;
        

        // DOWN CASE
        if( ! up_down ) {
            tmp_index = t_y + d/2;
            //printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_y, j ,up_down, tmp_index);
            x[tmp_index] = y[t_y];
            x[t_y] = y[t_y];
        }
        //UP CASE
        else{
            tmp_index = t_y - d/2;
            //printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_y, j ,up_down, tmp_index);
            p = (int)(tmp_index % d);
            cuDoubleComplex w = compute_wd_pow_h(d, make_cuDoubleComplex(static_cast<double>(p), 0.0));
            t[tmp_index] = cuCmul(y[t_y], w);
            t[t_y] = cuCmul(y[t_y], w);;
        }

     __syncthreads();
     if(t_y % d < d/2 ) {
        y[t_y] =  cuCadd(x[t_y],t[t_y] );
     }else{
        y[t_y] = cuCsub(x[t_y],t[t_y] );
     }
      __syncthreads();
    }
}




int main(){

    std::vector<std::complex<double>> input = {std::complex<double>(1,0), std::complex<double>(2,0), std::complex<double>(4,0), std::complex<double>(3,0)
                                                , std::complex<double>(1,0), std::complex<double>(2,0), std::complex<double>(4,0), std::complex<double>(3,0)};
    std::vector<std::complex<double>> y1r = recursive_FF(input);
    std::vector<std::complex<double>> y1i = iterative_FF(input);
   

    std::cout << "RECURSIVE-------Vector 1:" << std::endl;
    for(int i = 0; i < y1r.size(); i++){
        std::cout << y1r[i] << std::endl;
    }

    std::cout << "ITERATIVE-------Vector 1:" << std::endl;
    for(int i = 0; i < y1i.size(); i++){
        std::cout << y1i[i] << std::endl;
    }

    input = permutation(input);

    cuDoubleComplex *a; //input for GPU
    cuDoubleComplex *y; //permutation for GPU
    cuDoubleComplex *x;  
    cuDoubleComplex *t;
    cuDoubleComplex *sum;

    cudaMallocManaged((void **)&a, sizeof(cuDoubleComplex) *  input.size());
    cudaMallocManaged((void **)&y, sizeof(cuDoubleComplex) *  input.size());
    cudaMallocManaged((void **)&x, sizeof(cuDoubleComplex) *  input.size());
    cudaMallocManaged((void **)&t, sizeof(cuDoubleComplex) *  input.size());

     // Copy data from std::complex to cuDoubleComplex
    for (size_t i = 0; i < input.size(); ++i) {
        a[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
    }

    dim3 dimGrid(1, 1);   // 40 columns, 2 rows
    dim3 dimBlock(1, input.size()); // 1 column, 32 rows
    int log_n = (int)(log(input.size()) / log(2)); // This will correctly return 2
    parallel_fft<<<dimGrid,dimBlock>>>(a,x,t,log_n);
    cudaDeviceSynchronize();

    std::cout << "PARALLEL-------Vector 1:" << std::endl;
    for (int i = 0; i < input.size(); i++) {
        std::cout << "(" << cuCreal(a[i]) << ", " << cuCimag(a[i]) << ")" << std::endl;
    }


    
    return 0;
}
std::vector<std::complex<double>> permutation(std::vector<std::complex<double>> input) {
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

    return y;
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