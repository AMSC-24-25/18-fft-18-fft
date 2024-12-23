//#include "../include/Cooley-Tukey.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuComplex.h>
#define N std::pow(2, 10) // Must be a power of 2
#define THREAD_PER_BLOCK  1024 //this could be changed depends on GPU

std::vector<std::complex<double>> recursive_FF(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> iterative_FF(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> permutation(std::vector<std::complex<double>> input);
using namespace std;



/**
 * @brief Compares two vectors of complex numbers with a specified tolerance.
 * 
 * @param y1i Vector of complex numbers representing the first dataset.
 * @param cuda_output_vector Vector of complex numbers representing the second dataset.
 * @param tolerance Tolerance value for comparing real and imaginary parts.
 * @return true if the vectors are the same within the specified tolerance, false otherwise.
 */
bool compareComplexVectors(const std::vector<std::complex<double>>& y1i,
                           const std::vector<std::complex<double>>& cuda_output_vector,
                           double tolerance = 1e-3) {
    if (y1i.size() != cuda_output_vector.size()) {
        std::cerr << "Vectors have different sizes!\n";
        return false;
    }

    for (size_t i = 0; i < y1i.size(); i++) {
        // Get the real and imaginary parts
        double real_diff = std::abs(y1i[i].real() - cuda_output_vector[i].real());
        double imag_diff = std::abs(y1i[i].imag() - cuda_output_vector[i].imag());

        // Check if the differences in both real and imaginary parts are within the tolerance
        if (real_diff > tolerance || imag_diff > tolerance) {
            std::cout.precision(15);
            std::cout << "y1i[" << i << "] = " << y1i[i] 
                      << ", cuda[" << i << "] = " << cuda_output_vector[i]
                      << ", real diff = " << real_diff
                      << ", imag diff = " << imag_diff << std::endl;
            std::cout << "Different result in line " << i << std::endl;
            return false;  // Return false on the first mismatch
        }
    }

    std::cout << "Same result for the methods" << std::endl;
    return true;
}


// Function to convert cuDoubleComplex* to std::vector<std::complex<double>>
std::vector<std::complex<double>> cuDoubleComplexToVector(const cuDoubleComplex* a, size_t size) {
    std::vector<std::complex<double>> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::complex<double>(cuCreal(a[i]), cuCimag(a[i]));
    }
    return result;
}
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
    unsigned int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    bool up_down; // up = 1 down = 0 ( if down it will write on x otherwise it will write on t)
    int tmp_index;
    int p;
    for(int j= 1; j<=log_n; j++){
        d = 1 << j;
        if( (t_x % d ) < d/2 ) {up_down = 0;}
        else up_down = 1;
        

        // DOWN CASE
        if( ! up_down ) {
            tmp_index = t_x + d/2;
            //printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_x, j ,up_down, tmp_index);
            x[tmp_index] = y[t_x];
            x[t_x] = y[t_x];
        }
        //UP CASE
        else{
            tmp_index = t_x - d/2;
            //printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_x, j ,up_down, tmp_index);
            p = (int)(tmp_index % d);
            cuDoubleComplex w = compute_wd_pow_h(d, make_cuDoubleComplex(static_cast<double>(p), 0.0));
            t[tmp_index] = cuCmul(y[t_x], w);
            t[t_x] = cuCmul(y[t_x], w);
        }

     __syncthreads();
     if(t_x % d < d/2 ) {
        y[t_x] =  cuCadd(x[t_x],t[t_x] );
     }else{
        y[t_x] = cuCsub(x[t_x],t[t_x] );
     }
      __syncthreads();
    }
}


int main(){

    srand(95);
    std::vector<std::complex<double>> input;
    for(int i = 0; i < N; i++) {
        // Generate random numbers between -1.0 and 1.0
        double real_part = (rand() % (RAND_MAX + 1)) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        double imag_part = (rand() % (RAND_MAX + 1)) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        input.push_back(std::complex<double>(real_part, imag_part));
    }

    int grid_size = input.size() / THREAD_PER_BLOCK;
    if(input.size() % THREAD_PER_BLOCK != 0) grid_size ++ ;

    std::vector<std::complex<double>> y1r = recursive_FF(input);
    std::vector<std::complex<double>> y1i = iterative_FF(input);
   


    input = permutation(input);

    cuDoubleComplex *a; //input for GPU
    cuDoubleComplex *y; //permutation for GPU
    cuDoubleComplex *x;  
    cuDoubleComplex *t;

    cudaMallocManaged((void **)&a, sizeof(cuDoubleComplex) *  ( THREAD_PER_BLOCK * grid_size ));
    cudaMallocManaged((void **)&y, sizeof(cuDoubleComplex) *  ( THREAD_PER_BLOCK * grid_size ));
    cudaMallocManaged((void **)&x, sizeof(cuDoubleComplex) *  ( THREAD_PER_BLOCK * grid_size ));
    cudaMallocManaged((void **)&t, sizeof(cuDoubleComplex) *  ( THREAD_PER_BLOCK * grid_size ));

    // Copy data from std::complex to cuDoubleComplex
    for (size_t i = 0; i < ( THREAD_PER_BLOCK * grid_size ); ++i) {
        if(i < input.size())  a[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
        else a[i] = make_cuDoubleComplex(0,0);
    }

    dim3 dimGrid(grid_size);   // 40 columns, 2 rows dsfsdf
    dim3 dimBlock(THREAD_PER_BLOCK); // 1 column, 32 rows
    int log_n = (int)(log(THREAD_PER_BLOCK) / log(2)); // This will correctly return 2
    parallel_fft<<<dimGrid,dimBlock>>>(a,x,t,log_n);
    cudaDeviceSynchronize();

 



    std::vector<std::complex<double>> cuda_output_vector = cuDoubleComplexToVector(a,input.size());
    compareComplexVectors(y1i,cuda_output_vector);    


    
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