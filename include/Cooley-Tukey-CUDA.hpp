#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuComplex.h>
#define N std::pow(2, 16) // Must be a power of 2
#define THREAD_PER_BLOCK  1024 //this could be changed depends on GPU


//  ------------------- FUNCTION THAT REALLY NEED TO STAY HERE

/**
 * Computes \( w_d^h \), where \( w_d = e^{2\pi i / d} \) is the d-th root of unity, 
 * and \( h \) is a complex number. The computation uses \( w_d^h = exp(h \cdot ln(w_d)) \).
 * 
 * @param d the order of the root of unity.
 * @param h the complex number (cuDoubleComplex).
 * @return the computed value \( w_d^h \) as a cuDoubleComplex.
 */
__host__ __device__ cuDoubleComplex compute_wd_pow_h(int d, cuDoubleComplex h) {
    // Extract real and imaginary parts of h
    double real_h = cuCreal(h);
    double imag_h = cuCimag(h);

    // Compute the natural logarithm of wd, ln(wd) = i * (2π/d)
    double ln_wd_real = 0.0;             // Real part of ln(wd) is 0
    double ln_wd_imag = 2 * M_PI / d;    // Imaginary part of ln(wd)

    // Compute h * ln(wd)
    double exponent_real = -imag_h * ln_wd_imag; // Real part of h * ln(wd)
    double exponent_imag = real_h * ln_wd_imag;  // Imaginary part of h * ln(wd)

    // Compute e^(h * ln(wd)) = e^(exponent_real + i * exponent_imag)
    double magnitude = exp(exponent_real);  // Magnitude = e^(real part of exponent)
    double result_real = magnitude * cos(exponent_imag); // Real part of result
    double result_imag = magnitude * sin(exponent_imag); // Imaginary part of result

    return make_cuDoubleComplex(result_real, result_imag);
}

__device__ void thread_computation(int t_x, int d, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t){
    bool up_down; // up = 1 down = 0 ( if down it will write on x otherwise it will write on t)
    int tmp_index;
    int p;

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

__device__ void write_thread_computation(int t_x, int d, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t){
    bool up_down; // up = 1 down = 0 ( if down it will write on x otherwise it will write on t)
    int tmp_index;
    int p;

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

}

__device__ void sum_thread_computation(int t_x, int d, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t){
    if(t_x % d < d/2 ) {
        y[t_x] =  cuCadd(x[t_x],t[t_x] );
    }else{
        y[t_x] = cuCsub(x[t_x],t[t_x] );
    }
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
using namespace std;

/**
 * @brief Compares two vectors of complex numbers with a specified tolerance.
 * 
 * @param output_iterative Vector of complex numbers representing the first dataset.
 * @param cuda_output_vector Vector of complex numbers representing the second dataset.
 * @param tolerance Tolerance value for comparing real and imaginary parts.
 * @return true if the vectors are the same within the specified tolerance, false otherwise.
 */
bool compareComplexVectors(const std::vector<std::complex<double>>& output_iterative,
                           const std::vector<std::complex<double>>& cuda_output_vector,
                           double tolerance = 1e-3) {
    if (output_iterative.size() != cuda_output_vector.size()) {
        std::cerr << "Vectors have different sizes!\n";
        return false;
    }

    for (size_t i = 0; i < output_iterative.size(); i++) {
        // Get the real and imaginary parts
        double real_diff = std::abs(output_iterative[i].real() - cuda_output_vector[i].real());
        double imag_diff = std::abs(output_iterative[i].imag() - cuda_output_vector[i].imag());

        // Check if the differences in both real and imaginary parts are within the tolerance
        if (real_diff > tolerance || imag_diff > tolerance) {
            std::cout.precision(15);
            std::cout << "output_iterative[" << i << "] = " << output_iterative[i] 
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
