#include "../include/Cooley-Tukey-CUDA.hpp"





__global__ void parallel_fft(cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int log_n ){
    int d;
    unsigned int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int flag_block_index = 1;
    int element_thread_computation;
    
    for(int j= 1; j<=log_n; j++){
        d = 1 << j;
        if( d <= THREAD_PER_BLOCK ){
            thread_computation(t_x, d, y, x, t);
        }
        else {
            flag_block_index *= 2;
            if(blockIdx.x % flag_block_index == 0 ){
                for(int i = 0; i < flag_block_index; i++ ){
                    element_thread_computation = t_x + blockDim.x * i ;
                    write_thread_computation(element_thread_computation, d, y, x, t);
                }
                __syncthreads();
                for(int i = 0; i < flag_block_index; i++ ){
                    element_thread_computation = t_x + blockDim.x * i ;
                    sum_thread_computation(element_thread_computation,d,y,x,t);
                }
                __syncthreads();
            }else{
                return;
            }
        }

    }
        
}


int main(){

    srand(95);
    std::vector<std::complex<double>> input;
    for(int i = 0; i < N; i++) {
        // Generate random numbers between -1.0 and 1.0
        double real_part = (rand() % (RAND_MAX )) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        double imag_part = (rand() % (RAND_MAX )) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        input.push_back(std::complex<double>(real_part, imag_part));
    }

    int grid_size = input.size() / THREAD_PER_BLOCK;
    if(input.size() % THREAD_PER_BLOCK != 0) grid_size ++ ;

    std::vector<std::complex<double>> output_iterative = iterative_FF(input);
   


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
    cout << "grid size : " << grid_size << endl;
    cout << "Thread x Block " << THREAD_PER_BLOCK << endl;
    dim3 dimGrid(grid_size);   // 40 columns, 2 rows dsfsdf
    dim3 dimBlock(THREAD_PER_BLOCK); // 1 column, 32 rows
    int log_n = (int)(log(input.size()) / log(2)); // This will correctly return 2
    parallel_fft<<<dimGrid,dimBlock>>>(a,x,t,log_n);
    cudaDeviceSynchronize();

 



    std::vector<std::complex<double>> cuda_output_vector = cuDoubleComplexToVector(a,input.size());
    compareComplexVectors(output_iterative,cuda_output_vector);    


    
    return 0;
}
