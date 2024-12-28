#include "../../include/Cooley-Tukey-CUDA.hpp"

__global__ void parallel_fft(int *atomic_array, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int log_n)
{
    int d, prec_d;
    unsigned int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int flag_block_index = 1;
    int element_thread_computation;
    bool block_end_computation = false;
    unsigned int ns = 10;
    int tmp;
    for (int j = 1; j <= log_n - END_ITERATION; j++)
    {
        if (j >= 2)
            prec_d = d;

        d = 1 << j;
        if (d <= THREAD_PER_BLOCK)
        {
            thread_computation(t_x, d, y, x, t);
        }
        else
        {
            if ((blockIdx.x % d == 0))
            {
                if (j >= 2)
                {
                    while (atomicAdd(&atomic_array[j - 1], 0) != N / prec_d)
                    {
                        for (int i = 0; i < 10000; i++)
                        {
                            tmp++;
                        }
                    }
                }

                for (int i = 0; i < d; i++)
                {

                    element_thread_computation = t_x + blockDim.x * i;
                    write_thread_computation(element_thread_computation, d, y, x, t);

                    if (element_thread_computation == SELECTED_ELEMENT)
                    {
                        if (d == (N / (1 << END_ITERATION)))
                        {
                            // printf("Element %d write in x : [%.3f, %.3f]\n", element_thread_computation, cuCreal(x[element_thread_computation]), cuCimag(x[element_thread_computation]));
                        }
                    }
                }

                __syncthreads();

                for (int i = 0; i < d; i++)
                {
                    element_thread_computation = t_x + blockDim.x * i;
                    if (element_thread_computation == SELECTED_ELEMENT)
                    {
                        if (d == (N / (1 << END_ITERATION)))
                        {
                            // printf("Element %d write in x : [%.3f, %.3f]\n", element_thread_computation, cuCreal(x[element_thread_computation]), cuCimag(x[element_thread_computation]));
                            // printf("Element %d write in t : [%.3f, %.3f]\n", element_thread_computation, cuCreal(t[element_thread_computation]), cuCimag(t[element_thread_computation]));
                        }
                    }
                    sum_thread_computation(element_thread_computation, d, y, x, t);
                    if (element_thread_computation == 68 && d == 16)
                        printf("TMP CHECK  y[68] :  [%.3f, %.3f]  d : %d\n", cuCreal(y[element_thread_computation]), cuCimag(y[element_thread_computation]), d);
                }
                __syncthreads();

                if (threadIdx.x == 0)
                {
                    atomicAdd(&atomic_array[j], 1);
                }
            }
            else
            {
                return;
            }
        }
    }
}

int main()
{

    srand(95);
    std::vector<std::complex<double>> input;
    for (int i = 0; i < N; i++)
    {
        // Generate random numbers between -1.0 and 1.0
        double real_part = (rand() % (RAND_MAX)) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        double imag_part = (rand() % (RAND_MAX)) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        input.push_back(std::complex<double>(real_part, imag_part));
    }

    int grid_size = input.size() / THREAD_PER_BLOCK;
    if (input.size() % THREAD_PER_BLOCK != 0)
        grid_size++;
    cout << "ITERATIVE\n";
    std::vector<std::complex<double>> output_iterative = iterative_FF(input);

    input = permutation(input);
    int log_n = (int)(log(input.size()) / log(2)); // This will correctly return 2

    cuDoubleComplex *a; // input for GPU
    cuDoubleComplex *y; // permutation for GPU
    cuDoubleComplex *x;
    cuDoubleComplex *t;
    int *atomic_array;

    cudaMallocManaged((void **)&a, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
    cudaMallocManaged((void **)&y, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
    cudaMallocManaged((void **)&x, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
    cudaMallocManaged((void **)&t, sizeof(cuDoubleComplex) * (THREAD_PER_BLOCK * grid_size));
    cudaMallocManaged((void **)&atomic_array, sizeof(int) * (log_n + 1)); // here I have put the +1 because I need to use the 0 position as the 1st

    // Copy data from std::complex to cuDoubleComplex
    for (size_t i = 0; i < (THREAD_PER_BLOCK * grid_size); ++i)
    {
        if (i < input.size())
            a[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
        else
            a[i] = make_cuDoubleComplex(0, 0);
    }
    cout << "\nGPU_PARALLEL\n";
    // cout << "grid size : " << grid_size << endl;
    // cout << "Thread x Block " << THREAD_PER_BLOCK << endl;
    dim3 dimGrid(grid_size);         // 40 columns, 2 rows dsfsdf
    dim3 dimBlock(THREAD_PER_BLOCK); // 1 column, 32 rows

    parallel_fft<<<dimGrid, dimBlock>>>(atomic_array, a, x, t, log_n);
    cudaDeviceSynchronize();

    std::vector<std::complex<double>> cuda_output_vector = cuDoubleComplexToVector(a, input.size());
    compareComplexVectors(output_iterative, cuda_output_vector);

    return 0;
}
