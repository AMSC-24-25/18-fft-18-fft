#include "./static_header.hpp"

__device__ void thread_write(int t_x, int d, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t)
{
    double result_real, result_imag;

    bool up_down; // up = 1 down = 0 ( if down it will write on x otherwise it will write on t)
    int tmp_index;
    int p;
    cuDoubleComplex w;

    if ((t_x % d) < d / 2)
    {
        up_down = 0;
    }
    else
        up_down = 1;

    // DOWN CASE
    if (!up_down)
    {
        tmp_index = t_x + d / 2;
        // printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_x, j ,up_down, tmp_index);
        x[tmp_index] = y[t_x];
        x[t_x] = y[t_x];
    }
    // UP CASE
    else
    {
        tmp_index = t_x - d / 2;
        // printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_x, j ,up_down, tmp_index);
        p = (int)(tmp_index % d);
        compute_wd_pow_h(&result_real, &result_imag, d, make_cuDoubleComplex(static_cast<double>(p), 0.0));
        w = make_cuDoubleComplex(result_real, result_imag);
        t[tmp_index] = cuCmul(y[t_x], w);
        t[t_x] = cuCmul(y[t_x], w);
    }

    __syncthreads();
}

__device__ void thread_sum(int t_x, int d, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t)
{
    if (t_x % d < d / 2)
    {
        y[t_x] = cuCadd(x[t_x], t[t_x]);
    }
    else
    {
        y[t_x] = cuCsub(x[t_x], t[t_x]);
    }
    __syncthreads();
}

__global__ void parallel_fft(int input_size, int *atomic_array, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int log_n)
{
    double result_real, result_imag;
    int d, prec_d;
    unsigned int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    bool up_down; // up = 1 down = 0 ( if down it will write on x otherwise it will write on t)
    int tmp_index;
    int p;
    int element_thread_computation;
    int num_iteration;
    int tmp;
    cuDoubleComplex w;
    bool flag = false;
    for (int j = 1; j <= log_n - END_ITERATION; j++)
    {
        prec_d = d;
        d = 1 << j;
        num_iteration = d / THREAD_PER_BLOCK;
        if (d <= THREAD_PER_BLOCK)
        {
            thread_write(t_x, d, y, x, t);
            thread_sum(t_x, d, y, x, t);
            __syncthreads();
            if (threadIdx.x == 0)
            {
                printf("ATOMICADD =>    d: %d, blockIdx.x: %d\n", d, blockIdx.x);
                atomicAdd(&atomic_array[j], 1);
            }
        }
        else
        {
            if ((blockIdx.x % num_iteration == 0))

            {

                while (atomicAdd(&atomic_array[j - 1], 0) != input_size / prec_d)
                {
                    for (int i = 0; i < 100; i++)
                    {
                        tmp++;
                    }

                    if (threadIdx.x == 0 && flag == false)
                    {
                        printf("d: %d , BlockIdx.x : %d is in wait =>        N : %d,   prec_d : %d      N/prec_d : %d\n", d, blockIdx.x, input_size, prec_d, input_size / prec_d);
                    }

                    flag = true;
                }

                flag = false;

                for (int i = 0; i < num_iteration; i++)
                {
                    element_thread_computation = t_x + blockDim.x * i;
                    thread_write(element_thread_computation, d, y, x, t);
                }

                for (int i = 0; i < num_iteration; i++)
                {
                    element_thread_computation = t_x + blockDim.x * i;
                    thread_sum(element_thread_computation, d, y, x, t);
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(&atomic_array[j], 1);
                    printf("ATOMICADD =>    d: %d, blockIdx.x: %d\n", d, blockIdx.x);
                }
            }
            else
            {
                return;
            }
        }
    }
}
