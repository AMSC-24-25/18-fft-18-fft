#include <cuda_runtime.h>

__global__ void my_kernel()
{
    // Sleep for 100 nanoseconds (this is an example duration)
    __nanosleep(1);
}

int main()
{
    // Launch kernel
    my_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
