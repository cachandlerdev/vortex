#include <vx_spawn2.h>
#include "common.h"

// SGEMM kernel. this handles matrix multiplication between A and B, 
// puts result in C.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<TYPE*>(arg->A_addr);
	auto B = reinterpret_cast<TYPE*>(arg->B_addr);
	auto C = reinterpret_cast<TYPE*>(arg->C_addr);
    auto size = arg->size;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: Maybe add tiling/shared memory?
    // Instructions say tiling, shared memory, loop optimization
    // Reference sgemm2 for shared memory
    // Also reference pg 108, 109 in textbook about 
    // tiled matrix multiplication kernels b/c got no clue what to do

    TYPE sum(0);
    #pragma unroll 8
    for (uint32_t e = 0; e < size; ++e) {
        sum += A[row * size + e] * B[e * size + col];
    }

    C[row * size + col] = sum;
}
