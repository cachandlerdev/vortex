#include <vx_spawn2.h>
#include "common.h"

// A * B
// A = [1 2 3]    B = [3 2 1]
//     [4 5 6]        [6 5 4]
//     [7 8 9]        [9 8 7]

//
// A * B = [1*3 + 2*6 + 3*9  1*2 + 2*5 + 3*8  1*1 + 2*4 + 3*7]
//         [4*3 + 5*6 + 6*9  4*2 + 5*5 + 6*8  4*1 + 5*4 + 6*7]
//         [7*3 + 8*6 + 9*9  7*2 + 8*5 + 9*8  7*1 + 8*4 + 9*7]
// CUDA C uses row major layout: linearized [row 0, row 1, row 2, row 3]
// so it's probably:
// A = [1 2 3 4 5 6 7 8 9]     B = [3 2 1 6 5 4 9 8 7]

// SGEMM kernel. this handles matrix multiplication between A and B, 
// puts result in C.

// Instructions say tiling, shared memory, loop optimization

// HOW DOES TILING WORK (pg 65 & 103-9)
// ==========================================
// Suppose you have 2x2 tiles and 4x4 matrices
// [P00 P01 | ... ...]
// [P10 P11 | ... ...]
//  -----------------
// [... ... | ... ...]
// [... ... | ... ...]
// P00 uses A's row0, B's col0
// P01 uses A's row0, B's col1
// P10 uses A's row1, B's col0
// P11 uses A's row1, B's col1
// Look at pg 103 on tiling
// 
// Basically you have multiple phases for memory allocation.
// [row, col] notation
// 1st phase: A [0-1, 0-1], B [0-1, 0-1] (gets half of C[0-1, 0-1])
// 2nd phase: A [0-1, 2-3], B [2-3, 0-1] (other half of C[0-1, 0-1])
// ==========================================

// HOW TO GET THE RESULT FOR EACH ENTRY (pg. 62)
// ==========================================
// Inner/dot product of two vectors.
// C_{row, col} = sum [ A_{row,k} * B_{k, col} ] for k=0, 1...Width-1
// (Matrix multiplication for 1 element of resulting C)
// ==========================================

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<TYPE*>(arg->A_addr);
	auto B = reinterpret_cast<TYPE*>(arg->B_addr);
	auto C = reinterpret_cast<TYPE*>(arg->C_addr);
    auto size = arg->size; // size of matrix. (always square?)
    
    // Have each thread compute tile_size * tile_size elements
    auto tile_size = arg->tile_size;

    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // (y, x)
    // t_00 t_01 -> [0-3, 0-3], [4-7, 0-3]
    // t_10 t_11 -> [4-7, 0-3], [4-7, 4-7]
    
    float sum[4][4];
    for (uint32_t y = 0; y < tile_size; ++y) {
        uint32_t global_y = row * tile_size + y;

        for (uint32_t x = 0; x < tile_size; ++x) {
            uint32_t global_x = col * tile_size + x;
            
            TYPE partial_sum(0);

            #pragma unroll 8
            for (size_t e = 0; e < size; ++e) {
                partial_sum += A[global_y * size + e] * B[e * size + global_x];
            }
            sum[y][x] = partial_sum;
        }
    }

    for (uint32_t y = 0; y < tile_size; ++y) {
        uint32_t global_y = row * tile_size + y;

        for (uint32_t x = 0; x < tile_size; ++x) {
            uint32_t global_x = col * tile_size + x;
            C[global_y * size + global_x] = sum[y][x];
        }
    }

    /**
        TYPE sum(0);
        #pragma unroll 8
        for (size_t e = 0; e < size; ++e) {
            sum += A[row * size + e] * B[e * size + col];
        }

        C[row * size + col] = sum;
    */
    
    
    /** 

    // Shared tile memory for A and B. Each block works on a tile
    auto tile_size = arg->tile_size;

    // Global row/col coordinates
    // (NOTE: sgemm2 uses col and row flipped)
    // Get the row/col of the C element to work on
    // (block #) * (block size) + current_offset_in_block
    auto g_col = blockIdx.x * blockDim.x + threadIdx.x;
    auto g_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Local row/col
    auto l_col = threadIdx.x;
    auto l_row = threadIdx.y;

    // Shared memory tiles, each tile_size^2 entries:
    auto tile_elems_a = tile_size * tile_size;
    auto local_ptr = reinterpret_cast<TYPE*>(__local_mem());
    auto local_A = local_ptr;
    auto local_B = local_A + tile_elems_a;
    
    TYPE sum(0);
    #pragma unroll 8
    for (uint32_t phase = 0; phase < size / tile_size; ++phase) {
        // Collaborative loading of A and B tiles into shared memory
        
        // A idx: (account for prev rows) + (starting col for phase) + (thread offset in block)
        int aPrevRowsOffset = g_row * size;
        int aStartingCol = phase * tile_size;
        int aIdx = aPrevRowsOffset + aStartingCol + l_col;

        // B idx: (correct row idx) * size of rows + col
        // correct row idx: (starting row for phase) + (thread offset in block)
        int bStartingRow = phase * tile_size;
        int bIdx = (bStartingRow + l_row) * size + g_col;

        local_A[l_row * tile_size + l_col] = A[aIdx];
        local_B[l_row * tile_size + l_col] = B[bIdx];
        __syncthreads();

        // Get the summation for the inner product
        for (uint32_t e = 0; e < tile_size; ++e) {
            // A: go horizontally right. B: go vertically down
            sum += local_A[l_row * tile_size + e] * local_B[e * tile_size + l_col];
        }
        __syncthreads();
    }

    // Set the result accordingly.
    C[g_row * size + g_col] = sum;
    
    */

}
