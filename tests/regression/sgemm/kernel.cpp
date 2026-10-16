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
    
    // Shared tile memory for A and B. Each block works on a tile
    auto tile_size = arg->tile_size;

    // Local row/col
    auto l_col = threadIdx.x;
    auto l_row = threadIdx.y;

    // Shared memory tiles, each tile_size^2 entries:
    auto tile_elems_a = tile_size * tile_size;
    auto local_ptr = reinterpret_cast<TYPE*>(__local_mem());
    auto local_A = local_ptr;
    auto local_B = local_A + tile_elems_a;
    
    const int reg_tile_size = 4;
    
    TYPE sum[reg_tile_size][reg_tile_size] = {};
    
    //TYPE sum(0);
    //#pragma unroll 8
    for (uint32_t phase = 0; phase < size / tile_size; ++phase) {
        // Collaborative loading of A and B tiles into shared memory
        for (uint32_t y = 0; y < reg_tile_size; ++y) {
            for (uint32_t x = 0; x < reg_tile_size; ++x) {
                // Indexing for local A/B
                // Account for previous rows + y and previous columns + x
                int l_prev_rows = (l_row * reg_tile_size + y) * tile_size;
                int l_offset = l_col * reg_tile_size + x;
                int local_idx = l_prev_rows + l_offset;
                
                // Indexing global A: the row. 
                // Account for previous blocks and previous rows in this block.
                int a_prev_blocks_rows = blockIdx.y * tile_size;
                int a_this_block_prev_rows = l_row * reg_tile_size + y;
                int a_prev_rows = (a_prev_blocks_rows + a_this_block_prev_rows) * size;

                // Indexing global A: the column.
                // Account for previous phases and previous columns in this block.
                int a_prev_phase_cols = phase * tile_size;
                int a_this_phase_cols = l_col * reg_tile_size + x;
                int a_offset = a_prev_phase_cols + a_this_phase_cols;

                local_A[local_idx] = A[a_prev_rows + a_offset];

                // Indexing global B: the row.
                int b_prev_phase_rows = phase * tile_size;
                int b_this_phase_cols = l_row * reg_tile_size + y;
                int b_prev_rows = (b_prev_phase_rows + b_this_phase_cols) * size;

                // Indexing global B: the column.
                int b_prev_blocks_cols = blockIdx.x * tile_size;
                int b_this_block_prev_cols = l_col * reg_tile_size + x;
                int b_offset = b_prev_blocks_cols + b_this_block_prev_cols;

                local_B[local_idx] = B[b_prev_rows + b_offset];
            }
        }
        
        __syncthreads();

        // Get the summation for the inner product for each elem in reg tile.
        for (uint32_t e = 0; e < tile_size; ++e) { 
            for (uint32_t y = 0; y < reg_tile_size; ++y) {
                for (uint32_t x = 0; x < reg_tile_size; ++x) {
                    // A: go horizontally right. B: go vertically down
                    sum[y][x] += 
                        local_A[(l_row * reg_tile_size + y) * tile_size + e] * 
                        local_B[e * tile_size + reg_tile_size * l_col + x];
                }
            }
        }
        __syncthreads();
    }
    
    // Set the result accordingly.
    for (uint32_t y = 0; y < reg_tile_size; ++y) {
        for (uint32_t x = 0; x < reg_tile_size; ++x) {

            int c_row = blockIdx.y * tile_size + l_row * reg_tile_size + y;
            int c_col = blockIdx.x * tile_size + l_col * reg_tile_size + x;

            C[c_row * size + c_col] = sum[y][x];
        }
    }
    
}
