#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t size;
  uint32_t tile_size;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

// TODO: Maybe add BLOCK_SIZE and GRID_SIZE defines?
// No clue what we're supposed to do here.

uint32_t tile_size = 32;

#endif
