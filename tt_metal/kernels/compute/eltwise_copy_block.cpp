#include <cstdint>

#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);

    for(int block = 0; block < num_blocks; ++block) {
       acquire_dst(tt::DstMode::Half);

       // Wait tiles on the input / copy to dst / pop from input
       cb_wait_front(tt::CB::c_in0, block_num_tiles);
       for(int t = 0; t < block_num_tiles; ++t) {
           copy_tile(tt::CB::c_in0, t, t);
       }
       cb_pop_front(tt::CB::c_in0, block_num_tiles);

       // Reserve space in output / pack / push to output
       cb_reserve_back(tt::CB::c_out0, block_num_tiles);
       for(int t = 0; t < block_num_tiles; ++t) {
            pack_tile(t, tt::CB::c_out0);
       }
       cb_push_back(tt::CB::c_out0, block_num_tiles);

       release_dst(tt::DstMode::Half);
    }

}
}
