// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    unary_op_init_common(tt::CB::c_in0, tt::CB::c_out0);

    for(uint32_t b = 0; b < per_core_block_cnt; ++ b) {
        cb_wait_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::c_out0, per_core_block_tile_cnt);

        tile_regs_acquire();
        copy_tile_init();
        for (uint32_t i = 0; i < per_core_block_tile_cnt; ++i) {
            copy_tile(tt::CB::c_in0, i, i);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_untilize_dst_init_short<per_core_block_tile_cnt>(tt::CB::c_out0);
        pack_untilize_dst<per_core_block_tile_cnt>(tt::CB::c_out0);
        tile_regs_release();

        cb_push_back(tt::CB::c_out0, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::c_in0, per_core_block_tile_cnt);
    }

    pack_untilize_uninit();
}
}
