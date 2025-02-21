// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <c_tensix_core.h>
#include "debug/dprint.h"

constexpr bool mcaster = get_compile_time_arg_val(0);
constexpr uint32_t tlx = get_compile_time_arg_val(1);
constexpr uint32_t tly = get_compile_time_arg_val(2);
constexpr uint32_t width = get_compile_time_arg_val(3);
constexpr uint32_t height = get_compile_time_arg_val(4);
constexpr uint64_t duration = (uint64_t)get_compile_time_arg_val(5) * 1000 * 1000 * 1000;
constexpr uint32_t ucast_size = get_compile_time_arg_val(6);
constexpr uint32_t mcast_size = get_compile_time_arg_val(7);
constexpr uint32_t virtual_grid_offset = get_compile_time_arg_val(8);
constexpr uint32_t nrands = get_compile_time_arg_val(9);
constexpr bool enable_rnd_delay = get_compile_time_arg_val(10);
constexpr uint32_t ucast_l1_addr = get_compile_time_arg_val(11);
constexpr uint32_t mcast_l1_addr = get_compile_time_arg_val(12);
constexpr bool atomics = get_compile_time_arg_val(13);
constexpr uint32_t atomic_start = get_compile_time_arg_val(14);

inline uint32_t next_rand(tt_l1_ptr uint8_t* rnds, uint32_t& rnd_index) {
    uint32_t rnd = rnds[rnd_index];
    rnd_index = (rnd_index + 1) & (nrands - 1);
    return rnd;
}

void kernel_main() {
    uint64_t done_time = c_tensix_core::read_wall_clock() + duration;
    tt_l1_ptr uint8_t* rnds = (tt_l1_ptr uint8_t*)(get_arg_addr(0));
    uint32_t rnd_index = 0;
    DPRINT << (uint32_t)atomics << " " << (uint32_t)mcaster << " " << atomic_start << ENDL();
    uint64_t stall_time = 0;
    uint32_t temp = 0;
    while (c_tensix_core::read_wall_clock() < done_time) {
        for (uint32_t count = 0; count < 1000; count++) {
            if (enable_rnd_delay) {
                // reading time here biases us to have more ~0 cycle stalls as this
                // includes the write time
                while (c_tensix_core::read_wall_clock() < stall_time);
                stall_time = c_tensix_core::read_wall_clock() + next_rand(rnds, rnd_index);
            }

            if (mcaster) {
                uint64_t dst_noc_multicast_addr =
                    get_noc_multicast_addr(tlx, tly, tlx + width - 1, tly + height - 1, ucast_l1_addr);
                noc_async_write_multicast(mcast_l1_addr, dst_noc_multicast_addr, mcast_size, width * height, false);
            } else if (atomics) {
                uint64_t noc_write_addr = NOC_XY_ADDR(NOC_X(3), NOC_Y(1), atomic_start + 16 + count / 100 * 32);
                temp++;
                if (temp == 1000001) {
                    // DPRINT << "writing to " << noc_write_addr << ENDL();
                    temp = 0;
                }
                noc_async_write_one_packet(ucast_l1_addr, noc_write_addr, 16);
                noc_write_addr = NOC_XY_ADDR(NOC_X(3), NOC_Y(1), atomic_start + 8 + count / 100 * 32);
                noc_semaphore_inc(noc_write_addr, 1);
                (*(uint32_t*)ucast_l1_addr)++;
            } else {
                uint32_t dst_x, dst_y;
                uint8_t noc_addr = next_rand(rnds, rnd_index);
                dst_x = (noc_addr & 0xf) + virtual_grid_offset;
                dst_y = (noc_addr >> 4) + virtual_grid_offset;
                uint64_t noc_write_addr = NOC_XY_ADDR(NOC_X(dst_x), NOC_Y(dst_y), ucast_l1_addr);
                noc_async_write(ucast_l1_addr, noc_write_addr, ucast_size);
            }
        }
    }
    if (!mcaster && atomics) {
        uint64_t noc_write_addr = NOC_XY_ADDR(NOC_X(3), NOC_Y(1), atomic_start);
        noc_semaphore_inc(noc_write_addr, 1);
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
