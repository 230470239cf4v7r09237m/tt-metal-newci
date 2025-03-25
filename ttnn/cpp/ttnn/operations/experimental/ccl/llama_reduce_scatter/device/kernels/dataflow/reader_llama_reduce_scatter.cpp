// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(1);
    constexpr uint32_t fabric_sender_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t fabric_receiver_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t client_interface_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t start_tile = get_compile_time_arg_val(5);
    constexpr uint32_t end_tile = get_compile_time_arg_val(6);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t dst_cb_index = get_compile_time_arg_val(8);
    constexpr uint32_t chip_id = get_compile_time_arg_val(9);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in);
    const DataFormat data_format = get_dataformat(cb_id_in);

    uint64_t input_noc_addr = get_noc_addr(get_read_ptr(cb_id_in));
    DPRINT << "src_addr: " << get_noc_addr(src_addr) << " dst_addr: " << dst_addr << ENDL();
    DPRINT << "start_tile: " << start_tile << " end_tile: " << end_tile << ENDL();
    for (uint32_t tile = start_tile; tile < end_tile; ++tile) {
    }
}
