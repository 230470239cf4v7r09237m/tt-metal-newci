// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"  // required in all kernels using DPRINT

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"

// topk llk needs a global variable atm
// this can only be removed once that's fixed
int32_t topk_replay_init = 0;

namespace NAMESPACE {

#if 0
void print_all_tiles(uint32_t input_transposed_cb_index, uint32_t start) {
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = uint8_t(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        UNPACK(
            DPRINT << "for iter i/p " << start << " " << (int)r << " : " << TSLICE(input_transposed_cb_index, start, sr)
                   << ENDL());
    }
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = uint8_t(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        UNPACK(
            DPRINT << "for iter i/p " << start + 1 << " " << (int)r << " : "
                   << TSLICE(input_transposed_cb_index, start + 1, sr) << ENDL());
    }
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = uint8_t(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        UNPACK(
            DPRINT << "for iter i/p " << start + 2 << " " << (int)r << " : "
                   << TSLICE(input_transposed_cb_index, start + 2, sr) << ENDL());
    }
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = uint8_t(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        UNPACK(
            DPRINT << "for iter i/p " << start + 3 << " " << (int)r << " : "
                   << TSLICE(input_transposed_cb_index, start + 3, sr) << ENDL());
    }
}
#endif

void MAIN {
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t K = get_compile_time_arg_val(8);
    constexpr uint32_t logk = get_compile_time_arg_val(9);
    constexpr uint32_t logWt = get_compile_time_arg_val(10);
    constexpr uint32_t largest = get_compile_time_arg_val(11);
    constexpr uint32_t sorted = get_compile_time_arg_val(12);

    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    constexpr uint32_t tiles_per_seq = std::max(K / 32, (uint32_t)1);

    int end_phase = (K <= 64) ? logk - 1 : 5;
    // init pack, compute and unpack

    ckernel::topk_tile_init();
    transpose_wh_init(input_cb_index, input_transposed_cb_index);

    bool ascending = !largest;
    bool switch_dir = (K == 64);
    int target_tiles = (Wt * Ht == 1 || ((Wt == 1) && (tiles_per_seq == 1))) ? 1 : 2;
    int seq_per_2tiles = std::max((2 * 32) / K, (uint32_t)2);

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        cb_reserve_back(input_transposed_cb_index, Wt);
        cb_reserve_back(index_transposed_cb_index, Wt);

        // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
        for (uint32_t wt = 0; wt < Wt; wt += 2) {
            acquire_dst();
            // local sort into k groups
            cb_wait_front(input_cb_index, 2);
            cb_wait_front(index_cb_index, 2);

            reconfig_data_format_srca(input_cb_index);
            transpose_wh_init_short(input_cb_index);
            transpose_wh_tile(input_cb_index, 0, 0);
            transpose_wh_tile(input_cb_index, 1, 1);

            reconfig_data_format_srca(index_cb_index);
            transpose_wh_init_short(index_cb_index);
            transpose_wh_tile(index_cb_index, 0, 2);
            transpose_wh_tile(index_cb_index, 1, 3);

            // llk_topk_sort -> inplace
            ckernel::topk_local_sort(0, (int)ascending, end_phase);

            // pack value tiles into cb_intermed0
            pack_reconfig_data_format(input_transposed_cb_index);
            pack_tile(0, input_transposed_cb_index);
            pack_tile(1, input_transposed_cb_index);

            // pack index tiles into cb_intermed1
            pack_reconfig_data_format(index_transposed_cb_index);
            pack_tile(2, index_transposed_cb_index);
            pack_tile(3, index_transposed_cb_index);

            cb_pop_front(input_cb_index, 2);
            cb_pop_front(index_cb_index, 2);
            release_dst();
            ascending = switch_dir ? !ascending : ascending;
        }

        cb_push_back(input_transposed_cb_index, Wt);
        cb_push_back(index_transposed_cb_index, Wt);

        uint32_t num_k_sequences = (Wt * 32) / K;

        // iterative divide and conquer on pairs of tiles (bitonic topk merge and rebuild)
        // first iteration we compare 0th and 1st tile, then 2nd and 3rd, etc. We get the sorted top 32 values in each
        // pair. second iteration we compare 0th and 2nd tile, then 4th and 6th, etc. logWt iteration we compare 0th and
        // Wt/2 tile single buffer as we can pack tiles back in-place
        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            cb_wait_front(input_transposed_cb_index, Wt);
            cb_wait_front(index_transposed_cb_index, Wt);

            uint32_t i = 0;
            uint32_t dist = (1 << m_iter) * tiles_per_seq;
            uint32_t left_tile_id = 0;
            while (i < num_k_sequences) {
                for (uint32_t t = 0; t < tiles_per_seq; t++) {
                    left_tile_id = i * (1 << m_iter) * tiles_per_seq + t;
                    uint32_t right_tile_id = left_tile_id + dist;
                    if (left_tile_id == right_tile_id) {
                        right_tile_id = left_tile_id + 1;
                    }

                    if (left_tile_id >= Wt || right_tile_id >= Wt) {
                        break;
                    }

                    acquire_dst();

                    copy_tile_to_dst_init_short_with_dt(index_transposed_cb_index, input_transposed_cb_index);
                    copy_tile(input_transposed_cb_index, left_tile_id, input_dest_start);
                    copy_tile(input_transposed_cb_index, right_tile_id, input_dest_end);

                    // unpack indices into dest
                    copy_tile_to_dst_init_short_with_dt(input_transposed_cb_index, index_transposed_cb_index);
                    copy_tile(index_transposed_cb_index, left_tile_id, index_dest_start);
                    copy_tile(index_transposed_cb_index, right_tile_id, index_dest_end);

                    // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
                    ckernel::topk_merge(0, m_iter, K);

                    // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values
                    // for topk, which was in input_dest_start
                    pack_reconfig_data_format(input_transposed_cb_index);
                    pack_tile<true>(input_dest_start, input_transposed_cb_index, left_tile_id);
                    pack_tile<true>(input_dest_end, input_transposed_cb_index, right_tile_id);

                    // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values
                    // for topk, which was in index_dest_start
                    pack_reconfig_data_format(index_transposed_cb_index);
                    pack_tile<true>(index_dest_start, index_transposed_cb_index, left_tile_id);
                    pack_tile<true>(index_dest_end, index_transposed_cb_index, right_tile_id);
                    release_dst();
                }
                i += seq_per_2tiles;
            }

            cb_reserve_back(input_transposed_cb_index, Wt);
            cb_reserve_back(index_transposed_cb_index, Wt);

            cb_pop_front(input_transposed_cb_index, Wt);
            cb_pop_front(index_transposed_cb_index, Wt);

            cb_push_back(input_transposed_cb_index, Wt);
            cb_push_back(index_transposed_cb_index, Wt);
            // print_all_tiles(input_transposed_cb_index, 0);

            // we have decreased our search space by half
            num_k_sequences = num_k_sequences >> 1;
            int target_tiles = (Wt == 1 || ((num_k_sequences == 1) && (tiles_per_seq == 1))) ? 1 : 2;
            uint32_t idx = 0;
            int sel_tile_id[2];
            int sel_tile_id_ptr = 0;
            seq_per_2tiles = (seq_per_2tiles == 2) ? 2 : seq_per_2tiles >> 1;
            bool a = !largest;

            cb_wait_front(input_transposed_cb_index, Wt);
            cb_wait_front(index_transposed_cb_index, Wt);

            while (idx < num_k_sequences) {
                for (uint32_t t = 0; t < tiles_per_seq; t++) {
                    uint32_t left_ind = idx * (1 << (m_iter + 1)) * tiles_per_seq + t;
                    if (left_ind >= Wt) {
                        break;
                    }
                    sel_tile_id[sel_tile_id_ptr] = left_ind;
                    sel_tile_id_ptr++;
                    if (sel_tile_id_ptr == target_tiles) {
                        acquire_dst();

                        copy_tile_to_dst_init_short_with_dt(index_transposed_cb_index, input_transposed_cb_index);
                        copy_tile(input_transposed_cb_index, sel_tile_id[0], input_dest_start);
                        if (target_tiles != 1) {
                            copy_tile(input_transposed_cb_index, sel_tile_id[1], input_dest_end);
                        }

                        // unpack indices into dest
                        copy_tile_to_dst_init_short_with_dt(input_transposed_cb_index, index_transposed_cb_index);
                        copy_tile(index_transposed_cb_index, sel_tile_id[0], index_dest_start);
                        if (target_tiles != 1) {
                            copy_tile(index_transposed_cb_index, sel_tile_id[1], index_dest_end);
                        }

                        // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
                        // sort within the larger 32 values
                        ckernel::topk_rebuild(0, (uint32_t)a, m_iter, K, logk, target_tiles == 1);

                        // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values
                        // for topk, which was in input_dest_start
                        pack_reconfig_data_format(input_transposed_cb_index);
                        pack_tile<true>(input_dest_start, input_transposed_cb_index, sel_tile_id[0]);
                        if (target_tiles != 1) {
                            pack_tile<true>(input_dest_end, input_transposed_cb_index, sel_tile_id[1]);
                        }

                        // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values
                        // for topk, which was in index_dest_start
                        pack_reconfig_data_format(index_transposed_cb_index);
                        pack_tile<true>(index_dest_start, index_transposed_cb_index, sel_tile_id[0]);
                        if (target_tiles != 1) {
                            pack_tile<true>(index_dest_end, index_transposed_cb_index, sel_tile_id[1]);
                        }
                        release_dst();
                        sel_tile_id_ptr = 0;
                        a = switch_dir ? !a : a;
                    }
                }
                idx += (seq_per_2tiles >> 1);
            }

            cb_reserve_back(input_transposed_cb_index, Wt);
            cb_reserve_back(index_transposed_cb_index, Wt);

            cb_pop_front(input_transposed_cb_index, Wt);
            cb_pop_front(index_transposed_cb_index, Wt);

            cb_push_back(input_transposed_cb_index, Wt);
            cb_push_back(index_transposed_cb_index, Wt);
        }

        constexpr uint32_t Kt = K % TILE_WIDTH == 0 ? K / TILE_WIDTH : K / TILE_WIDTH + 1;

        // transpose value tiles and pack into output buffer
        reconfig_data_format_srca(input_transposed_cb_index);
        transpose_wh_init_short(input_transposed_cb_index);
        pack_reconfig_data_format(input_transposed_cb_index);
        cb_wait_front(input_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst();
            cb_reserve_back(values_cb_index, 1);
            transpose_wh_tile(input_transposed_cb_index, i, 0);
            pack_tile(0, values_cb_index);
            cb_push_back(values_cb_index, 1);
            release_dst();
        }
        cb_wait_front(input_transposed_cb_index, Wt);
        cb_pop_front(input_transposed_cb_index, Wt);

        // transpose index tiles and pack into output buffer
        reconfig_data_format_srca(index_transposed_cb_index);
        transpose_wh_init_short(index_transposed_cb_index);
        pack_reconfig_data_format(index_transposed_cb_index);
        cb_wait_front(index_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst();
            cb_reserve_back(output_ind_cb_index, 1);
            transpose_wh_tile(index_transposed_cb_index, i, 0);
            pack_tile(0, output_ind_cb_index);
            cb_push_back(output_ind_cb_index, 1);
            release_dst();
        }
        cb_wait_front(index_transposed_cb_index, Wt);
        cb_pop_front(index_transposed_cb_index, Wt);
    }
}
}  // namespace NAMESPACE
