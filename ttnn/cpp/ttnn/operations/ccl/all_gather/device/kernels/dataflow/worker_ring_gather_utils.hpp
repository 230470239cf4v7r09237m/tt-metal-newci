// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "dataflow_api.h"
#include "debug/assert.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"

using ttnn::ccl::ShardType;
using ttnn::ccl::UNINITIALIZED_VALUE_U16;
using ttnn::ccl::UNINITIALIZED_VALUE_U32;
using ttnn::ccl::WorkerXY;


template <typename AddrGen>
FORCE_INLINE void write_and_send_chunk_write_to_tensor_segment(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t num_cols,
    const uint32_t num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    uint32_t l1_read_addr) {

    // for (uint32_t i = 0; i < num_pages; ++i) {
    int32_t contig_pages = 1;
    for (int32_t pages_remaining = num_pages; pages_remaining != 0; pages_remaining -= contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset] = d.get_page_location(output_page_idx);
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, d.bank_base_address + (page_offset * d.page_size) + 0);
        ASSERT(false);  // untested && unimplemented
    #endif
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        // auto const&[noc_yx, page_offset] = d.get_page_location(output_page_idx);
        auto [noc_yx, page_offset, contig_pages_] = d.get_page_location_with_contiguous_pages_in_row_in_bank(output_page_idx);
        contig_pages = std::min<int32_t>(pages_remaining, std::min<int32_t>(contig_pages_, num_cols - col_idx));
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, d.bank_base_address + (page_offset * d.page_size) + 0);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size * contig_pages);
    #endif
        output_page_idx += contig_pages;
        col_idx += contig_pages;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
#endif
        l1_read_addr += page_size * contig_pages;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}


template <typename AddrGen, ttnn::ccl::EriscDataMoverTerminationMode termination_mode>
FORCE_INLINE void write_and_send_chunk(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t num_cols,
    const uint32_t num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    ccl::edm::WorkerToEdmSender<termination_mode> &sender_adapter) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    sender_adapter.send_payload_non_blocking(cb_id, num_pages, page_size);

    write_and_send_chunk_write_to_tensor_segment(
        output_page_idx,
        col_idx,
        row_idx,
        cb_id,
        d,
        num_cols,
        num_rows,
        col_offset,
        row_offset,
        num_pages,
        page_size,
        l1_read_addr);

}

template <typename AddrGen>
FORCE_INLINE void write_and_send_chunk(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t num_cols,
    const uint32_t num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    uint64_t remote_l1_write_addr,
    uint64_t eth_l1_sender_semaphore_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);

    write_and_send_chunk_write_to_tensor_segment(
        output_page_idx,
        col_idx,
        row_idx,
        cb_id,
        d,
        num_cols,
        num_rows,
        col_offset,
        row_offset,
        num_pages,
        page_size,
        l1_read_addr);
}

void write_tile_partial(uint32_t l1_read_addr, uint64_t dst_noc_addr, uint32_t page_size, uint32_t in_tile_row_start, uint32_t in_tile_row_end) {
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t ELEMENT_SIZE = page_size / (TILE_WIDTH * TILE_HEIGHT);
    // we are assuming here that num_heads_to_write = nh/nkv is a power of 2 here, so that we don't write partial across phase
    uint32_t SUBTILE_LINE_BYTES = 16*ELEMENT_SIZE; //size of 16 elements (in a row)

    for (uint32_t row = in_tile_row_start; row < in_tile_row_end; ++row) {
        uint32_t in_tile_offset_by_starting_row = row < 16 ? row * SUBTILE_LINE_BYTES : (row - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;
        uint64_t out_writer_noc_addr_row = dst_noc_addr + in_tile_offset_by_starting_row;
        uint32_t l1_read_addr_row = l1_read_addr + in_tile_offset_by_starting_row;

        // Write first phase
        noc_async_write(l1_read_addr_row, out_writer_noc_addr_row, SUBTILE_LINE_BYTES);

        // Write second phase
        noc_async_write(l1_read_addr_row+256*ELEMENT_SIZE, out_writer_noc_addr_row+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);
    }
}

template <typename AddrGen>
FORCE_INLINE void write_and_send_chunk_write_to_tensor_segment_partial(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t num_cols,
    const uint32_t num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    uint32_t l1_read_addr,
    const uint32_t in_tile_row_start,
    const uint32_t in_tile_row_end) {

    // for (uint32_t i = 0; i < num_pages; ++i) {
    int32_t contig_pages = 1;
    for (int32_t pages_remaining = num_pages; pages_remaining != 0; pages_remaining -= contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset] = d.get_page_location(output_page_idx);
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, d.bank_base_address + (page_offset * d.page_size) + 0);
        ASSERT(false);  // untested && unimplemented
    #endif
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        // auto const&[noc_yx, page_offset] = d.get_page_location(output_page_idx);
        auto [noc_yx, page_offset, contig_pages_] = d.get_page_location_with_contiguous_pages_in_row_in_bank(output_page_idx);
        contig_pages = std::min<int32_t>(pages_remaining, std::min<int32_t>(contig_pages_, num_cols - col_idx));
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, d.bank_base_address + (page_offset * d.page_size) + 0);
        // noc_async_write(l1_read_addr, dst_noc_addr, page_size * contig_pages);
        for (int32_t i = 0; i < contig_pages; i++) {
            write_tile_partial(l1_read_addr+i*page_size, dst_noc_addr+i*page_size, page_size, in_tile_row_start, in_tile_row_end);
        }
    #endif
        output_page_idx += contig_pages;
        col_idx += contig_pages;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
#endif
        l1_read_addr += page_size * contig_pages;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <typename AddrGen, ttnn::ccl::EriscDataMoverTerminationMode termination_mode>
FORCE_INLINE void write_and_send_chunk_partial(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t num_cols,
    const uint32_t num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    ccl::edm::WorkerToEdmSender<termination_mode> &sender_adapter,
    const uint32_t in_tile_row_start,
    const uint32_t in_tile_row_end) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    sender_adapter.send_payload_non_blocking(cb_id, num_pages, page_size);

    write_and_send_chunk_write_to_tensor_segment_partial(
        output_page_idx,
        col_idx,
        row_idx,
        cb_id,
        d,
        num_cols,
        num_rows,
        col_offset,
        row_offset,
        num_pages,
        page_size,
        l1_read_addr,
        in_tile_row_start,
        in_tile_row_end);

}

template <typename AddrGen>
FORCE_INLINE void write_and_send_chunk_partial(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t num_cols,
    const uint32_t num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    uint64_t remote_l1_write_addr,
    uint64_t eth_l1_sender_semaphore_addr,
    const uint32_t in_tile_row_start,
    const uint32_t in_tile_row_end) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);

    write_and_send_chunk_write_to_tensor_segment_partial(
        output_page_idx,
        col_idx,
        row_idx,
        cb_id,
        d,
        num_cols,
        num_rows,
        col_offset,
        row_offset,
        num_pages,
        page_size,
        l1_read_addr,
        in_tile_row_start,
        in_tile_row_end);
}

template <typename AddrGen>
FORCE_INLINE void write_chunk(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t& num_cols,
    const uint32_t& num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    int32_t contig_pages = 1;

    for (int32_t pages_remaining = num_pages; pages_remaining != 0; pages_remaining -= contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset] = d.get_page_location(output_page_idx);
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, d.bank_base_address + (page_offset * d.page_size) + 0);
        ASSERT(false);  // untested && unimplemented
    #endif
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
    #elif defined SHARDED_MEM_LAYOUT
        auto [noc_yx, page_offset, contig_pages_] = d.get_page_location_with_contiguous_pages_in_row_in_bank(output_page_idx);
        contig_pages = std::min<int32_t>(pages_remaining, std::min<int32_t>(contig_pages_, num_cols - col_idx));
        uint32_t local_address = d.bank_base_address + (page_offset * d.page_size) + 0;
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), static_cast<uint32_t>(noc_yx.noc_y), local_address);
        ASSERT(((dst_noc_addr >> 32) & 0xF) == 0);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size * contig_pages);
    #endif
        output_page_idx += contig_pages;
        col_idx += contig_pages;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
#endif
        l1_read_addr += page_size * contig_pages;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <typename AddrGen>
FORCE_INLINE void write_chunk_partial(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t& num_cols,
    const uint32_t& num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    const uint32_t& in_tile_row_start,
    const uint32_t& in_tile_row_end
    ) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    int32_t contig_pages = 1;

    for (int32_t pages_remaining = num_pages; pages_remaining != 0; pages_remaining -= contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset] = d.get_page_location(output_page_idx);
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, d.bank_base_address + (page_offset * d.page_size) + 0);
        ASSERT(false);  // untested && unimplemented
    #endif
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
    #elif defined SHARDED_MEM_LAYOUT
        auto [noc_yx, page_offset, contig_pages_] = d.get_page_location_with_contiguous_pages_in_row_in_bank(output_page_idx);
        contig_pages = std::min<int32_t>(pages_remaining, std::min<int32_t>(contig_pages_, num_cols - col_idx));
        uint32_t local_address = d.bank_base_address + (page_offset * d.page_size) + 0;
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), static_cast<uint32_t>(noc_yx.noc_y), local_address);
        ASSERT(((dst_noc_addr >> 32) & 0xF) == 0);
        // noc_async_write(l1_read_addr, dst_noc_addr, page_size * contig_pages);
        for (int32_t i = 0; i < contig_pages; i++) {
            write_tile_partial(l1_read_addr+i*page_size, dst_noc_addr+i*page_size, page_size, in_tile_row_start, in_tile_row_end);
        }
    #endif
        output_page_idx += contig_pages;
        col_idx += contig_pages;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
#endif
        l1_read_addr += page_size * contig_pages;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

// read chunk from input tensor (local chip)
template <typename AddrGen>
FORCE_INLINE void read_chunk_from_input_tensor(
    uint32_t& input_page_idx,
    const uint32_t& cb_id,
    const AddrGen& s,
    const uint32_t& num_pages,
    const uint32_t& page_size) {
    const uint32_t end_read_idx = input_page_idx + num_pages;
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    int32_t contig_pages = 1;

    for (int32_t pages_remaining = num_pages; pages_remaining != 0; pages_remaining -= contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
    // #ifdef INTERLEAVED_MEM_LAYOUT || defined SHARDED_MEM_LAYOUT
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset, contig_pages_] = s.get_page_location_with_contiguous_pages_in_row_in_bank(input_page_idx);
        contig_pages = std::min<int32_t>(pages_remaining, contig_pages_);
        uint64_t src_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), static_cast<uint32_t>(noc_yx.noc_y), s.bank_base_address + (page_offset * s.page_size) + 0);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size * contig_pages);
    #endif
#endif
        local_l1_read_addr += (page_size * contig_pages);
        input_page_idx += contig_pages;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

void read_tile_partial(uint64_t src_noc_addr, uint32_t local_l1_read_addr, uint32_t page_size, uint32_t read_in_tile_row_start, uint32_t read_in_tile_row_end, uint32_t write_in_tile_row_start) {
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t ELEMENT_SIZE = page_size / (TILE_WIDTH * TILE_HEIGHT);
    // we are assuming here that num_heads_to_write = nh/nkv is a power of 2 here, so that we don't write partial across phase
    uint32_t SUBTILE_LINE_BYTES = 16*ELEMENT_SIZE; //size of 16 elements (in a row)

    for (uint32_t row = read_in_tile_row_start; row < read_in_tile_row_end; ++row) {
        uint32_t read_in_tile_offset_by_starting_row = row < 16 ? row * SUBTILE_LINE_BYTES : (row - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;
        uint32_t write_in_tile_offset_by_starting_row = write_in_tile_row_start < 16 ? write_in_tile_row_start * SUBTILE_LINE_BYTES : (write_in_tile_row_start - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;

        uint64_t src_noc_addr_row = src_noc_addr + read_in_tile_offset_by_starting_row;
        uint32_t local_l1_read_addr_row = local_l1_read_addr + write_in_tile_offset_by_starting_row;

        // Write first phase
        noc_async_read(src_noc_addr_row, local_l1_read_addr_row, SUBTILE_LINE_BYTES);

        // Write second phase
        noc_async_read(src_noc_addr_row+256*ELEMENT_SIZE, local_l1_read_addr_row+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);

        write_in_tile_row_start++;
    }
}

// read chunk from input tensor (local chip)
template <typename AddrGen>
FORCE_INLINE void read_partial_chunk_from_input_tensor(
    uint32_t& input_page_idx,
    const uint32_t& cb_id,
    const AddrGen& s,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    const uint32_t& read_in_tile_row_start,
    const uint32_t& read_in_tile_row_end,
    const uint32_t& write_in_tile_row_start) {
    const uint32_t end_read_idx = input_page_idx + num_pages;
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    int32_t contig_pages = 1;

    for (int32_t pages_remaining = num_pages; pages_remaining != 0; pages_remaining -= contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
    // #ifdef INTERLEAVED_MEM_LAYOUT || defined SHARDED_MEM_LAYOUT
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset, contig_pages_] = s.get_page_location_with_contiguous_pages_in_row_in_bank(input_page_idx);
        contig_pages = std::min<int32_t>(pages_remaining, contig_pages_);
        uint64_t src_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), static_cast<uint32_t>(noc_yx.noc_y), s.bank_base_address + (page_offset * s.page_size) + 0);
        // noc_async_read(src_noc_addr, local_l1_read_addr, page_size * contig_pages);
        for (int32_t i = 0; i < contig_pages; i++) {
            read_tile_partial(src_noc_addr+i*page_size, local_l1_read_addr+i*page_size, page_size, read_in_tile_row_start, read_in_tile_row_end, write_in_tile_row_start);
        }
    #endif
#endif
        local_l1_read_addr += (page_size * contig_pages);
        input_page_idx += contig_pages;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

// read chunk from output tensor (local chip)
template <typename AddrGen>
FORCE_INLINE void read_chunk_from_output_tensor(
    uint32_t& input_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& s,
    const uint32_t& num_cols,
    const uint32_t& num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    uint32_t contig_pages = 1;
    for (int32_t pages_remaining = num_pages; pages_remaining != 0; pages_remaining -= contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset] = s.get_page_location(input_page_idx);
        uint64_t src_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, s.bank_base_address + (page_offset * s.page_size) + 0);
        ASSERT(false);  // unimplemented
    #endif

        input_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            input_page_idx += row_offset;
        }
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto [noc_yx, page_offset, contig_pages_] = s.get_page_location_with_contiguous_pages_in_row_in_bank(input_page_idx);
        contig_pages = std::min<int32_t>(pages_remaining, std::min<int32_t>(contig_pages_, num_cols - col_idx));
        uint64_t src_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), static_cast<uint32_t>(noc_yx.noc_y), s.bank_base_address + (page_offset * s.page_size) + 0);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size * contig_pages);
    #endif
        input_page_idx += contig_pages;
        col_idx += contig_pages;
        if (col_idx == num_cols) {
            input_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                input_page_idx += row_offset;
            }
        }
#endif
        local_l1_read_addr += page_size * contig_pages;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

template <typename AddrGen>
FORCE_INLINE void read_chunk_from_output_tensor_v2(
    uint32_t& curr_page_idx,
    ttnn::ccl::coord_t& offset_into_worker_slice,
    const ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const ttnn::ccl::coord_t& tensor_shape,
    const uint32_t cb_id,
    const AddrGen& s,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {
    // we expected caller to reset this and the last curr_page_idx when we set it true
    ASSERT(last_page_of_worker == false);
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef ROW_MAJOR_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t src_noc_addr = get_noc_addr(curr_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        ASSERT(false);  // unimplemented
    #elif defined SHARDED_MEM_LAYOUT
        ASSERT(false);  // unimplemented
    #endif

#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_read_tile(curr_page_idx, s, local_l1_read_addr);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset] = s.get_page_location(curr_page_idx);
        uint64_t src_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, s.bank_base_address + (page_offset * s.page_size) + 0);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
    #endif
        // common with `write_chunk_v2`
        offset_into_worker_slice.x++;
        bool end_of_worker_slice_row = offset_into_worker_slice.x == worker_slice_shape.x;
        if (end_of_worker_slice_row) {
            offset_into_worker_slice.x = 0;
            offset_into_worker_slice.y++;
            bool end_of_worker_slice = offset_into_worker_slice.y == worker_slice_shape.y;
            if (end_of_worker_slice) {
                offset_into_worker_slice.y = 0;
                last_page_of_worker = true;
            } else {
                curr_page_idx += tensor_shape.x - worker_slice_shape.x;
            }
        } else {
            curr_page_idx++;
        }
#endif
        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

template <typename AddrGen>
FORCE_INLINE void write_chunk_v2(
    uint32_t& curr_page_idx,
    ttnn::ccl::coord_t& offset_into_worker_slice,
    const ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const ttnn::ccl::coord_t& tensor_shape,
    uint32_t cb_id,
    const AddrGen& d,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef ROW_MAJOR_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t dst_noc_addr = get_noc_addr(curr_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        ASSERT(false);  // unimplemented
    #elif defined SHARDED_MEM_LAYOUT
        ASSERT(false);  // unimplemented
    #endif
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_write_tile(curr_page_idx, d, l1_read_addr);
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset] = d.get_page_location(curr_page_idx);
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, d.bank_base_address + (page_offset * d.page_size) + 0);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
    #endif
        // Common with `read_chunk_from_output_tensor_v2`
        offset_into_worker_slice.x++;
        bool end_of_worker_slice_row = offset_into_worker_slice.x == worker_slice_shape.x;
        if (end_of_worker_slice_row) {
            offset_into_worker_slice.x = 0;
            offset_into_worker_slice.y++;
            bool end_of_worker_slice = offset_into_worker_slice.y == worker_slice_shape.y;
            if (end_of_worker_slice) {
                offset_into_worker_slice.y = 0;
                last_page_of_worker = true;
            } else {
                curr_page_idx += tensor_shape.x - worker_slice_shape.x;
            }
        } else {
            curr_page_idx++;
        }
#endif
        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <typename AddrGen>
FORCE_INLINE void read_wrapped_chunk_from_output_tensor_to_address(
    uint32_t& curr_page_idx,
    uint32_t& offset_into_worker_slice,
    const ttnn::ccl::coord_t& offset_worker_slice,
    const ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const ttnn::ccl::coord_t& tensor_shape,
    const ttnn::ccl::coord_t& tensor_slice_shape,
    const uint32_t local_l1_scratch_buffer_address,
    const AddrGen& s,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {

    // we expected caller to reset this and the last curr_page_idx when we set it true
    uint32_t local_l1_read_addr = local_l1_scratch_buffer_address;

    int32_t contig_pages = 1;
    for (uint32_t i = 0; i < num_pages; i+= contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
  #ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t src_noc_addr = get_noc_addr(curr_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
  #elif defined SHARDED_MEM_LAYOUT
        ASSERT(false);  // unimplemented
  #endif
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_read_tile(curr_page_idx, s, local_l1_read_addr);
        // common with `write_chunk_v2`
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset, contig_pages_] = s.get_page_location_with_contiguous_pages_in_row_in_bank(curr_page_idx);
        /*
         * num_pages - i: check if we are outside the number of pages remaining
         * contig_pages_: check if we are outside the max number of contig pages we can read in a row in a bank
         * contig_edge_of_tensor_slice: check if we are outside the edge of the tensor slice (in which case, we wrap around if we aren't at the end)
         */
        uint32_t flattened_offset_worker_slice = offset_worker_slice.x + (offset_worker_slice.y * tensor_slice_shape.x);
        uint32_t contig_edge_of_tensor_slice = tensor_slice_shape.x - ((flattened_offset_worker_slice + offset_into_worker_slice) % tensor_slice_shape.x);

        contig_pages = std::min<int32_t>(num_pages - i, std::min<int32_t>(contig_pages_, contig_edge_of_tensor_slice));
        uint64_t src_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, s.bank_base_address + (page_offset * s.page_size) + 0);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size * contig_pages);
    #endif

        // Update the curr_page_idx based on how the worker chunks + tensor slice is laid out in global tensor
        advance_worker_global_page_interleaved(
            curr_page_idx, // Updated internally
            offset_into_worker_slice,
            offset_worker_slice,
            worker_slice_shape,
            tensor_slice_shape,
            tensor_shape,
            contig_pages,
            last_page_of_worker
        );

#endif
        local_l1_read_addr += page_size * contig_pages;
    }
    noc_async_read_barrier();
}

template <typename AddrGen>
FORCE_INLINE void read_wrapped_chunk_from_output_tensor(
    uint32_t& curr_page_idx,
    uint32_t& offset_into_worker_slice,
    const ttnn::ccl::coord_t& offset_worker_slice,
    const ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const ttnn::ccl::coord_t& tensor_shape,
    const ttnn::ccl::coord_t& tensor_slice_shape,
    const uint32_t cb_id,
    const AddrGen& s,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {

    // we expected caller to reset this and the last curr_page_idx when we set it true
    ASSERT(last_page_of_worker == false);
    cb_reserve_back(cb_id, num_pages);

    read_wrapped_chunk_from_output_tensor_to_address(
        curr_page_idx,
        offset_into_worker_slice,
        offset_worker_slice,
        worker_slice_shape,
        tensor_shape,
        tensor_slice_shape,
        get_write_ptr(cb_id),
        s,
        num_pages,
        page_size,
        last_page_of_worker);
    cb_push_back(cb_id, num_pages);
}

template <typename AddrGen>
FORCE_INLINE void write_wrapped_chunk(
    uint32_t& curr_page_idx,
    uint32_t& offset_into_worker_slice,
     ttnn::ccl::coord_t& offset_worker_slice,
    const  ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const  ttnn::ccl::coord_t& tensor_shape,
    const  ttnn::ccl::coord_t& tensor_slice_shape,
    uint32_t cb_id,
    const AddrGen& d,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {

    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);

    int32_t contig_pages = 1;
    for (uint32_t i = 0; i < num_pages; i+= contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t dst_noc_addr = get_noc_addr(curr_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        ASSERT(false);  // unimplemented
    #elif defined SHARDED_MEM_LAYOUT
        ASSERT(false);  // unimplemented
    #endif

#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_write_tile(curr_page_idx, d, l1_read_addr);
        // Common with `read_chunk_from_output_tensor_v2`
    #elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const&[noc_yx, page_offset, contig_pages_] = d.get_page_location_with_contiguous_pages_in_row_in_bank(curr_page_idx);
        /*
         * Shared with `read_wrapped_chunk_from_output_tensor`
         */
        uint32_t flattened_offset_worker_slice = offset_worker_slice.x + (offset_worker_slice.y * tensor_slice_shape.x);
        uint32_t contig_edge_of_tensor_slice = tensor_slice_shape.x - ((flattened_offset_worker_slice + offset_into_worker_slice) % tensor_slice_shape.x);

        contig_pages = std::min<int32_t>(num_pages - i, std::min<int32_t>(contig_pages_, contig_edge_of_tensor_slice));
        uint64_t dst_noc_addr = get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, d.bank_base_address + (page_offset * d.page_size) + 0);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size * contig_pages);
    #endif

        // Update the curr_page_idx based on how the worker chunks + tensor slice is laid out in global tensor
        advance_worker_global_page_interleaved(
            curr_page_idx, // Updated internally
            offset_into_worker_slice,
            offset_worker_slice,
            worker_slice_shape,
            tensor_slice_shape,
            tensor_shape,
            contig_pages,
            last_page_of_worker
        );
#endif
        l1_read_addr += page_size * contig_pages;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
