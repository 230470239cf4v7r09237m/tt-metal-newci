// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dram_prefetcher_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"

namespace ttnn::operations::dram_prefetcher {

using std::vector;
using namespace tt::constants;
using namespace tt::tt_metal;

void get_max_page_size_and_num_pages(
    uint32_t num_tiles, uint32_t num_datums_per_tile, uint32_t& page_size, uint32_t& num_pages) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * num_datums_per_tile;

    page_size = (8192 / num_datums_per_tile) * num_datums_per_tile;
    while (total_size % page_size != 0 && page_size >= num_datums_per_tile) {
        page_size -= num_datums_per_tile;
    }
    num_pages = total_size / page_size;
}

operation::ProgramWithCallbacks dram_prefetcher_multi_core(
    const std::vector<Tensor>& tensors,
    const Tensor& tensor_addrs,
    const uint32_t num_layers,
    const std::optional<const tt::tt_metal::v1::experimental::GlobalCircularBuffer>& global_cb,
    std::vector<Tensor>& output_tensor) {
    TT_FATAL(global_cb != std::nullopt, "Global circular buffer must be provided");

    /* Buffers */
    const Buffer& global_cb_buffer = global_cb->cb_buffer();
    Buffer* tensor_addrs_buffer = tensor_addrs.buffer();
    std::vector<Buffer*> tensor_buffers;
    for (const auto& tensor : tensors) {
        tensor_buffers.push_back(tensor.buffer());
    }
    Buffer* reader_output_buffer = output_tensor[0].buffer();
    // Buffer* writer_output_buffer = output_tensor[1].buffer();

    /* Tiles */
    tt::tt_metal::Tile tensor_addrs_tile = tensor_addrs.get_tensor_spec().tile();
    std::vector<tt::tt_metal::Tile> tensor_tiles;
    for (const auto& tensor : tensors) {
        tensor_tiles.push_back(tensor.get_tensor_spec().tile());
    }

    /* Dataforamts */
    tt::DataFormat reader_cb_data_format = tt::DataFormat::Float16_b;  // TODO: update?
    tt::DataFormat tensor_addrs_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_addrs.get_dtype());
    std::vector<tt::DataFormat> tensor_data_formats;
    for (const auto& tensor : tensors) {
        tensor_data_formats.push_back(tt::tt_metal::datatype_to_dataformat_converter(tensor.get_dtype()));
    }

    Program program{};

    // In validate we make sure that all tensors are on the same device
    tt::tt_metal::Device* device = tensors[0].device();
    uint32_t num_tensors = tensors.size();
    uint32_t num_receivers_per_reader = global_cb->receiver_cores().num_cores() / global_cb->sender_cores().num_cores();

    // TODO: What does this granularity depend on?
    uint32_t num_blocks = global_cb->receiver_cores().num_cores();
    std::vector<uint32_t> tensor_block_num_tiles;
    std::vector<std::vector<uint32_t>> tensor_shapes;
    std::vector<uint32_t> tensor_tile_sizes;
    for (uint32_t t = 0; t < num_tensors; t++) {
        uint32_t height_in_tiles = tensor_buffers[t]->shard_spec().shape()[0] / tensor_tiles[t].get_tile_shape()[0];
        uint32_t width_in_tiles = tensor_buffers[t]->shard_spec().shape()[1] / tensor_tiles[t].get_tile_shape()[1];

        tensor_shapes.push_back({height_in_tiles, width_in_tiles});
        tensor_block_num_tiles.push_back(height_in_tiles * width_in_tiles / num_blocks);
        tensor_tile_sizes.push_back(tensor_tiles[t].get_tile_size(tensor_data_formats[t]));
    }

    /* Cores setup */
    auto reader_core_range = global_cb->sender_cores();  // CoreRangeSet({CoreRange(CoreCoord(0, 0))});
    auto receiver_core_range = global_cb->receiver_cores();

    /* read cb setup */
    uint32_t reader_cb_single_tile_size = 2048;  // bfloat16 tile size
    uint32_t reader_cb_size = (global_cb->size() / reader_cb_single_tile_size) * reader_cb_single_tile_size;
    // uint32_t reader_cb_size = 2048*2;

    // uint32_t reader_cb_single_tile_size = 1088;  // bfp8_b tile size
    // uint32_t reader_cb_single_tile_size = 576;  // bfp4_b tile size

    uint32_t reader_cb_index = tt::CB::c_in0;
    CircularBufferConfig reader_cb_config =
        CircularBufferConfig(reader_cb_size, {{reader_cb_index, reader_cb_data_format}})
            .set_page_size(reader_cb_index, reader_cb_single_tile_size)
            .set_globally_allocated_address(global_cb_buffer);
    auto reader_cb = CreateCircularBuffer(program, reader_core_range, reader_cb_config);

    /* tensor addresses cb setup */
    uint32_t tensor_addrs_single_tile_size =
        sizeof(uint32_t);  // tensor_addrs_tile.get_tile_size(tensor_addrs_data_format);
    uint32_t tensor_addrs_cb_num_tiles = tensor_addrs_buffer->shard_spec().shape()[0] *
                                         tensor_addrs_buffer->shard_spec().shape()[1];  // TODO: check this
    uint32_t tensor_addrs_cb_size =
        num_layers * num_tensors *
        tensor_addrs_single_tile_size;  // tensor_addrs_cb_num_tiles * tensor_addrs_single_tile_size;

    uint32_t tensor_addrs_cb_index = tt::CB::c_in1;
    CircularBufferConfig tensor_addrs_cb_config =
        CircularBufferConfig(tensor_addrs_cb_size, {{tensor_addrs_cb_index, tensor_addrs_data_format}})
            .set_page_size(tensor_addrs_cb_index, tensor_addrs_single_tile_size)
            .set_globally_allocated_address(*tensor_addrs_buffer);
    auto tensor_addrs_cb = CreateCircularBuffer(program, reader_core_range, tensor_addrs_cb_config);

    /* remote cb setup */
    uint32_t remote_cb_size = global_cb->size();
    uint32_t remote_cb_single_tile_size = reader_cb_single_tile_size;  // 16B aligned

    uint32_t remote_cb_index = tt::CBIndex::c_31;
    CircularBufferConfig remote_cb_config = CircularBufferConfig(remote_cb_size);
    remote_cb_config.remote_index(remote_cb_index)
        .set_page_size(remote_cb_single_tile_size)
        .set_data_format(reader_cb_data_format);
    auto remote_cb =
        tt::tt_metal::v1::experimental::CreateCircularBuffer(program, reader_core_range, remote_cb_config, *global_cb);

    /* Compile time args */

    // Reader kernel
    std::vector<uint32_t> reader_ct_args = {num_layers, num_tensors, num_blocks, reader_cb_size};

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/reader_dram_v2.cpp",
        reader_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            // .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,  // TODO: Is this needed?
            .compile_args = reader_ct_args});

    // Writer kernel
    std::vector<uint32_t> writer_ct_args = {num_layers, num_tensors, num_blocks, num_receivers_per_reader};

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp",
        reader_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            // .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,  // TODO: Is this needed?
            .compile_args = writer_ct_args});

    /* Runtime args */
    std::vector<uint32_t> page_sizes;
    std::vector<uint32_t> block_num_pages;

    std::vector<uint32_t> coalesced_page_sizes;
    std::vector<uint32_t> coalesced_num_pages;

    for (uint32_t t = 0; t < num_tensors; t++) {
        uint32_t page_size, num_pages;
        get_max_page_size_and_num_pages(
            tensor_block_num_tiles[t], tt::tt_metal::detail::TileSize(tensor_data_formats[t]), page_size, num_pages);
        page_sizes.push_back(page_size);
        block_num_pages.push_back(num_pages);

        uint32_t coalesced_page_size, coalesced_num_page;
        uint32_t block_width_in_tiles = tensor_shapes[t][1];
        get_max_page_size_and_num_pages(
            block_width_in_tiles / num_receivers_per_reader,
            tt::tt_metal::detail::TileSize(tensor_data_formats[t]),
            coalesced_page_size,
            coalesced_num_page);
        coalesced_page_sizes.push_back(coalesced_page_size);
        coalesced_num_pages.push_back(coalesced_num_page);
    }

    uint32_t total_num_blocks_in_buffer = 3;  // TODO: how big should reader CB be? here it's triple buffered
    uint32_t bank_start_id = 1;               // TODO: What is this for?
    std::vector<uint32_t> bank_ids;
    const auto& reader_cores = corerange_to_cores(reader_core_range, std::nullopt, true);  // TODO: fix order??

    // Runtime args for the reader cores
    for (uint32_t core_index = 0; core_index < reader_core_range.num_cores(); core_index++) {
        const auto& core = reader_cores[core_index];
        uint32_t noc = 1;  // TODO: Update this

        /* reader kernel */
        uint32_t bank_id = core_index;
        uint32_t vc = bank_id & 0x1;
        bank_ids.push_back(bank_id);

        // Compare with previous cores' vc
        for (size_t j = 0; j < core_index; ++j) {
            const CoreCoord& prev_core = reader_cores[j];
            if (prev_core.y == core.y and ((bank_id & 0x1) == (bank_ids[j] & 0x1))) {  // same vc and same row
                vc = (vc + 1) & 0x1;
                break;
            }
        }

        const uint32_t total_num_blocks_in_buffer = 3;  // TODO: parametrize this

        std::vector<uint32_t> reader_rt_args = {bank_id, vc, total_num_blocks_in_buffer};
        reader_rt_args.insert(reader_rt_args.end(), page_sizes.begin(), page_sizes.end());
        reader_rt_args.insert(reader_rt_args.end(), block_num_pages.begin(), block_num_pages.end());
        reader_rt_args.insert(reader_rt_args.end(), tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        // /* writer kernel */
        std::vector<uint32_t> writer_rt_args;
        writer_rt_args.insert(writer_rt_args.end(), coalesced_page_sizes.begin(), coalesced_page_sizes.end());
        writer_rt_args.insert(writer_rt_args.end(), coalesced_num_pages.begin(), coalesced_num_pages.end());
        writer_rt_args.insert(writer_rt_args.end(), tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());
        writer_rt_args.insert(writer_rt_args.end(), tensor_tile_sizes.begin(), tensor_tile_sizes.end());
        for (auto tensor_shape : tensor_shapes) {  // block_height_in_itles
            writer_rt_args.push_back(tensor_shape[0] / num_blocks);
        }
        writer_rt_args.push_back(noc);

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, reader_core_range](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensor) {
        // TODO: update the CB addrs for the output tensor

        // for (const auto& range : reader_core_range.ranges()) {
        //     for (const auto& core_coord : range) {
        //         // TODO: set runtime args for reader and writer
        //         auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core_coord);
        //     }
        // }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::dram_prefetcher
