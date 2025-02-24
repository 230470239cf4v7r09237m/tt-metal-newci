// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_device_operation.hpp"
#include "conv3d_program_factory.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::conv::conv3d::detail {

operation::ProgramWithCallbacks conv3d_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const Conv3dConfig& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    Program program = CreateProgram();
    auto core_grid = CoreRange({0, 0}, {0, 0});
    /*
    First implementation just performs vol2col on a single core.
    */

    auto input_tensor_shape = input_tensor.get_logical_shape();
    uint32_t N = input_tensor_shape[0];
    uint32_t T_in = input_tensor_shape[1];
    uint32_t H_in = input_tensor_shape[2];
    uint32_t W_in = input_tensor_shape[3];
    uint32_t C_in = input_tensor_shape[4];
    auto [T_out, H_out, W_out] = detail::compute_output_dims(T_in, H_in, W_in, config.padding, config.kernel_size);
    uint32_t C_out = config.output_channels;

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    auto dtype_bytes = input_tensor.element_size();
    auto tile_size = tt::tt_metal::detail::TileSize(data_format);

    bool use_bias = bias_tensor.has_value();

    /* Shapes/sizes needed in the kernel
        Reader does volume2column to convert some `T_block x H_block x W_block` of activation
        to `T_block x H_block x W_block, kD x kH x kW x C_in` patches.
        Compute takes this `num_patches x patch_size` CB and tilizes it.

        Writer reads the weights of size `kD x kH x kW x C_in, C_out`, tilized.
        Writer reads the bias of size `1, C_out`, tilized.
        Compute runs matmul on `patches @ kernel` and adds bias.
        Compute untilizes the result.
        Writer writes the result to the output tensor.


    Padding/tilizing constraints:
        - ceil(num_patches / TILE_HEIGHT) is number of tile rows of matmul
        - `kD x kH x kW x C_in` of the kernel weight is padded to tile size (since it's tilized)
            and must be padded with zeros so the MM result is correct.
    */

    uint32_t patch_size = config.kernel_size[0] * config.kernel_size[1] * config.kernel_size[2] * C_in;
    uint32_t num_patches = config.T_out_block * config.H_out_block * config.W_out_block;

    // If C_out_block is set, use it. Otherwise, use the full number of output channels.
    uint32_t C_out_block = config.C_out_block > 0 ? config.C_out_block : C_out;
    uint32_t C_out_num_blocks = tt::div_up(C_out, C_out_block);
    TT_FATAL(C_out_num_blocks * C_out_block == C_out, "C_out_num_blocks * C_out_block must equal C_out");

    uint32_t matmul_M_t = tt::div_up(num_patches, tt::constants::TILE_HEIGHT);
    uint32_t matmul_K_t = tt::div_up(patch_size, tt::constants::TILE_WIDTH);
    uint32_t matmul_N_t = tt::div_up(C_out_block, tt::constants::TILE_WIDTH);

    uint32_t num_patches_tile_padded = tt::round_up(num_patches, tt::constants::TILE_HEIGHT);

    // NOTE: Should this be padded up to tile_size for tilize_block?
    uint32_t patch_size_bytes =
        tt::round_up(patch_size, tt::constants::TILE_WIDTH) * dtype_bytes;  // bytes per patch row
    // NOTE: Also padded up to tile size
    uint32_t C_out_block_bytes = C_out_block * dtype_bytes;  // bytes per output channel row

    log_info("Block sizes:");
    log_info("  T_out_block: {}", config.T_out_block);
    log_info("  H_out_block: {}", config.H_out_block);
    log_info("  W_out_block: {}", config.W_out_block);
    log_info("  C_out_block: {}", C_out_block);
    log_info("  C_out_num_blocks: {}", C_out_num_blocks);
    log_info("Patch size: {}", patch_size);
    log_info("Num patches: {}", num_patches);
    log_info("Patch size bytes: {}", patch_size_bytes);
    log_info("C_out block bytes: {}", C_out_block_bytes);
    log_info("Num patches tile padded: {}", num_patches_tile_padded);
    log_info("Matmul M_t: {}", matmul_M_t);
    log_info("Matmul K_t: {}", matmul_K_t);
    log_info("Matmul N_t: {}", matmul_N_t);
    // Log CB sizes
    log_info("CB vol2col_rm: page_size={} bytes, num_pages={}", patch_size_bytes, num_patches);

    log_info("CB vol2col_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_M_t * matmul_K_t);

    log_info("CB weight_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_K_t * matmul_N_t);

    log_info("CB matmul_interm_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_M_t * matmul_N_t);

    log_info("CB matmul_result_rm: page_size={} bytes, num_pages={}", C_out_block_bytes, num_patches_tile_padded);

    uint32_t cb_vol2col_rm_id = tt::CBIndex::c_0;
    uint32_t cb_vol2col_tiled_id = tt::CBIndex::c_1;
    uint32_t cb_weight_tiled_id = tt::CBIndex::c_2;
    uint32_t cb_matmul_interm_tiled_id = tt::CBIndex::c_3;
    uint32_t cb_matmul_result_rm_id = tt::CBIndex::c_4;
    uint32_t cb_bias_tiled_id = tt::CBIndex::c_5;

    // Create circular buffers for vol2col, weights, bias and matmul intermediates
    auto [_, cb_vol2col_rm_handle] =
        tt::tt_metal::create_cb(cb_vol2col_rm_id, program, core_grid, patch_size_bytes, num_patches, data_format);

    auto [__, cb_vol2col_tiled_handle] = tt::tt_metal::create_cb(
        cb_vol2col_tiled_id, program, core_grid, tile_size, matmul_M_t * matmul_K_t, data_format);

    auto [___, cb_weight_tiled_handle] = tt::tt_metal::create_cb(
        cb_weight_tiled_id, program, core_grid, tile_size, matmul_K_t * matmul_N_t, data_format);

    auto [_____, cb_matmul_interm_tiled_handle] = tt::tt_metal::create_cb(
        cb_matmul_interm_tiled_id, program, core_grid, tile_size, matmul_M_t * matmul_N_t, data_format);

    // NOTE: Most kernels create RM CB with tile_size pages and num_tile number of pages.
    // Using stick pages led to PCC issues.
    auto [______, cb_matmul_result_rm_handle] = tt::tt_metal::create_cb(
        cb_matmul_result_rm_id,
        program,
        core_grid,
        tile_size,
        matmul_M_t * matmul_N_t,  // untilize will write padded rows, so this must be sized to avoid overflowing CB
        data_format);

    if (use_bias) {
        auto [____, cb_bias_tiled_handle] =
            tt::tt_metal::create_cb(cb_bias_tiled_id, program, core_grid, tile_size, matmul_N_t, data_format);
    }

    bool is_padding_zeros = config.padding_mode == "zeros";

    uint32_t in_row_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t out_row_size_bytes = output_tensor.buffer()->aligned_page_size();

    tt::log_info("Input tensor shape: N={}, T={}, H={}, W={}, C={}", N, T_in, H_in, W_in, C_in);
    tt::log_info("Output tensor shape: T={}, H={}, W={}, C={}", T_out, H_out, W_out, C_out);
    tt::log_info("Kernel size: {}x{}x{}", config.kernel_size[0], config.kernel_size[1], config.kernel_size[2]);
    tt::log_info("Stride: {}x{}x{}", config.stride[0], config.stride[1], config.stride[2]);
    tt::log_info("Padding: {}x{}x{}", config.padding[0], config.padding[1], config.padding[2]);
    tt::log_info("Groups: {}", config.groups);
    tt::log_info("Patch size: {}", patch_size);
    tt::log_info("Input row size (bytes): {}", in_row_size_bytes);
    tt::log_info("Output row size (bytes): {}", out_row_size_bytes);
    tt::log_info("Data format: {}", data_format);

    std::vector<uint32_t> reader_compile_time_args = {
        cb_vol2col_rm_id,
        N,
        T_in,
        H_in,
        W_in,
        C_in,
        T_out,
        H_out,
        W_out,
        C_out,
        config.padding[0],
        config.padding[1],
        config.padding[2],
        config.kernel_size[0],
        config.kernel_size[1],
        config.kernel_size[2],
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        in_row_size_bytes,
        out_row_size_bytes,
        is_padding_zeros,
    };

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv3d/device/kernels/reader_vol2col.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Matmul parameters
    IDevice* device = input_tensor.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t in0_block_w = matmul_K_t;

    const uint32_t out_subblock_w = std::min(matmul_N_t, dst_size);
    TT_FATAL(matmul_N_t % out_subblock_w == 0, "matmul_N_t must be divisible by out_subblock_w");
    // If out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to maintain
    // row-major intermediate buffer.
    const uint32_t out_subblock_h =
        (out_subblock_w == matmul_N_t) ? (std::min(matmul_M_t, dst_size / out_subblock_w)) : 1;

    const uint32_t in0_num_subblocks = matmul_M_t / out_subblock_h;
    const uint32_t in1_num_subblocks = matmul_N_t / out_subblock_w;

    log_info("Matmul parameters:");
    log_info("  matmul_M_t: {}", matmul_M_t);
    log_info("  matmul_K_t: {}", matmul_K_t);
    log_info("  matmul_N_t: {}", matmul_N_t);
    log_info("  dst_size: {}", dst_size);
    log_info("  in0_block_w: {}", in0_block_w);
    log_info("  out_subblock_w: {}", out_subblock_w);
    log_info("  out_subblock_h: {}", out_subblock_h);
    log_info("  in0_num_subblocks: {}", in0_num_subblocks);
    log_info("  in1_num_subblocks: {}", in1_num_subblocks);

    std::vector<uint32_t> compute_compile_time_args = {
        cb_vol2col_rm_id,
        cb_vol2col_tiled_id,
        cb_weight_tiled_id,
        cb_bias_tiled_id,
        cb_matmul_interm_tiled_id,
        cb_matmul_result_rm_id,
        num_patches,
        matmul_M_t,
        matmul_K_t,
        matmul_N_t,
        (uint32_t)use_bias,
        T_out,
        H_out,
        W_out,
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        in0_num_subblocks,
        in1_num_subblocks,
        in0_block_w,
        out_subblock_h,
        out_subblock_w};

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv3d/device/kernels/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        cb_matmul_result_rm_id,
        cb_weight_tiled_id,
        cb_bias_tiled_id,
        N,
        T_out,
        H_out,
        W_out,
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        matmul_M_t,
        matmul_K_t,
        matmul_N_t,
        num_patches_tile_padded,
        out_row_size_bytes,
        C_out_block_bytes,
        (uint32_t)use_bias,
    };

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv3d/device/kernels/writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    CoreCoord core = {0, 0};
    SetRuntimeArgs(
        program, reader_kernels_id, core, {input_tensor.buffer()->address(), output_tensor.buffer()->address()});

    uint32_t out_addr = output_tensor.buffer()->address();
    uint32_t weight_addr = weight_tensor.buffer()->address();
    uint32_t bias_addr = bias_tensor.has_value() ? bias_tensor.value().buffer()->address() : 0;
    log_info("Out addr: {}", out_addr);
    SetRuntimeArgs(program, writer_kernels_id, core, {out_addr, weight_addr, bias_addr});

    auto override_runtime_arguments_callback =
        [](const void* operation,
           Program& program,
           const std::vector<Tensor>& input_tensors,
           const std::vector<std::optional<const Tensor>>& optional_input_tensors,
           const std::vector<Tensor>& output_tensors) { TT_FATAL(false, "not implemented"); };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::conv::conv3d::detail
