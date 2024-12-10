// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "tt_metal/common/core_coord.hpp"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include <sstream>
#include <type_traits>
#include <ranges>

#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"

#include <optional>
using namespace tt::constants;

namespace ttnn {



using namespace ccl;
// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
operation::ProgramWithCallbacks all_gather_multi_core_with_workers_new(
    const Tensor& input_tensor,
    std::optional<Device*> forward_device,
    std::optional<Device*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore semaphore_handle)//,
    // std::optional<ccl::SyncModeSpec> sync_details)
     {
    tt::tt_metal::Program program{};
    auto global_semaphore_addr = semaphore_handle.address();

    // Sleep for ring_index * 5 seconds to stagger startup
    std::this_thread::sleep_for(std::chrono::seconds(ring_index * 5));

    auto drain_sync_core = CoreCoord(4,4);
    std::optional<ccl::SyncModeSpec> sync_details = ttnn::ccl::SyncModeSpec {
        1, // num_device
        drain_sync_core,
        // {CreateSemaphore(program, {drain_sync_core}, 0)},
        // {CreateGlobalSemaphore(input_tensor.device(), {drain_sync_core}, 0)},
        {global_semaphore_addr},
        {ring_size * num_links}
    };
    log_info(tt::LogOp, "DEBUG: device: {}", input_tensor.device()->id());

    Device *device = input_tensor.device();
    auto local_device_fabric_interface = ccl::EdmLineFabricOpInterface (
        device,
        forward_device,
        backward_device,
        &program,
        num_links);

    LineTopology line_topology(ring_size, ring_index);

    std::unique_ptr<ccl::CclOpTensorConfig> input_tensor_config = ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(input_tensor);
    std::unique_ptr<ccl::CclOpTensorConfig> output_tensor_config = ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(output_tensor);

    bool is_sharded = input_tensor.is_sharded();

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    auto const& op_config =ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto const& input_tensor_partition = ttnn::ccl::TensorPartition(1, 0);  // one partition, 0 index
    auto const& output_tensor_partition = ttnn::ccl::TensorPartition(ring_size, ring_index);  // ring_size partitions, ring_index index

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    auto const& sender_worker_core_range = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_workers_per_link-1, num_links - 1)));
    auto const& sender_worker_cores = corerange_to_cores(sender_worker_core_range, std::nullopt, true);

    // L1 Scratch CB Creation
    uint32_t num_pages_per_packet = 1; // we assume 1 page per packet for now
    uint32_t cb_num_pages = 3*num_pages_per_packet; // tripple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size() + sizeof(tt::fabric::PacketHeader);
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Create Tensor slicer
    // read the entire input tensor (partition size = 1, partition index = 0)
    // write to the output tensor on its corresponding partition (partition size = ring_size, partition index = ring_index)
    auto input_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicer (
        input_tensor,
        input_tensor,
        dim,
        0, // partition index
        1, // partition size
        num_links, // num_workers_per_slicer, set 1 per link for now
        UINT32_MAX, // max_worker_slice_in_bytes, set as infinite for now
        cb_num_pages / 2);
    auto output_tensor_slicer = ttnn::ccl::GenericWrappedTensorSlicer (
        output_tensor,
        output_tensor,
        dim,
        ring_index, // partition index
        ring_size, // partition size
        num_links, // num_workers_per_slicer, set 1 per link for now
        UINT32_MAX, // max_worker_slice_in_bytes, set as infinite for now
        cb_num_pages / 2);

    // KERNEL CREATION
    auto worker_arg_builder = ccl::worker_detail::CCLWorkerArgBuilder(
        device,
        op_config,
        input_tensor_partition,
        output_tensor_partition,
        dim);

    auto const& worker_defines = op_config.emit_worker_defines();
    static std::string const& sender_kernel_reader_path = "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader.cpp";
    static std::string const& sender_kernel_writer_path = "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_writer.cpp";

    // KernelHandle worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
    //     program,
    //     sender_kernel_reader_path,
    //     sender_worker_core_range,
    //     tt::tt_metal::ReaderDataMovementConfig(worker_arg_builder.generate_sender_reader_kernel_ct_args(), worker_defines));
    KernelHandle worker_sender_reader_kernel_id = ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
        program,
        {src0_cb_index},
        {&input_tensor},
        sender_worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig{},
        1 // num_command_streams
    );

    KernelHandle worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_kernel_writer_path,
        sender_worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(worker_arg_builder.generate_sender_writer_kernel_ct_args(), worker_defines));

    const size_t forward_direction_distance_to_end_of_line = line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::FORWARD);
    const size_t backward_direction_distance_to_end_of_line = line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD);
    // RT Args
    log_info(tt::LogOp, "DEBUG: CreateSemaphore: {}, &program: {}", input_tensor.device()->id(), (void*)&program);
    size_t sender_worker_forward_flow_control_semaphore_id = CreateSemaphore(program, sender_worker_core_range,0);
    size_t sender_worker_forward_buffer_index_semaphore_id = CreateSemaphore(program, sender_worker_core_range,0);
    size_t sender_worker_backward_flow_control_semaphore_id = CreateSemaphore(program, sender_worker_core_range,0);
    size_t sender_worker_backward_buffer_index_semaphore_id = CreateSemaphore(program, sender_worker_core_range,0);

    auto reader_tensor_slices = ttnn::ccl::cmd::builder::generate_worker_tensor_slices(
        1, input_tensor, num_workers_per_link*num_links, dim);
    log_info(tt::LogOp, "reader_tensor_slices size: {}", reader_tensor_slices.size());
    log_info(tt::LogOp, "reader_tensor_slices[0] size: {}", reader_tensor_slices[0].size());

    for (std::size_t link = 0; link < num_links; link++) {
        CoreCoord core = {num_workers_per_link-1, link};
        std::size_t worker_tensor_slice_index = link;
        auto const& input_worker_slice = input_tensor_slicer.get_worker_slice(worker_tensor_slice_index);
        auto const& output_worker_slice = output_tensor_slicer.get_worker_slice(worker_tensor_slice_index);
        auto worker_arg_builder = ccl::worker_detail::CCLWorkerArgBuilder(
            device,
            op_config,
            input_tensor_partition,
            output_tensor_partition,
            dim);

        // tt::log_info("Creating RT Args for worker core ({},{})", core.x, core.y);
        log_info(tt::LogOp, "reference input worker slice");
        input_worker_slice.print();
        log_info(tt::LogOp, "reference output worker slice");
        output_worker_slice.print();

        // auto const sender_reader_rt_args = worker_arg_builder.generate_sender_reader_kernel_rt_args(input_worker_slice, worker_arg_builder.operating_dim, num_pages_per_packet, worker_tensor_slice_index);
        // tt::tt_metal::SetRuntimeArgs(
        //     program,
        //     worker_sender_reader_kernel_id,
        //     core,
        //     sender_reader_rt_args);
        std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> reader_cmd_stream;
        // reader_cmd_stream.push_back(
        // ttnn::ccl::cmd::uops::read_tensor_slice_to_cb_for_eventual_fabric_write(reader_tensor_slices[link][0], src0_cb_index));
        auto const& input_worker_slice_v2 = input_tensor_slicer.get_worker_slice_v2(worker_tensor_slice_index);
        reader_cmd_stream.push_back( // use the reader_tensor_slices after the bug is fixed
        ttnn::ccl::cmd::uops::read_tensor_slice_to_cb_for_eventual_fabric_write(input_worker_slice_v2, src0_cb_index));

        // void print(ttnn::ccl::v2::TensorSlice const& slice_v2) const {
        //     log_info("TensorSlice:");
        //     log_info("  tensor_shape:        [w=%u, z=%u, y=%u, x=%u]",
        //         slice_v2.tensor_shape.w, slice_v2.tensor_shape.z, slice_v2.tensor_shape.y, slice_v2.tensor_shape.x);
        //     log_info("  tensor_slice_shape:  [w=%u, z=%u, y=%u, x=%u]",
        //         slice_v2.tensor_slice_shape.w, slice_v2.tensor_slice_shape.z, slice_v2.tensor_slice_shape.y, slice_v2.tensor_slice_shape.x);
        //     log_info("  tensor_slice_offset: [w=%u, z=%u, y=%u, x=%u]",
        //         slice_v2.tensor_slice_offset.w, slice_v2.tensor_slice_offset.z, slice_v2.tensor_slice_offset.y, slice_v2.tensor_slice_offset.x);
        //     log_info("  worker_slice_shape:  [w=%u, z=%u, y=%u, x=%u]",
        //         slice_v2.worker_slice_shape.w, slice_v2.worker_slice_shape.z, slice_v2.worker_slice_shape.y, slice_v2.worker_slice_shape.x);
        //     log_info("  worker_slice_offset: [w=%u, z=%u, y=%u, x=%u]",
        //         slice_v2.worker_slice_offset.w, slice_v2.worker_slice_offset.z, slice_v2.worker_slice_offset.y, slice_v2.worker_slice_offset.x);
        // }
        // log_info(tt::LogOp, "DEBUG: tensor slice v2 new:");
        // reader_tensor_slices[link][0].print();
        // log_info(tt::LogOp, "DEBUG: tensor slice v2 old:");
        // input_worker_slice_v2.print();

        ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
            program,
            worker_sender_reader_kernel_id,
            {&input_tensor},
            {op_config.get_page_size()},
            input_tensor.device(),
            num_pages_per_packet, // num_pages_per_edm_buffer
            {core},
            reader_cmd_stream,
            std::nullopt,
            std::nullopt,
            std::nullopt
        );

        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> forward_fabric_connection =
            line_topology.is_first_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD) ?
            std::nullopt :
            std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(local_device_fabric_interface.uniquely_connect_worker(device, ttnn::ccl::EdmLineFabricOpInterface::FORWARD));
        std::optional<ttnn::ccl::SenderWorkerAdapterSpec> backward_fabric_connection =
            line_topology.is_last_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD) ?
            std::nullopt :
            std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(local_device_fabric_interface.uniquely_connect_worker(device, ttnn::ccl::EdmLineFabricOpInterface::BACKWARD));

        log_info(tt::LogOp, "DEBUG: line_index: {}, line_size: {}, forward_fabric_connection: {}", line_topology.line_index(), line_topology.line_size(), forward_fabric_connection.has_value());
        log_info(tt::LogOp, "DEBUG: line_index: {}, line_size: {}, backward_fabric_connection: {}", line_topology.line_index(), line_topology.line_size(), backward_fabric_connection.has_value());

        auto const sender_writer_rt_args = worker_arg_builder.generate_sender_writer_kernel_rt_args(
            forward_fabric_connection,
            sender_worker_forward_flow_control_semaphore_id,
            sender_worker_forward_buffer_index_semaphore_id,
            backward_fabric_connection,
            sender_worker_backward_flow_control_semaphore_id,
            sender_worker_backward_buffer_index_semaphore_id,
            forward_direction_distance_to_end_of_line,
            backward_direction_distance_to_end_of_line,
            output_worker_slice,
            worker_arg_builder.operating_dim,
            num_pages_per_packet,
            worker_tensor_slice_index,
            sync_details);
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_sender_writer_kernel_id,
            core,
            sender_writer_rt_args);
    }

    if (sync_details.has_value()) {
        ttnn:L:ccl::worker_detail::build_sync_kernels(device, program, sync_details.value(), true, local_device_fabric_interface);
    }

    local_device_fabric_interface.build_kernels();

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         semaphore_handle,
         sender_worker_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto& input = input_tensors[0];
        const auto& output = output_tensors[0];

        // update senders
        auto &worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
        auto &worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
        for (auto const& core : sender_worker_cores) {
            // reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args.at(0) = input.buffer()->address();
            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args.at(0) = output.buffer()->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace ttnn
