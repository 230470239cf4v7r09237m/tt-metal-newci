// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "all_reduce_async_op.hpp"
#include "ttnn/operations/math.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

#include <tt-metalium/host_api.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
namespace ccl {
namespace all_reduce_detail {

AllReduceAsync create_all_reduce_async_struct(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<SubDeviceId>& sub_device_id,
    bool enable_persistent_fabric_mode) {
    uint32_t num_devices = devices.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    std::optional<GlobalSemaphore> semaphore = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            device_index = i;
            semaphore = semaphores.at(i);  // Get raw pointer
            if (i != 0) {
                backward_device = devices.at(i - 1);
            }
            if (i != num_devices - 1) {
                forward_device = devices.at(i + 1);
            }
        }
    }

    return ttnn::AllReduceAsync{
        forward_device,
        backward_device,
        num_links,
        num_devices,
        device_index,
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        semaphore.value(),
        sub_device_id,
        enable_persistent_fabric_mode};
}

uint32_t find_scatter_dim(const ttnn::Shape& input_tensor_padded_shape, size_t num_workers) {
    // iterate until we find a dimension that is divisible by num_workers
    TT_FATAL(input_tensor_padded_shape.size() == 4, "Expected input tensor to have 4 dimensions");
    ttnn::Shape input_tensor_shape_in_tiles{
        input_tensor_padded_shape[0],
        input_tensor_padded_shape[1],
        input_tensor_padded_shape[2] / tt::constants::TILE_HEIGHT,
        input_tensor_padded_shape[3] / tt::constants::TILE_WIDTH};
    for (uint32_t dim = 0; dim < 4; ++dim) {
        if (input_tensor_shape_in_tiles[dim] % num_workers == 0) {
            tt::log_debug(
                "Found scatter dimension {} for input tensor with padded shape {}", dim, input_tensor_padded_shape);
            return dim;
        }
    }
    TT_THROW(
        "No scatter dimension found for input tensor with padded shape {} and num_workers {}",
        input_tensor_padded_shape,
        num_workers);
}

}  // namespace all_reduce_detail
}  // namespace ccl

void AllReduceAsync::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout);
}

static void validate_output_tensor_allocation(const std::vector<Tensor>& output_tensors) {
    for (const auto& output_tensor : output_tensors) {
        const auto& buffers = output_tensor.buffers();
        const auto first_address = buffers.front()->address();
        TT_FATAL(
            std::all_of(
                buffers.begin(),
                buffers.end(),
                [&first_address](const auto& buffer) {
                    return buffer != nullptr && buffer->address() == first_address;
                }),
            "Output buffers for all_reduce async must be lock-step allocated but some of the tensors were allocated at "
            "different addresses across devices.");
    }
}

std::vector<ttnn::TensorSpec> AllReduceAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.get_padded_shape();  // TODO: Replace with get_logical_shape()
    return {TensorSpec(
        shape,
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), output_mem_config))};
}

operation::ProgramWithCallbacks AllReduceAsync::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");

    auto input_tensor_shape = input_tensors[0].get_padded_shape();
    auto input_tensor_buffer_layout = input_tensors[0].buffer()->buffer_layout();
    auto input_tensor_page_layout = input_tensors[0].layout();

    auto input_tensor_memory_config = input_tensors[0].memory_config();
    auto output_tensor_memory_config = output_tensors[0].memory_config();
    uint32_t input_shard_num_cores = input_tensor_memory_config.shard_spec->grid.num_cores();
    uint32_t output_shard_num_cores = output_tensor_memory_config.shard_spec->grid.num_cores();

    tt::log_debug(tt::LogOp, "input_tensor_shape: {}", input_tensor_shape);
    tt::log_debug(tt::LogOp, "input_tensor_memory_config: {}", input_tensor_memory_config);
    tt::log_debug(tt::LogOp, "output_tensor_memory_config: {}", output_tensor_memory_config);
    tt::log_debug(tt::LogOp, "input_shard_num_cores: {}", input_shard_num_cores);
    tt::log_debug(tt::LogOp, "output_shard_num_cores: {}", output_shard_num_cores);
    tt::log_debug(
        tt::LogOp, "input_tensor_memory_config.shard_spec->shape: {}", input_tensor_memory_config.shard_spec->shape);
    tt::log_debug(
        tt::LogOp, "output_tensor_memory_config.shard_spec->shape: {}", output_tensor_memory_config.shard_spec->shape);

    tt::log_info(tt::LogOp, "Running TG Llama specific all_reduce_async_minimal_multi_core_with_workers");
    return all_reduce_async_minimal_multi_core_with_workers(
        input_tensors[0],
        this->forward_device,
        this->backward_device,
        output_tensors[0],
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->topology,
        this->semaphore,
        this->sub_device_id,
        this->enable_persistent_fabric_mode);
}

const operation::Hash AllReduceAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    return operation::hash_operation<AllReduceAsync>(
        input_tensors[0].get_padded_shape(),
        input_tensors[0].get_dtype(),
        input_tensors[0].memory_config(),
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->output_mem_config,
        this->topology,
        this->semaphore);
}

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_reduce_async(
    const Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode) {
    TT_FATAL(
        topology == ttnn::ccl::Topology::Linear,
        "This all_reduce API with cluster_axis is currently supported only for the Linear topology");
    const auto mesh_view = mesh_device.get_view();
    auto devices = input_tensor.get_workers();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;

    operation::launch_op(
        [num_preferred_links,
         memory_config,
         mesh_view,
         cluster_axis,
         num_devices,
         topology,
         semaphores,
         subdevice_id,
         enable_persistent_fabric_mode](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_device_tensor = input_tensors.at(0);

            const auto coordinate = mesh_view.find_device(input_device_tensor.device()->id());
            std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(coordinate.col)
                                                                : mesh_view.get_devices_on_row(coordinate.row);

            const auto& input_tensor = input_tensors.at(0);

            return operation::run(
                ttnn::ccl::all_reduce_detail::create_all_reduce_async_struct(
                    input_device_tensor,
                    num_preferred_links.has_value() ? num_preferred_links.value() : 1,
                    memory_config,
                    devices,
                    topology,
                    semaphores,
                    subdevice_id,
                    enable_persistent_fabric_mode),
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
