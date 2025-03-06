// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_concat_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
namespace ccl {
namespace all_gather_concat_detail {

AllGatherConcat create_all_gather_concat_struct(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool enable_persistent_fabric_mode,
    const uint32_t num_heads,
    const bool on_subcoregrids) {
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

    return ttnn::AllGatherConcat{
        forward_device,
        backward_device,
        dim,
        num_links,
        num_devices,
        device_index,
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        semaphore.value(),
        sub_device_id,
        enable_persistent_fabric_mode,
        num_heads,
        on_subcoregrids};
}

}  // namespace all_gather_concat_detail
}  // namespace ccl

void AllGatherConcat::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
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

static void validate_output_tensor_alloc(const std::vector<Tensor>& output_tensors) {
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
            "Output buffers for all_gather async must be lock-step allocated but some of the tensors were allocated at "
            "different addresses across devices.");
    }
}

std::vector<ttnn::TensorSpec> AllGatherConcat::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto input_shape = input_tensor.get_padded_shape();  // TODO: Replace with get_logical_shape()
    auto num_heads = this->num_heads;
    auto sequence_length = input_shape[0];
    auto batch = input_shape[1];
    auto head_dim = input_shape[3];
    // pad batch to 32 if necessary
    if (batch < 32) {
        batch = 32;
    }
    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({sequence_length, 1, batch, hidden_dim});

    CoreRangeSet output_core_grid;
    if (this->on_subcoregrids) {
        const auto input_core_ranges = input_tensor.shard_spec().value().grid.ranges();
        CoreRangeSet input_core_grid = input_tensor.shard_spec().value().grid;
        const auto start_coord = input_core_ranges[0].start_coord;
        output_core_grid =
            tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_coord, num_heads, input_core_grid, true);
    } else {
        output_core_grid = tt::tt_metal::num_cores_to_corerangeset(
            num_heads, input_tensor.device()->compute_with_storage_grid_size(), true);
    }
    tt::tt_metal::ShardSpec shard_spec{output_core_grid, {batch, head_dim}};
    auto mem_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, tt::tt_metal::BufferType::L1};
    mem_config.shard_spec = shard_spec;

    return {TensorSpec(
        output_shape, tt::tt_metal::TensorLayout(input_tensor.get_dtype(), tt::tt_metal::Layout::TILE, mem_config))};
}

AllGatherConcatVersion AllGatherConcat::select_version(const Tensor& input_tensor) const {
    return AllGatherConcatVersion::LLAMA_POST_BINARY_MATMUL;
}

tt::tt_metal::operation::ProgramWithCallbacks AllGatherConcat::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");

    AllGatherConcatVersion version = select_version(input_tensors[0]);

    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));

    log_trace(tt::LogOp, "Detected all gather specialized shape. all_gather_concat_llama_post_binary_matmul is called");
    printf("in create program before calling concat\n");
    CoreCoord compute_with_storage_grid_size = input_tensors[0].device()->compute_with_storage_grid_size();
    if (this->on_subcoregrids) {
        return all_gather_concat_llama_post_binary_matmul_subgrids(
            input_tensors[0],
            this->forward_device,
            this->backward_device,
            output_tensors[0],
            this->dim,
            this->num_links,
            this->ring_size,
            this->ring_index,
            this->topology,
            this->semaphore,
            this->sub_device_id,
            this->enable_persistent_fabric_mode,
            this->num_heads);
    }
    return all_gather_concat_llama_post_binary_matmul(
        input_tensors[0],
        this->forward_device,
        this->backward_device,
        output_tensors[0],
        this->dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->topology,
        this->semaphore,
        this->sub_device_id,
        this->enable_persistent_fabric_mode,
        this->num_heads);
}

const tt::tt_metal::operation::Hash AllGatherConcat::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    AllGatherConcatVersion version = select_version(input_tensors[0]);
    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return tt::tt_metal::operation::hash_operation<AllGatherConcat>(
        this->dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->output_mem_config,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        this->num_heads);
}

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_concat(
    const Tensor& input_tensor,
    const uint32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_heads,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool enable_persistent_fabric_mode,
    bool on_subcoregrids) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "all_gather_concat op is only supported for Fast Dispatch");
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "all_gather_concat op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};

    tt::log_debug(
        tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    tt::log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    // create this semaphore for all cores since we don't know which core will be used for teardown draining
    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});

    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;

    tt::tt_metal::operation::launch_op(
        [dim,
         num_links,
         num_devices,
         memory_config,
         devices,
         ccl_topology,
         semaphores,
         sub_device_id,
         enable_persistent_fabric_mode,
         num_heads,
         on_subcoregrids](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors.at(0);

            return tt::tt_metal::operation::run(
                ttnn::ccl::all_gather_concat_detail::create_all_gather_concat_struct(
                    input_tensor,
                    dim,
                    num_links,
                    memory_config,
                    devices,
                    ccl_topology,
                    semaphores,
                    sub_device_id,
                    enable_persistent_fabric_mode,
                    num_heads,
                    on_subcoregrids),
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
