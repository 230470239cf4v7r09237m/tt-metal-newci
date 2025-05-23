// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

#include "cpp/ttnn/operations/data_movement/pad/pad.hpp"
#include "cpp/ttnn/operations/copy.hpp"

namespace ttnn {
namespace ccl {
namespace all_gather_detail {

AllGather create_all_gather_struct(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology) {
    uint32_t num_devices = devices.size();
    auto [device_index, sender_device_id, receiver_device_id] =
        get_device_index_and_sender_receiver_ids(input_tensor, devices, topology);

    return ttnn::AllGather{
        dim,
        num_links,
        num_devices,
        device_index,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel,
        receiver_device_id,
        sender_device_id,
        memory_config.value_or(input_tensor.memory_config()),
        topology};
}
}  // namespace all_gather_detail
}  // namespace ccl

AllGatherBidirectionalMode AllGatherConfig::choose_bidirectional_mode(Tensor const& input_tensor, bool fuse_op) {
    if (fuse_op) {
        return AllGatherBidirectionalMode::FULL_TENSOR;
    }

    std::size_t eth_l1_capacity = tt::tt_metal::hal::get_erisc_l1_unreserved_size();
    std::size_t tensor_size_bytes = input_tensor.volume() * input_tensor.element_size();
    // This is currently a guestimate. We need a lot more hard data to identify where this dividing line is.
    bool perf_degradation_from_full_tensor_mode = tensor_size_bytes > (2 * eth_l1_capacity);
    if (perf_degradation_from_full_tensor_mode) {
        return AllGatherBidirectionalMode::SPLIT_TENSOR;
    }
    return AllGatherBidirectionalMode::FULL_TENSOR;
}

AllGatherConfig::AllGatherConfig(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    uint32_t dim,
    uint32_t ring_size,
    uint32_t num_links,
    ttnn::ccl::Topology topology,
    std::size_t num_edm_buffers_per_channel,
    bool fuse_op,
    const std::optional<size_t> user_defined_num_workers) :
    num_links(num_links),
    semaphore_size(32),
    ring_size(ring_size),

    erisc_handshake_address(tt::round_up(tt::tt_metal::hal::get_erisc_l1_unreserved_base(), 16)),
    topology(topology),
    enable_bidirectional(topology == ttnn::ccl::Topology::Ring),

    input_is_dram(input_tensor.buffer()->buffer_type() == BufferType::DRAM),
    output_is_dram(output_tensor.buffer()->buffer_type() == BufferType::DRAM),

    bidirectional_mode(choose_bidirectional_mode(input_tensor, fuse_op)),
    enable_merged_payload_and_channel_sync(true),
    num_edm_buffers_per_channel(num_edm_buffers_per_channel) {
    TT_FATAL(num_edm_buffers_per_channel > 0, "num_edm_buffers_per_channel must be > 0");
    TT_ASSERT(erisc_handshake_address >= tt::tt_metal::hal::get_erisc_l1_unreserved_base());
    TT_ASSERT(erisc_handshake_address < tt::tt_metal::hal::get_erisc_l1_unreserved_base() + 16);
    TT_ASSERT((erisc_handshake_address & (16 - 1)) == 0);
    if (input_tensor.get_layout() == Layout::TILE && dim != 3) {
        // See issue #6448
        int outer_dims_size = 1;
        for (std::size_t i = 0; i < dim; i++) {
            outer_dims_size *= input_tensor.get_padded_shape()[i];
        }
        if (outer_dims_size > 1) {
            this->enable_bidirectional = false;
        }
    }

    // "duplicate" directions are a short hand to enable linear/mesh all-gather topologies with
    // less code-changes. Ideally a new concept is added amongst "num_eth_buffers", "num_workers_per_link", etc.
    uint32_t num_duplicate_directions =
        (topology == ttnn::ccl::Topology::Ring && bidirectional_mode != AllGatherBidirectionalMode::FULL_TENSOR) ? 1
                                                                                                                 : 2;

    uint32_t total_l1_buffer_space = tt::tt_metal::hal::get_erisc_l1_unreserved_size();

    this->is_sharded = input_tensor.is_sharded();
    if (user_defined_num_workers.has_value()) {
        this->num_eth_buffers = user_defined_num_workers.value() / num_duplicate_directions;
    } else {
        this->num_eth_buffers =
            (this->enable_bidirectional ? 8 /*1*/ : (topology != ttnn::ccl::Topology::Linear ? 8 : 4));
    }

    constexpr std::int32_t MAX_NUM_CONCURRENT_TRANSACTIONS = 8;
    if (bidirectional_mode == AllGatherBidirectionalMode::FULL_TENSOR) {
        this->num_eth_buffers =
            std::min(this->num_eth_buffers, MAX_NUM_CONCURRENT_TRANSACTIONS / num_duplicate_directions);
    }

    this->num_workers_per_link = this->num_eth_buffers;
    this->eth_sems_l1_base_byte_address = this->erisc_handshake_address + 16 * 3;  // 16;
    // Really should be called offset_after_semaphore_region
    this->semaphore_offset =
        this->semaphore_size * this->num_eth_buffers *
        num_duplicate_directions;  // TODO: Remove this once dedicated semaphore space for user kernels are added
    this->eth_buffers_l1_base_byte_address = this->eth_sems_l1_base_byte_address + this->semaphore_offset;

    std::size_t channel_sync_bytes_overhead = (enable_merged_payload_and_channel_sync * 16);
    uint32_t const page_size = input_tensor.buffer()->page_size();
    std::size_t l1_per_buffer_region =
        ((total_l1_buffer_space - this->semaphore_offset) /
         (this->num_eth_buffers * num_duplicate_directions * this->num_edm_buffers_per_channel)) -
        channel_sync_bytes_overhead;
    this->eth_buffer_size = tt::round_down(l1_per_buffer_region, page_size);

    TT_FATAL(
        (this->eth_buffer_size + channel_sync_bytes_overhead) *
                    (this->num_eth_buffers * num_duplicate_directions * this->num_edm_buffers_per_channel) +
                this->semaphore_offset <=
            total_l1_buffer_space,
        "Error");
    TT_FATAL(
        eth_buffer_size == 0 or (this->num_eth_buffers * num_duplicate_directions) <= MAX_NUM_CONCURRENT_TRANSACTIONS,
        "Error");
}

void AllGather::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    // TODO: This can be removed by passing two page sizes, actual and aligned to be used for address offsets
    // Buffer sizes also need to take this aligned page size into consideration
    // TODO: Validate ring
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");
    TT_FATAL(
        this->receiver_device_id.has_value() || this->sender_device_id.has_value(),
        "Error, All-gather was unable to identify either a sender or receiver device ID and atleast one must be "
        "identified for a valid all-gather configuration. The input mesh tensor or all-gather arguments may be "
        "incorrect");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout);

    // Sharding Config checks
    bool input_sharded = input_tensor.is_sharded();
    if (input_sharded) {
        // TODO(snijjar)
    }
}

std::vector<ttnn::TensorSpec> AllGather::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    auto output_shape = input_tensors[0].get_logical_shape();
    output_shape[this->dim] *= this->ring_size;

    const auto& input_tensor = input_tensors[0];
    TensorSpec spec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), output_mem_config));
    if (this->output_mem_config.is_sharded()) {
        return {TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), output_mem_config))};
    }
    return std::vector<TensorSpec>(input_tensors.size(), spec);
}

std::vector<Tensor> AllGather::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, {});
}

tt::tt_metal::operation::ProgramWithCallbacks AllGather::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return all_gather_multi_core_with_workers(
        input_tensors[0],
        output_tensors[0],
        this->dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->receiver_device_id,
        this->sender_device_id,
        this->topology,
        this->user_defined_num_workers,
        this->user_defined_num_buffers_per_channel);
}

namespace operations {
namespace ccl {

Tensor all_gather(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "all_gather op is only supported for Fast Dispatch");
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "all_gather op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    int32_t rank = input_tensor.get_logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
    tt::tt_metal::operation::launch_op(
        [gather_dim,
         num_links,
         dim,
         num_devices,
         memory_config,
         user_defined_num_workers,
         user_defined_num_buffers_per_channel,
         devices,
         ccl_topology](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto input_tensor = input_tensors.at(0);

            ttnn::SmallVector<uint32_t> unpad_elements = {
                input_tensor.get_logical_shape()[-4],
                input_tensor.get_logical_shape()[-3],
                input_tensor.get_logical_shape()[-2],
                input_tensor.get_logical_shape()[-1]};

            const uint32_t w_pad = input_tensor.get_logical_shape()[-1] % tt::constants::TILE_WIDTH;
            const uint32_t h_pad = input_tensor.get_logical_shape()[-2] % tt::constants::TILE_HEIGHT;
            bool needs_padding = input_tensor.get_layout() == Layout::TILE && (h_pad != 0 || w_pad != 0);
            if (needs_padding) {
                ttnn::SmallVector<std::pair<uint32_t, uint32_t>> padding = {{0, 0}, {0, 0}, {0, h_pad}, {0, w_pad}};
                DataType original_dtype = input_tensor.get_dtype();
                if (input_tensor.get_dtype() != DataType::BFLOAT16 && input_tensor.get_dtype() != DataType::FLOAT32) {
                    input_tensor = ttnn::typecast(input_tensor, DataType::BFLOAT16);
                }
                input_tensor = ttnn::pad(input_tensor, padding, 0, false, std::nullopt);
                if (original_dtype != input_tensor.get_dtype()) {
                    input_tensor = ttnn::typecast(input_tensor, original_dtype);
                }
            }

            auto output_tensor = tt::tt_metal::operation::run(
                ttnn::ccl::all_gather_detail::create_all_gather_struct(
                    input_tensor,
                    gather_dim,
                    num_links,
                    memory_config,
                    user_defined_num_workers,
                    user_defined_num_buffers_per_channel,
                    devices,
                    ccl_topology),
                {input_tensor});

            if (needs_padding) {
                return ttnn::ccl::unpad_output_tensor(output_tensor, num_devices, unpad_elements, dim);
            } else {
                return output_tensor;
            }
        },
        {input_tensor},
        output_tensors);

    return output_tensors.at(0);
}

Tensor all_gather(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    TT_FATAL(
        topology == ttnn::ccl::Topology::Linear,
        "This all_gather API with cluster_axis is currently supported only for the Linear topology");
    const auto mesh_view = mesh_device.get_view();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input_tensor.get_logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};

    tt::tt_metal::operation::launch_op(
        [gather_dim,
         num_links,
         memory_config,
         mesh_view,
         cluster_axis,
         user_defined_num_workers,
         user_defined_num_buffers_per_channel,
         num_devices,
         topology](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_device_tensor = input_tensors.at(0);

            TT_FATAL(
                mesh_view.is_mesh_2d(),
                "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
            const auto coordinate = mesh_view.find_device(input_device_tensor.device()->id());
            const auto view_index = (cluster_axis == 0) ? coordinate[1] : coordinate[0];
            const auto device_index = (cluster_axis == 0) ? coordinate[0] : coordinate[1];

            auto get_chip_id = [&](std::size_t line_index) -> std::optional<chip_id_t> {
                auto new_row = coordinate[0];
                auto new_col = coordinate[1];
                if (cluster_axis == 0) {
                    new_row = line_index % num_devices;
                } else {
                    new_col = line_index % num_devices;
                }
                return mesh_view.find_device_id(MeshCoordinate(new_row, new_col));
            };

            bool is_last_chip_in_clockwise_direction = device_index == (num_devices - 1);
            bool is_last_chip_in_counter_clockwise_direction = device_index == 0;
            auto receiver_device_id =
                is_last_chip_in_clockwise_direction ? std::nullopt : get_chip_id(device_index + 1);
            auto sender_device_id = is_last_chip_in_counter_clockwise_direction
                                        ? std::nullopt
                                        : get_chip_id(device_index + num_devices - 1);

            return tt::tt_metal::operation::run(
                ttnn::AllGather{
                    gather_dim,
                    num_links,
                    num_devices,
                    device_index,
                    user_defined_num_workers,
                    user_defined_num_buffers_per_channel,
                    receiver_device_id,
                    sender_device_id,
                    memory_config.value_or(input_device_tensor.memory_config()),
                    topology},
                {input_device_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace operations

}  // namespace ttnn
