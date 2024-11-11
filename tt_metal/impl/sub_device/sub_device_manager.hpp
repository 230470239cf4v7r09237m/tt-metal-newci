// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/dispatch/memcpy.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "tt_metal/impl/sub_device/sub_device.hpp"
#include "tt_metal/impl/sub_device/sub_device_types.hpp"
#include "tt_metal/tt_stl/span.hpp"

namespace tt::tt_metal {

class TraceBuffer;

inline namespace v0 {
class Device;
}  // namespace v0

namespace detail {
class SubDeviceManager {
   public:
    static constexpr uint32_t MAX_NUM_SUB_DEVICES = 16;
    static_assert(MAX_NUM_SUB_DEVICES <= std::numeric_limits<SubDeviceId::Id>::max(), "MAX_NUM_SUB_DEVICES must be less than or equal to the max value of SubDeviceId::Id");
    // Constructor used for the default/global device
    SubDeviceManager(Device *device);
    // Constructor used for regular sub-devices
    SubDeviceManager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size, Device *device);

    SubDeviceManager(const SubDeviceManager &other) = delete;
    SubDeviceManager &operator=(const SubDeviceManager &other) = delete;

    SubDeviceManager(SubDeviceManager &&other) = default;
    SubDeviceManager &operator=(SubDeviceManager &&other) = default;

    ~SubDeviceManager();

    const SubDevice &sub_device(SubDeviceId sub_device_id) const;
    const vector_memcpy_aligned<uint32_t> &noc_mcast_data(SubDeviceId sub_device_id) const;
    const vector_memcpy_aligned<uint32_t> &noc_unicast_data(SubDeviceId sub_device_id) const;
    const vector_memcpy_aligned<uint32_t> &noc_mcast_unicast_data(SubDeviceId sub_device_id) const;

    const Allocator &get_initialized_allocator(SubDeviceId sub_device_id) const;
    Allocator &get_initialized_allocator(SubDeviceId sub_device_id);

    std::unique_ptr<Allocator> &sub_device_allocator(SubDeviceId sub_device_id);

    std::shared_ptr<TraceBuffer> &create_trace(uint32_t tid);
    void release_trace(uint32_t tid);
    std::shared_ptr<TraceBuffer> get_trace(uint32_t tid);

    uint8_t num_sub_devices() const;
    bool has_allocations() const;
    DeviceAddr local_l1_size() const;

    // friend class tt::tt_metal::Device;

   private:
    void validate_sub_devices() const;
    uint8_t get_sub_device_index(SubDeviceId sub_device_id) const;
    void populate_num_cores();
    void populate_sub_allocators();
    void populate_noc_data();

    // TODO: We have a max number of sub-devices, so we can use a fixed size array
    std::vector<SubDevice> sub_devices_;
    Device *device_;

    DeviceAddr local_l1_size_;
    std::vector<Allocator *> local_allocators_;
    std::vector<std::unique_ptr<Allocator>> owned_sub_device_allocators_;

    std::array<uint32_t, NumHalProgrammableCoreTypes> num_cores_{};
    std::vector<vector_memcpy_aligned<uint32_t>> noc_mcast_data_;
    std::vector<vector_memcpy_aligned<uint32_t>> noc_unicast_data_;
    // Concatenation of noc_mcast_data_ and noc_unicast_data_
    // Useful for optimized copying of all coords when constructing FD commands
    std::vector<vector_memcpy_aligned<uint32_t>> noc_mcast_unicast_data_;

    std::unordered_map<uint32_t, std::shared_ptr<TraceBuffer>> trace_buffer_pool_;
};

}  // namespace detail

}  // namespace tt_metal
