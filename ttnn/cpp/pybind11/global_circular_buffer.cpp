// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_circular_buffer.hpp"

#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "ttnn/cpp/ttnn/global_circular_buffer.hpp"
#include "pybind11/pybind11.h"

namespace ttnn::global_circular_buffer {

void py_module_types(py::module& module) {
    py::class_<GlobalCircularBuffer, std::shared_ptr<GlobalCircularBuffer>>(module, "global_circular_buffer");
    py::class_<MultiDeviceGlobalCircularBuffer>(module, "multi_device_global_circular_buffer");
}

void py_module(py::module& module) {
    // Single Device APIs
    module.def(
        "create_global_circular_buffer",
        py::overload_cast<Device*, const std::unordered_map<CoreCoord, CoreRangeSet>&, uint32_t, BufferType>(
            &create_global_circular_buffer),
        py::arg("device"),
        py::arg("sender_receiver_core_mapping"),
        py::arg("size"),
        py::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a GlobalCircularBuffer Object on a single device.

            Args:
                device (Device): The device on which to create the global circular buffer.
                sender_receiver_core_mapping (dict): The mapping of remote sender to remote receiver cores for the circular buffer.
                size (int): Size of the global circular buffer per core in bytes.
                buffer_type (BufferType): The type of buffer to use for the global circular buffer.
            )doc");

    // Multi Device APIs
    module.def(
        "create_global_circular_buffer",
        py::overload_cast<MeshDevice*, const std::unordered_map<CoreCoord, CoreRangeSet>&, uint32_t, BufferType>(
            &create_global_circular_buffer),
        py::arg("mesh_device"),
        py::arg("sender_receiver_core_mapping"),
        py::arg("size"),
        py::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a GlobalCircularBuffer Object on a single device.

            Args:
                mesh_device (MeshDevice): The mesh device on which to create the global circular buffer.
                sender_receiver_core_mapping (dict): The mapping of remote sender to remote receiver cores for the circular buffer.
                size (int): Size of the global circular buffer per core in bytes.
                buffer_type (BufferType): The type of buffer to use for the global circular buffer.
            )doc");
}

}  // namespace ttnn::global_circular_buffer
