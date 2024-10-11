/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/cpp/ttnn/operation.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_test2_impl(
    const Tensor &input1,
    const Tensor &cond,
    const Tensor &input2,
    const Tensor &output,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config);

struct MorehTest2 {
    MemoryConfig output_mem_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("output_mem_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->output_mem_config), std::cref(this->compute_kernel_config));
    }
};

Tensor moreh_test2(
    const Tensor &input1,
    const Tensor &cond,
    const Tensor &input2,
    const std::optional<const Tensor> output = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
