// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_op.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace operations {
namespace matmul {

namespace detail {

bool is_input_batched(const ttnn::Shape& shape);

}  // namespace detail

std::optional<UnaryWithParam> get_fused_activation(const std::optional<const std::string>& activation);

ttnn::Tensor bound_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const struct Matmul& parameters,
    const uint8_t& queue_id);

struct MatmulOperation {
        static Tensor invoke(const Tensor& input_tensor_a,
           const Tensor& input_tensor_b,
           const bool transpose_a,
           const bool transpose_b,
           const MemoryConfig& memory_config,
           const std::optional<const DataType> dtype,
           const std::optional<const MatmulProgramConfig> program_config,
           const std::optional<const std::string>& activation,
           const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
           const std::optional<const CoreGrid> core_grid);
};

struct LinearOperation {
        static Tensor invoke(const Tensor& input_tensor_a,
           const Tensor& input_tensor_b,
           const std::optional<const Tensor>& bias,
           const bool transpose_a,
           const bool transpose_b,
           const MemoryConfig& memory_config,
           const std::optional<const DataType> dtype,
           const std::optional<const MatmulProgramConfig> program_config,
           const std::optional<const std::string>& activation,
           const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
           const std::optional<const CoreGrid> core_grid);
};

}  // namespace matmul
}  // namespace operations
    constexpr auto composite_matmul = ttnn::register_operation<"ttnn::matmul", operations::matmul::MatmulOperation>();
    constexpr auto composite_linear = ttnn::register_operation<"ttnn::linear", operations::matmul::LinearOperation>();
}  // namespace ttnn
