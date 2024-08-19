/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

Tensor moreh_bmm(
    const Tensor& input,
    const Tensor& mat2,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
