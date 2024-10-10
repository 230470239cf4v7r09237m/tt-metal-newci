// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteBarrier {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const uint32_t num_samples,
        const uint32_t max_concurrent_samples,
        const uint32_t sample_page_size,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);
};
}  // namespace ttnn::operations::ccl end
}  // namespace ttnn::operations end
constexpr auto barrier =
    ttnn::register_operation<"ttnn::barrier", ttnn::operations::ccl::ExecuteBarrier>();
}  // namespace ttnn end
