// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

#include "rms_allgather_pybind.hpp"

namespace ttnn::operations::experimental::ccl {

namespace py = pybind11;

void bind_fused_rms_1_1_32_8192(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::fused_rms_1_1_32_8192,
        R"doc(Only works for sharded shape (1,1,32,8192) sharded on 1 core
        )doc",
        // Stats is internally computed
        ttnn::pybind_arguments_t{
            // Used by all
            py::arg("input_tensor"),
            py::arg("global_semaphore"),  // TODO: Build this internally
            py::kw_only(),
            py::arg("topology") = ttnn::ccl::Topology::Line,
            py::arg("dtype") = std::nullopt,  // Should default to BFLOAT 16 on pre, nullopt on post
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            // on pre only
            py::arg("residual_input_tensor") = std::nullopt,
            // on post only
            py::arg("epsilon") = 1e-12,  // constant 1e-12 on pre, value only affects post
            py::arg("weight") = std::nullopt,
            py::arg("bias") = std::nullopt,
        });
}
}  // namespace ttnn::operations::experimental::ccl
