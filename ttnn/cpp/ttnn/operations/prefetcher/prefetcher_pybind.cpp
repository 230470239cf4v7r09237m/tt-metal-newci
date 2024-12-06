// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefetcher_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "prefetcher/dram_prefetcher_pybind.hpp"

namespace ttnn::operations::prefetcher {

namespace py = pybind11;

void py_module(py::module& module) { py_bind_dram_prefetcher(module); }

}  // namespace ttnn::operations::prefetcher
