// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#pragma once

namespace ttnn {
namespace ccl {

enum class OpFabricMode {
    PERSISTENT_EDM,

    // Dispatch/launch the EDMs per op. They do not live past
    // the lifetime of the op
    TEMPORARY_EDM
};

enum class OpBuildMode {
    NON_PERSISTENT,
    PERSISTENT_BUILD_PERSISTENT_EDM,
    PERSISTENT_BUILD_NON_PERSISTENT_WORKERS
};

} // namespace ccl
} // namespace ttnn
