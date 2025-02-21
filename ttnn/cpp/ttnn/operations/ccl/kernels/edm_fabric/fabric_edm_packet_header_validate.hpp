// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "debug/assert.h"

namespace tt::fabric {

FORCE_INLINE void validate(const PacketHeader& packet_header) {
    ASSERT(packet_header.chip_send_type <= CHIP_SEND_TYPE_LAST);
}
FORCE_INLINE bool is_valid(PacketHeader const& packet_header) {
    return (packet_header.chip_send_type <= CHIP_SEND_TYPE_LAST) && (packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

}  // namespace tt::fabric
