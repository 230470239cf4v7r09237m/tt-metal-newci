// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "buffer_constants.hpp"

#include <magic_enum.hpp>

namespace tt {
namespace tt_metal {

std::ostream& operator<<(std::ostream& os, TensorMemoryLayout layout) {
    os << magic_enum::enum_name(layout);
    return os;
    // switch (layout) {
    //     case TensorMemoryLayout::INTERLEAVED: os << "INTERLEAVED"; break;
    //     case TensorMemoryLayout::SINGLE_BANK: os << "SINGLE_BANK"; break;
    //     case TensorMemoryLayout::HEIGHT_SHARDED: os << "HEIGHT_SHARDED"; break;
    //     case TensorMemoryLayout::WIDTH_SHARDED: os << "WIDTH_SHARDED"; break;
    //     case TensorMemoryLayout::BLOCK_SHARDED: os << "BLOCK_SHARDED"; break;
    //     default: os << "UNKNOWN";
    // }
    // return os;
}

std::ostream& operator<<(std::ostream& os, ShardOrientation orientation) {
    switch (orientation) {
        case ShardOrientation::ROW_MAJOR: os << "ROW_MAJOR"; break;
        case ShardOrientation::COL_MAJOR: os << "COL_MAJOR"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, BufferType buffer) {
    switch (buffer) {
        case BufferType::DRAM: os << "DRAM"; break;
        case BufferType::L1: os << "L1"; break;
        case BufferType::SYSTEM_MEMORY: os << "SYSTEM_MEMORY"; break;
        case BufferType::L1_SMALL: os << "L1_SMALL"; break;
        case BufferType::TRACE: os << "TRACE"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

} // namespace tt_metal
} // namespace tt
