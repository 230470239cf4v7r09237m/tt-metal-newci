// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <strong_type.hpp>

#define MAX_PACK_UNTILIZE_WIDTH 8  // pack untilize currently does not support > 8 width

namespace ttnn {

/*
    We have two software command queues available to overlap some work and reduce latency.
    For example, Op2 can be prepared in a different queue while the first queue is blocked, waiting for data readout by
   Op1. TT-NN operations allow specifying which queue should be used. The default queue is 0, and the possible values
   are 0 and 1.
*/

using QueueId = tt::stl::StrongType<uint8_t, struct QueueIdTag>;
static const QueueId DefaultQueueId = QueueId(0);

}  // namespace ttnn

namespace tt::tt_metal {
using QueueId = ttnn::QueueId;
}
