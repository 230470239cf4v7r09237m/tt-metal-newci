// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

struct TestBufferConfig {
    uint32_t num_pages;
    uint32_t page_size;
    tt::tt_metal::BufferType buftype;
};

inline pair<tt::tt_metal::Buffer, std::vector<uint32_t>> EnqueueWriteBuffer_prior_to_wrap(tt::tt_metal::Device* device, tt::tt_metal::CommandQueue& cq, const TestBufferConfig& config) {
    // This function just enqueues a buffer (which should be large in the config)
    // write as a precursor to testing the wrap mechanism
    size_t buf_size = config.num_pages * config.page_size;
    tt::tt_metal::Buffer buffer(device, buf_size, config.page_size, config.buftype);

    std::vector<uint32_t> src = create_random_vector_of_bfloat16(
      buf_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    EnqueueWriteBuffer(cq, buffer, src, false);
    return std::make_pair(std::move(buffer), src);
}
