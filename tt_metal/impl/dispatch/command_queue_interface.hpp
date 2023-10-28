/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/src/firmware/riscv/grayskull/noc/noc_overlay_parameters.h"

using namespace tt::tt_metal;

struct SystemMemoryCQWriteInterface {
    // Equation for fifo size is
    // | fifo_wr_ptr + command size B - fifo_rd_ptr |
    // Space available would just be fifo_limit - fifo_size
    const uint32_t fifo_size = ((1024 * 1024 * 1024) - 96) >> 4;
    const uint32_t fifo_limit = ((1024 * 1024 * 1024) >> 4) - 1;  // Last possible FIFO address

    uint32_t fifo_wr_ptr;
    bool fifo_wr_toggle;
};

uint32_t get_cq_rd_ptr(Device* device);

class SystemMemoryWriter {
   public:
    SystemMemoryCQWriteInterface cq_write_interface;
    SystemMemoryWriter();

    void cq_reserve_back(Device* device, uint32_t cmd_size_B);

    // Ideally, data should be an array or pointer, but vector for time-being
    void cq_write(Device* device, const uint32_t* data, uint32_t size, uint32_t write_ptr);

    void send_write_ptr(Device* device);
    void send_write_toggle(Device* device);

    void cq_push_back(Device* device, uint32_t push_size_B);
};
