// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/core_descriptor.hpp>  // For chip_id_t
#include <tt-metalium/hal_types.hpp>        // For HalProgrammableCoreType
#include <tt-metalium/dev_msgs.h>           // For go_msg_t
#include <tt-metalium/allocator_types.hpp>  // For BankMapping
#include <llrt/tt_cluster.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <impl/dispatch/dispatch_query_manager.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tt::tt_metal {

class MetalContext {
public:
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext& operator=(MetalContext&& other) noexcept = delete;
    MetalContext(const MetalContext&) = delete;
    MetalContext(MetalContext&& other) noexcept = delete;
    static MetalContext& instance();

    static Cluster& get_cluster();
    static dispatch_core_manager& get_dispatch_core_manager();
    static DispatchQueryManager& get_dispatch_query_manager();

    void initialize(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, BankMapping l1_bank_remap);

private:
    MetalContext();
    ~MetalContext();

    bool initialized_ = false;

    uint8_t num_hw_cqs_;
    BankMapping l1_bank_remap_;
    DispatchCoreConfig dispatch_core_config_;

    Cluster cluster_;
    std::unique_ptr<dispatch_core_manager> dispatch_core_manager_;
    std::unique_ptr<DispatchQueryManager> dispatch_query_manager_;

    static MetalContext* _inst;
};

}  // namespace tt::tt_metal
