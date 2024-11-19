// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(ARCH_BLACKHOLE)

#define COMPILE_FOR_ERISC

#include "llrt/hal.hpp"
#include "llrt/hal_asserts.hpp"
#include "llrt/blackhole/bh_hal.hpp"
#include "hw/inc/blackhole/core_config.h"
#include "hw/inc/blackhole/dev_mem_map.h"
#include "hw/inc/blackhole/eth_l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"
#include "hw/inc/dev_msgs.h"

#include <magic_enum.hpp>

#define GET_IERISC_MAILBOX_ADDRESS_HOST(x) ((uint64_t) & (((mailboxes_t *)MEM_IERISC_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

HalCoreInfoType create_idle_eth_mem_map() {

    uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    static_assert(MEM_IERISC_MAP_END % L1_ALIGNMENT == 0);

    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::COUNT));
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::BASE)] = MEM_ETH_BASE;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::MAILBOX)] = MEM_IERISC_MAILBOX_BASE;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH)] = GET_IERISC_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::WATCHER)] = GET_IERISC_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::DPRINT)] = GET_IERISC_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::PROFILER)] = GET_IERISC_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_IERISC_MAP_END;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::UNRESERVED)] = ((MEM_IERISC_MAP_END + L1_KERNEL_CONFIG_SIZE - 1) | (max_alignment - 1)) + 1;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::CORE_INFO)] = GET_IERISC_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::GO_MSG)] = GET_IERISC_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = GET_IERISC_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::COUNT));
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::BASE)] = MEM_ETH_SIZE;
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::MAILBOX)] = MEM_IERISC_MAILBOX_SIZE;
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_SIZE; // TODO: this is wrong, need idle eth specific value
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::UNRESERVED)] = MEM_ETH_SIZE - mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::UNRESERVED)];
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(NumEthDispatchClasses);
    std::vector<HalJitBuildConfig> processor_types(1);
    for (uint8_t processor_class_idx = 0; processor_class_idx < NumEthDispatchClasses; processor_class_idx++) {
        DeviceAddr fw_base, local_init;
        switch (processor_class_idx) {
            case 0: {
                fw_base = MEM_IERISC_FIRMWARE_BASE;
                local_init = MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH;
            }
            break;
            case 1: {
                fw_base = MEM_SLAVE_IERISC_FIRMWARE_BASE;
                local_init = MEM_SLAVE_IERISC_INIT_LOCAL_L1_BASE_SCRATCH;
            }
            break;
            default:
                TT_THROW("Unexpected processor class {} for Blackhole Idle Ethernet", processor_class_idx);
        }
        processor_types[0] = HalJitBuildConfig{
            .fw_base_addr = fw_base,
            .local_init_addr = local_init
        };
        processor_classes[processor_class_idx] = processor_types;
    }

    return {HalProgrammableCoreType::IDLE_ETH, CoreType::ETH, processor_classes, mem_map_bases, mem_map_sizes, false};
}

}  // namespace tt_metal
}  // namespace tt

#endif
