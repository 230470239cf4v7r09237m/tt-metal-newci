// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <array>
#include <tuple>
#include <vector>

#include "command_queue_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/buffers/global_semaphore.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/event/event.hpp"
#include "tt_metal/impl/sub_device/sub_device.hpp"
#include "tests/tt_metal/test_utils/stimulus.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/command_queue/command_queue_test_utils.hpp"

using namespace tt::tt_metal;

namespace basic_tests {

std::tuple<Program, CoreCoord, std::unique_ptr<GlobalSemaphore>> create_single_sync_program(Device *device, SubDevice sub_device) {
    auto syncer_coord = sub_device.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto global_sem = CreateGlobalSemaphore(device, sub_device.cores(HalProgrammableCoreType::TENSIX), INVALID);

    Program syncer_program = CreateProgram();
    auto syncer_kernel = CreateKernel(
        syncer_program,
        "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/sub_device/kernels/syncer.cpp",
        syncer_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 1> syncer_rt_args = {global_sem->address()};
    SetRuntimeArgs(syncer_program, syncer_kernel, syncer_core, syncer_rt_args);
    return {std::move(syncer_program), std::move(syncer_coord), std::move(global_sem)};
}

std::tuple<Program, Program, Program, std::unique_ptr<GlobalSemaphore>> create_basic_sync_program(Device *device, const SubDevice& sub_device_1, const SubDevice& sub_device_2) {
    auto waiter_coord = sub_device_2.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto waiter_core = CoreRangeSet(CoreRange(waiter_coord, waiter_coord));
    auto waiter_core_physical = device->worker_core_from_logical_core(waiter_coord);
    auto incrementer_cores = sub_device_1.cores(HalProgrammableCoreType::TENSIX);
    auto syncer_coord = incrementer_cores.ranges().back().end_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto syncer_core_physical = device->worker_core_from_logical_core(syncer_coord);
    auto all_cores = waiter_core.merge(incrementer_cores).merge(syncer_core);
    auto global_sem = CreateGlobalSemaphore(device, all_cores, INVALID);

    Program waiter_program = CreateProgram();
    auto waiter_kernel = CreateKernel(
        waiter_program,
        "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/sub_device/kernels/persistent_waiter.cpp",
        waiter_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 4> waiter_rt_args = {global_sem->address(), incrementer_cores.num_cores(), syncer_core_physical.x, syncer_core_physical.y};
    SetRuntimeArgs(waiter_program, waiter_kernel, waiter_core, waiter_rt_args);

    Program syncer_program = CreateProgram();
    auto syncer_kernel = CreateKernel(
        syncer_program,
        "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/sub_device/kernels/syncer.cpp",
        syncer_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 1> syncer_rt_args = {global_sem->address()};
    SetRuntimeArgs(syncer_program, syncer_kernel, syncer_core, syncer_rt_args);

    Program incrementer_program = CreateProgram();
    auto incrementer_kernel = CreateKernel(
        incrementer_program,
        "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/sub_device/kernels/incrementer.cpp",
        incrementer_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default});
    std::array<uint32_t, 3> incrementer_rt_args = {global_sem->address(), waiter_core_physical.x, waiter_core_physical.y};
    SetRuntimeArgs(incrementer_program, incrementer_kernel, incrementer_cores, incrementer_rt_args);
    return {std::move(waiter_program), std::move(syncer_program), std::move(incrementer_program), std::move(global_sem)};
}

std::tuple<Program, Program, Program, std::unique_ptr<GlobalSemaphore>> create_basic_eth_sync_program(Device *device, const SubDevice& sub_device_1, const SubDevice& sub_device_2) {
    auto waiter_coord = sub_device_2.cores(HalProgrammableCoreType::ACTIVE_ETH).ranges().at(0).start_coord;
    auto waiter_core = CoreRangeSet(CoreRange(waiter_coord, waiter_coord));
    auto waiter_core_physical = device->ethernet_core_from_logical_core(waiter_coord);
    auto tensix_waiter_coord = sub_device_2.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto tensix_waiter_core = CoreRangeSet(CoreRange(tensix_waiter_coord, tensix_waiter_coord));
    auto tensix_waiter_core_physical = device->worker_core_from_logical_core(tensix_waiter_coord);
    auto incrementer_cores = sub_device_1.cores(HalProgrammableCoreType::TENSIX);
    auto syncer_coord = incrementer_cores.ranges().back().end_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto syncer_core_physical = device->worker_core_from_logical_core(syncer_coord);
    auto all_cores = tensix_waiter_core.merge(incrementer_cores).merge(syncer_core);
    auto global_sem = CreateGlobalSemaphore(device, all_cores, INVALID);

    Program waiter_program = CreateProgram();
    auto waiter_kernel = CreateKernel(
        waiter_program,
        "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/sub_device/kernels/persistent_remote_waiter.cpp",
        waiter_core,
        EthernetConfig{
            .noc = NOC::RISCV_0_default,
            .processor = DataMovementProcessor::RISCV_0});
    std::array<uint32_t, 7> waiter_rt_args = {global_sem->address(), incrementer_cores.num_cores(), syncer_core_physical.x, syncer_core_physical.y, tensix_waiter_core_physical.x, tensix_waiter_core_physical.y, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE};
    SetRuntimeArgs(waiter_program, waiter_kernel, waiter_core, waiter_rt_args);

    Program syncer_program = CreateProgram();
    auto syncer_kernel = CreateKernel(
        syncer_program,
        "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/sub_device/kernels/syncer.cpp",
        syncer_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 1> syncer_rt_args = {global_sem->address()};
    SetRuntimeArgs(syncer_program, syncer_kernel, syncer_core, syncer_rt_args);

    Program incrementer_program = CreateProgram();
    auto incrementer_kernel = CreateKernel(
        incrementer_program,
        "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/sub_device/kernels/incrementer.cpp",
        incrementer_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default});
    std::array<uint32_t, 3> incrementer_rt_args = {global_sem->address(), tensix_waiter_core_physical.x, tensix_waiter_core_physical.y};
    SetRuntimeArgs(incrementer_program, incrementer_kernel, incrementer_cores, incrementer_rt_args);
    return {std::move(waiter_program), std::move(syncer_program), std::move(incrementer_program), std::move(global_sem)};
}

TEST_F(CommandQueueSingleCardFixture, TestSubDeviceAllocations) {
    uint32_t local_l1_size = 3200;
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});
    CoreRangeSet sharded_cores_2 = CoreRange({4, 4}, {4, 4});

    auto sharded_cores_1_vec = corerange_to_cores(sharded_cores_1, std::nullopt, true);
    auto sharded_cores_2_vec = corerange_to_cores(sharded_cores_2, std::nullopt, true);

    ShardSpecBuffer shard_spec_buffer_1 = ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, false, {1, 1}, {sharded_cores_1.num_cores(), 1});
    uint32_t page_size_1 = 32;
    ShardedBufferConfig shard_config_1 = {nullptr, sharded_cores_1.num_cores() * page_size_1, page_size_1, BufferType::L1, TensorMemoryLayout::HEIGHT_SHARDED, shard_spec_buffer_1};
    auto input_1 = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, shard_config_1.size / sizeof(uint32_t));

    ShardSpecBuffer shard_spec_buffer_2 = ShardSpecBuffer(sharded_cores_2, {1, 1}, ShardOrientation::ROW_MAJOR, false, {1, 1}, {sharded_cores_2.num_cores(), 1});
    uint32_t page_size_2 = 64;
    ShardedBufferConfig shard_config_2 = {nullptr, sharded_cores_2.num_cores() * page_size_2, page_size_2, BufferType::L1, TensorMemoryLayout::HEIGHT_SHARDED, shard_spec_buffer_2};
    auto input_2 = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, shard_config_2.size / sizeof(uint32_t));

    uint32_t page_size_3 = 1024;
    InterleavedBufferConfig interleaved_config = {nullptr, page_size_3, page_size_3, BufferType::L1, TensorMemoryLayout::INTERLEAVED};
    auto input_3 = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, interleaved_config.size / sizeof(uint32_t));

    for (Device *device : devices_) {
        auto sub_device_manager_1 = device->create_sub_device_manager({sub_device_1}, local_l1_size);
        auto sub_device_manager_2 = device->create_sub_device_manager({sub_device_1, sub_device_2}, local_l1_size);
        DeviceAddr l1_unreserved_base = device->get_base_allocator_addr(HalMemType::L1);
        DeviceAddr max_addr = l1_unreserved_base + local_l1_size;

        shard_config_1.device = device;
        shard_config_2.device = device;
        interleaved_config.device = device;

        std::vector<CoreCoord> physical_cores_1;
        physical_cores_1.reserve(sharded_cores_1_vec.size());
        for (const auto& core : sharded_cores_1_vec) {
            physical_cores_1.push_back(device->worker_core_from_logical_core(core));
        }

        std::vector<CoreCoord> physical_cores_2;
        physical_cores_2.reserve(sharded_cores_2_vec.size());
        for (const auto& core : sharded_cores_2_vec) {
            physical_cores_2.push_back(device->worker_core_from_logical_core(core));
        }

        device->load_sub_device_manager(sub_device_manager_1);

        auto buffer_1 = CreateBuffer(shard_config_1, SubDeviceId{0});
        EXPECT_EQ(buffer_1->address(), max_addr - page_size_1);
        EnqueueWriteBuffer(device->command_queue(), buffer_1, input_1, false);
        std::vector<uint32_t> output_1;
        EnqueueReadBuffer(device->command_queue(), buffer_1, output_1, true);
        EXPECT_EQ(input_1, output_1);
        auto input_1_it = input_1.begin();
        for (const auto& physical_core : physical_cores_1) {
            auto readback = tt::llrt::read_hex_vec_from_core(
                device->id(), physical_core, buffer_1->address(), page_size_1);
            EXPECT_TRUE(std::equal(input_1_it, input_1_it + page_size_1 / sizeof(uint32_t), readback.begin()));
            input_1_it += page_size_1 / sizeof(uint32_t);
        }

        auto buffer_2 = CreateBuffer(interleaved_config);
        EXPECT_THROW(CreateBuffer(shard_config_1, SubDeviceId{1}), std::exception);
        EXPECT_THROW(device->reset_active_sub_device_manager(), std::exception);
        EXPECT_THROW(device->load_sub_device_manager(sub_device_manager_2), std::exception);
        DeallocateBuffer(*buffer_1);
        device->reset_active_sub_device_manager();
        device->load_sub_device_manager(sub_device_manager_2);

        auto buffer_3 = CreateBuffer(shard_config_2, SubDeviceId{1});
        EXPECT_EQ(buffer_3->address(), max_addr - page_size_2);
        EnqueueWriteBuffer(device->command_queue(), buffer_3, input_2, false);
        std::vector<uint32_t> output_2;
        EnqueueReadBuffer(device->command_queue(), buffer_3, output_2, true);
        EXPECT_EQ(input_2, output_2);
        auto input_2_it = input_2.begin();
        for (const auto& physical_core : physical_cores_2) {
            auto readback = tt::llrt::read_hex_vec_from_core(
                device->id(), physical_core, buffer_3->address(), page_size_2);
            EXPECT_TRUE(std::equal(input_2_it, input_2_it + page_size_2 / sizeof(uint32_t), readback.begin()));
            input_2_it += page_size_2 / sizeof(uint32_t);
        }

        auto buffer_4 = CreateBuffer(shard_config_1,  SubDeviceId{0});
        EXPECT_EQ(buffer_4->address(), max_addr - page_size_1);
        EXPECT_THROW(CreateBuffer(interleaved_config, SubDeviceId{0}), std::exception);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestSubDeviceSynchronization) {
    uint32_t local_l1_size = 3200;
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});

    auto sharded_cores_1_vec = corerange_to_cores(sharded_cores_1, std::nullopt, true);

    ShardSpecBuffer shard_spec_buffer_1 = ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, false, {1, 1}, {sharded_cores_1.num_cores(), 1});
    uint32_t page_size_1 = 32;
    ShardedBufferConfig shard_config_1 = {nullptr, sharded_cores_1.num_cores() * page_size_1, page_size_1, BufferType::L1, TensorMemoryLayout::HEIGHT_SHARDED, shard_spec_buffer_1};
    auto input_1 = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, shard_config_1.size / sizeof(uint32_t));

    std::array sub_device_ids_to_block = {SubDeviceId{0}};
    for (Device *device : devices_) {
        auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, local_l1_size);

        shard_config_1.device = device;

        std::vector<CoreCoord> physical_cores_1;
        physical_cores_1.reserve(sharded_cores_1_vec.size());
        for (const auto& core : sharded_cores_1_vec) {
            physical_cores_1.push_back(device->worker_core_from_logical_core(core));
        }

        device->load_sub_device_manager(sub_device_manager);

        auto [program, syncer_core, global_semaphore] = create_single_sync_program(device, sub_device_2);
        EnqueueProgram(device->command_queue(), program, false);

        auto buffer_1 = CreateBuffer(shard_config_1, sub_device_ids_to_block[0]);

        // Test blocking synchronize doesn't stall
        Synchronize(device, 0, sub_device_ids_to_block);

        // Test blocking write buffer doesn't stall
        EnqueueWriteBuffer(device->command_queue(), buffer_1, input_1, true, sub_device_ids_to_block);

        // Test record event won't cause a stall
        auto event = std::make_shared<Event>();
        EnqueueRecordEvent(device->command_queue(), event, sub_device_ids_to_block);
        Synchronize(device, 0, sub_device_ids_to_block);

        // Test blocking read buffer doesn't stall
        std::vector<uint32_t> output_1;
        EnqueueReadBuffer(device->command_queue(), buffer_1, output_1, true, sub_device_ids_to_block);
        EXPECT_EQ(input_1, output_1);
        auto input_1_it = input_1.begin();
        for (const auto& physical_core : physical_cores_1) {
            auto readback = tt::llrt::read_hex_vec_from_core(
                device->id(), physical_core, buffer_1->address(), page_size_1);
            EXPECT_TRUE(std::equal(input_1_it, input_1_it + page_size_1 / sizeof(uint32_t), readback.begin()));
            input_1_it += page_size_1 / sizeof(uint32_t);
        }
        auto sem_addr = global_semaphore->address();
        auto physical_syncer_core = device->worker_core_from_logical_core(syncer_core);
        tt::llrt::write_hex_vec_to_core(device->id(), physical_syncer_core, std::vector<uint32_t>{1}, sem_addr);

        // Full synchronization
        Synchronize(device);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestSubDeviceBasicPrograms) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    uint32_t num_iters = 5;
    for (Device *device : devices_) {
        auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
        device->load_sub_device_manager(sub_device_manager);

        auto [waiter_program, syncer_program, incrementer_program, global_sem] = create_basic_sync_program(device, sub_device_1, sub_device_2);

        for (uint32_t i = 0; i < num_iters; i++) {
            EnqueueProgram(device->command_queue(), waiter_program, false);
            // Test blocking on one sub-device
            EnqueueProgram(device->command_queue(), syncer_program, true);
            EnqueueProgram(device->command_queue(), incrementer_program, false);
        }
        Synchronize(device);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestSubDeviceBasicEthPrograms) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    uint32_t num_iters = 5;
    for (Device *device : devices_) {
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "Skipping test because device " << device->id() << " does not have any active ethernet cores";
        }
        auto eth_core = *device->get_active_ethernet_cores(true).begin();
        SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})}), CoreRangeSet(CoreRange(eth_core, eth_core))});
        auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
        device->load_sub_device_manager(sub_device_manager);

        auto [waiter_program, syncer_program, incrementer_program, global_sem] = create_basic_eth_sync_program(device, sub_device_1, sub_device_2);

        for (uint32_t i = 0; i < num_iters; i++) {
            EnqueueProgram(device->command_queue(), waiter_program, false);
            // Test blocking on one sub-device
            EnqueueProgram(device->command_queue(), syncer_program, true);
            EnqueueProgram(device->command_queue(), incrementer_program, false);
        }
        Synchronize(device);
    }
}

TEST_F(CommandQueueSingleCardTraceFixture, TestSubDeviceTraceBasicPrograms) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    uint32_t num_iters = 5;
    for (Device *device : devices_) {
        auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
        device->load_sub_device_manager(sub_device_manager);

        auto [waiter_program, syncer_program, incrementer_program, global_sem] = create_basic_sync_program(device, sub_device_1, sub_device_2);

        // Compile the programs
        EnqueueProgram(device->command_queue(), waiter_program, false);
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        Synchronize(device);

        // Capture the trace
        auto tid_1 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), waiter_program, false);
        EnqueueProgram(device->command_queue(), syncer_program, false);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        EndTraceCapture(device, device->command_queue().id(), tid_1);

        auto tid_2 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), syncer_program, false);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        EndTraceCapture(device, device->command_queue().id(), tid_2);

        for (uint32_t i = 0; i < num_iters; i++) {
            // Regular program execution
            EnqueueProgram(device->command_queue(), waiter_program, false);
            // Test blocking on one sub-device
            EnqueueProgram(device->command_queue(), syncer_program, true);
            EnqueueProgram(device->command_queue(), incrementer_program, false);

            // Full trace execution
            ReplayTrace(device, device->command_queue().id(), tid_1, false);

            // Partial trace execution
            EnqueueProgram(device->command_queue(), waiter_program, false);
            ReplayTrace(device, device->command_queue().id(), tid_2, false);
        }
        Synchronize(device);
    }
}

TEST_F(CommandQueueSingleCardTraceFixture, TestSubDeviceTraceBasicEthPrograms) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    uint32_t num_iters = 5;
    for (Device *device : devices_) {
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "Skipping test because device " << device->id() << " does not have any active ethernet cores";
        }
        auto eth_core = *device->get_active_ethernet_cores(true).begin();
        SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})}), CoreRangeSet(CoreRange(eth_core, eth_core))});
        auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
        device->load_sub_device_manager(sub_device_manager);

        auto [waiter_program, syncer_program, incrementer_program, global_sem] = create_basic_eth_sync_program(device, sub_device_1, sub_device_2);

        // Compile the programs
        EnqueueProgram(device->command_queue(), waiter_program, false);
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        Synchronize(device);

        // Capture the trace
        auto tid_1 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), waiter_program, false);
        EnqueueProgram(device->command_queue(), syncer_program, false);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        EndTraceCapture(device, device->command_queue().id(), tid_1);

        auto tid_2 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), syncer_program, false);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        EndTraceCapture(device, device->command_queue().id(), tid_2);

        for (uint32_t i = 0; i < num_iters; i++) {
            // Regular program execution
            EnqueueProgram(device->command_queue(), waiter_program, false);
            // Test blocking on one sub-device
            EnqueueProgram(device->command_queue(), syncer_program, true);
            EnqueueProgram(device->command_queue(), incrementer_program, false);

            // Full trace execution
            ReplayTrace(device, device->command_queue().id(), tid_1, false);

            // Partial trace execution
            EnqueueProgram(device->command_queue(), waiter_program, false);
            ReplayTrace(device, device->command_queue().id(), tid_2, false);
        }
        Synchronize(device);
    }
}

TEST_F(CommandQueueSingleCardTraceFixture, TestSubDeviceTraceProgramsReconfigureSubDevices) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::array{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    SubDevice sub_device_3(std::array{CoreRangeSet(std::array{CoreRange({2, 4}, {3, 4}), CoreRange({5, 1}, {6, 3})})});
    uint32_t num_iters = 5;
    for (Device *device : devices_) {
        if (!does_device_have_active_eth_cores(device)) {
            GTEST_SKIP() << "Skipping test because device " << device->id() << " does not have any active ethernet cores";
        }
        auto eth_core = *device->get_active_ethernet_cores(true).begin();
        SubDevice sub_device_4(std::array{CoreRangeSet(std::array{CoreRange({2, 1}, {2, 2}), CoreRange({1, 5}, {5, 5})}), CoreRangeSet(CoreRange(eth_core, eth_core))});

        auto sub_device_manager_1 = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
        auto sub_device_manager_2 = device->create_sub_device_manager({sub_device_3, sub_device_4}, 3200);

        device->load_sub_device_manager(sub_device_manager_1);

        auto [waiter_program_1, syncer_program_1, incrementer_program_1, global_sem_1] = create_basic_sync_program(device, sub_device_1, sub_device_2);

        // Compile the programs
        EnqueueProgram(device->command_queue(), waiter_program_1, false);
        EnqueueProgram(device->command_queue(), syncer_program_1, false);
        EnqueueProgram(device->command_queue(), incrementer_program_1, false);
        Synchronize(device);

        // Capture the trace
        auto tid_1 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), waiter_program_1, false);
        EnqueueProgram(device->command_queue(), syncer_program_1, false);
        EnqueueProgram(device->command_queue(), incrementer_program_1, false);
        EndTraceCapture(device, device->command_queue().id(), tid_1);

        auto tid_2 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), syncer_program_1, false);
        EnqueueProgram(device->command_queue(), incrementer_program_1, false);
        EndTraceCapture(device, device->command_queue().id(), tid_2);

        device->load_sub_device_manager(sub_device_manager_2);

        auto [waiter_program_2, syncer_program_2, incrementer_program_2, global_sem_2] = create_basic_eth_sync_program(device, sub_device_3, sub_device_4);

        // Compile the programs
        EnqueueProgram(device->command_queue(), waiter_program_2, false);
        EnqueueProgram(device->command_queue(), syncer_program_2, false);
        EnqueueProgram(device->command_queue(), incrementer_program_2, false);
        Synchronize(device);

        // Capture the trace
        auto tid_3 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), waiter_program_2, false);
        EnqueueProgram(device->command_queue(), syncer_program_2, false);
        EnqueueProgram(device->command_queue(), incrementer_program_2, false);
        EndTraceCapture(device, device->command_queue().id(), tid_3);

        auto tid_4 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), syncer_program_2, false);
        EnqueueProgram(device->command_queue(), incrementer_program_2, false);
        EndTraceCapture(device, device->command_queue().id(), tid_4);

        for (uint32_t i = 0; i < num_iters; i++) {
            device->load_sub_device_manager(sub_device_manager_1);
            // Regular program execution
            EnqueueProgram(device->command_queue(), waiter_program_1, false);
            // Test blocking on one sub-device
            EnqueueProgram(device->command_queue(), syncer_program_1, false);
            EnqueueProgram(device->command_queue(), incrementer_program_1, false);

            // Full trace execution
            ReplayTrace(device, device->command_queue().id(), tid_1, false);

            // Partial trace execution
            EnqueueProgram(device->command_queue(), waiter_program_1, false);
            ReplayTrace(device, device->command_queue().id(), tid_2, false);

            device->load_sub_device_manager(sub_device_manager_2);
            // Regular program execution
            EnqueueProgram(device->command_queue(), waiter_program_2, false);
            // Test blocking on one sub-device
            EnqueueProgram(device->command_queue(), syncer_program_2, false);
            EnqueueProgram(device->command_queue(), incrementer_program_2, false);

            // Full trace execution
            ReplayTrace(device, device->command_queue().id(), tid_3, false);

            // Partial trace execution
            EnqueueProgram(device->command_queue(), waiter_program_2, false);
            ReplayTrace(device, device->command_queue().id(), tid_4, false);
        }
        Synchronize(device);
    }
}

TEST_F(CommandQueueSingleCardTraceFixture, TestSubDeviceIllegalOperations) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});

    // Assert no idle eth cores specified
    EXPECT_THROW(SubDevice sub_device_3(std::array{CoreRangeSet(CoreRange({3, 3}, {3, 3})), CoreRangeSet(CoreRange({4, 4}, {4, 4})), CoreRangeSet(CoreRange({5, 5}, {5, 5}))}), std::exception);
    for (Device *device : devices_) {
        auto sub_device_manager_1 = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
        auto sub_device_manager_2 = device->create_sub_device_manager({sub_device_2, sub_device_1}, 3200);
        device->load_sub_device_manager(sub_device_manager_1);

        auto [waiter_program_1, syncer_program_1, incrementer_program_1, global_sem_1] = create_basic_sync_program(device, sub_device_1, sub_device_2);

        // Compile the programs
        EnqueueProgram(device->command_queue(), waiter_program_1, false);
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program_1, false);
        EnqueueProgram(device->command_queue(), incrementer_program_1, false);
        Synchronize(device);

        // Capture the trace
        auto tid_1 = BeginTraceCapture(device, device->command_queue().id());
        // Can not load a sub-device manager while tracing
        EXPECT_THROW(device->load_sub_device_manager(sub_device_manager_2), std::exception);
        EnqueueProgram(device->command_queue(), waiter_program_1, false);
        EnqueueProgram(device->command_queue(), syncer_program_1, false);
        EnqueueProgram(device->command_queue(), incrementer_program_1, false);
        EndTraceCapture(device, device->command_queue().id(), tid_1);

        device->load_sub_device_manager(sub_device_manager_2);
        auto [waiter_program_2, syncer_program_2, incrementer_program_2, global_sem_2] = create_basic_sync_program(device, sub_device_2, sub_device_1);

        EnqueueProgram(device->command_queue(), waiter_program_2, false);
        EnqueueProgram(device->command_queue(), syncer_program_2, false);
        EnqueueProgram(device->command_queue(), incrementer_program_2, false);
        Synchronize(device);

        auto tid_2 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), waiter_program_2, false);
        EnqueueProgram(device->command_queue(), syncer_program_2, false);
        EnqueueProgram(device->command_queue(), incrementer_program_2, false);
        EndTraceCapture(device, device->command_queue().id(), tid_2);

        // Regular program execution
        // Can not run a program on a different sub-device manager
        EXPECT_THROW(EnqueueProgram(device->command_queue(), waiter_program_1, false), std::exception);

        // Full trace execution
        ReplayTrace(device, device->command_queue().id(), tid_2, false);

        // Can not replay a trace on a different sub-device manager
        EXPECT_THROW(ReplayTrace(device, device->command_queue().id(), tid_1, false), std::exception);

        Synchronize(device);

        device->remove_sub_device_manager(sub_device_manager_1);
        EXPECT_THROW(device->load_sub_device_manager(sub_device_manager_1), std::exception);
    }
}

}  // namespace basic_tests
