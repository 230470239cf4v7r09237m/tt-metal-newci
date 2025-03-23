#include "gtest/gtest.h"
#include "ttnn/device.hpp"
#include <vector>
#include <utility>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <stdlib.h>
#include <map>

#include <tt-metalium/logger.hpp>
#include "ttnn_test_fixtures.hpp"

#include "tools/profiler/op_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/functions.hpp"
#include "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp"

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/trace.hpp"

int get_device_freq() { return 1e9; }

std::vector<std::pair<int, int>> SUBBLOCK_HW_CHOICES = {{4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2},
                                                        {2, 3}, {6, 1}, {1, 6}, {5, 1}, {1, 5}, {2, 2}, {4, 1},
                                                        {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1}};

std::pair<int, int> get_subblock_sizes(
    int m_tiles_per_core, int n_tiles_per_core, bool out_sharded = false, bool fp32_dest_acc_en = false) {
    for (const auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
        int out_subblock_h = subblock_hw.first;
        int out_subblock_w = subblock_hw.second;

        if (fp32_dest_acc_en) {
            if ((out_subblock_h * out_subblock_w) > 4) {
                continue;
            }
        }

        if (out_sharded) {
            if (n_tiles_per_core % out_subblock_w != 0 || out_subblock_h != 1) {
                continue;
            }
        }

        if (m_tiles_per_core % out_subblock_h == 0 && n_tiles_per_core % out_subblock_w == 0) {
            return {out_subblock_h, out_subblock_w};
        }
    }

    return {1, 1};
}

class Matmul2DHostPerfTestFixture : public ttnn::distributed::test::TTNNFixtureWithTraceEnabledDevice,
                                    public testing::WithParamInterface<std::tuple<
                                        /* grid_size */ std::tuple<int, int>,
                                        /* tile_h */ int,
                                        /* tile_w */ int,
                                        /* num_warmup_iterations */ int,
                                        /* num_measurement_iterations */ int,
                                        /* use_program_cache */ bool>> {
public:
    Matmul2DHostPerfTestFixture() : ttnn::distributed::test::TTNNFixtureWithTraceEnabledDevice(24576, 200000) {}
};

TEST_P(Matmul2DHostPerfTestFixture, Matmul2DHostPerfTest) {
    const std::tuple<int, int>& grid_size = std::get<0>(GetParam());
    const int& tile_h = std::get<1>(GetParam());
    const int& tile_w = std::get<2>(GetParam());
    const int& num_measurement_iterations = std::get<4>(GetParam());
    const bool& use_program_cache = std::get<5>(GetParam());

    TT_FATAL(std::get<0>(grid_size) > 0 && std::get<1>(grid_size) > 0, "Invalid grid size");
    TT_ASSERT(num_measurement_iterations > 0, "Won't have data without at least one measurement iteration");
    tt::tt_metal::IDevice* device = &getDevice();

    std::map<int, DataType> i_to_dt{{0, DataType::BFLOAT4_B}, {1, DataType::BFLOAT8_B}, {2, DataType::BFLOAT16}};
    std::map<int, MathFidelity> i_to_fi{{0, MathFidelity::LoFi}, {1, MathFidelity::HiFi2}, {2, MathFidelity::HiFi4}};

    // SET START ID DEPENDING ON CORE GRID TO GET HANG.
    // 1X1  start_id = 88;
    // 2X1  start_id = 4;
    // 1X2  start_id = 75;
    //
    // 1X1 w/Milos's change start_id = 145;
    // 2X1 w/Milos's change start_id = 145
    // 1X2 w/Milos's change start_id = 145
    //
    int start_id = 145;
    int test_id = 0;
    for (int dt_a = 0; dt_a < 3; dt_a++) {
        for (int m_a = 1; m_a <= 6; m_a++) {
            for (int k_a = 1; k_a <= 6; k_a++) {
                for (int n_a = 1; n_a <= 6; n_a++) {
                    for (int dt_b = 0; dt_b < 3; dt_b++) {
                        for (int m_b = 1; m_b <= 6; m_b++) {
                            for (int k_b = 1; k_b <= 6; k_b++) {
                                for (int n_b = 1; n_b <= 6; n_b++) {
                                    for (int shard_flip = 0; shard_flip < 2; shard_flip++) {
                                        if (test_id < start_id) {
                                            test_id++;
                                            continue;
                                        }

                                        std::cout << "Test = " << test_id << std::endl;

                                        // Resetting within c code hangs the system.
                                        // system("/home/software/syseng/bh/tt-smi -r 0");
                                        std::vector<
                                            std::
                                                tuple<DataType, MathFidelity, int, int, int, bool, bool, int, int, int>>
                                            configs;
                                        configs.push_back(std::make_tuple(
                                            i_to_dt[dt_a],
                                            i_to_fi[dt_a],
                                            (64 * m_a),
                                            (64 * k_a),
                                            (64 * n_a),
                                            !shard_flip,
                                            !shard_flip,
                                            2,
                                            2,
                                            2));
                                        configs.push_back(std::make_tuple(
                                            i_to_dt[dt_b],
                                            i_to_fi[dt_b],
                                            (64 * m_b),
                                            (64 * k_b),
                                            (64 * n_b),
                                            shard_flip,
                                            shard_flip,
                                            2,
                                            2,
                                            2));
                                        for (auto& config : configs) {
                                            DataType dtype = std::get<0>(config);
                                            MathFidelity math_fidelity = std::get<1>(config);
                                            int m = std::get<2>(config) * std::get<1>(grid_size);
                                            int k = std::get<3>(config) * std::get<0>(grid_size);
                                            int n = std::get<4>(config) * std::get<0>(grid_size);
                                            bool in0_sharded = std::get<5>(config);
                                            bool out_sharded = std::get<6>(config);
                                            const int in0_block_w_div = std::get<7>(config);
                                            const int num_out_blocks_h = std::get<8>(config);
                                            const int num_out_blocks_w = std::get<9>(config);

                                            tt::log_info(
                                                "Running test with dtype: {}, math_fidelity: {}", dtype, math_fidelity);

                                            const std::vector<int64_t> in0_shape = {1, 1, m, k};
                                            const std::vector<int64_t> in1_shape = {1, 1, k, n};
                                            const int in0_block_w = k / std::get<0>(grid_size) / 32 / in0_block_w_div;
                                            const int per_core_M = m / std::get<1>(grid_size) / tile_h;
                                            const int per_core_N = n / std::get<0>(grid_size) / tile_w;
                                            const int out_block_h = per_core_M / num_out_blocks_h;
                                            const int out_block_w = per_core_N / num_out_blocks_w;
                                            const auto [out_subblock_h, out_subblock_w] =
                                                get_subblock_sizes(out_block_h, out_block_w, out_sharded);

                                            tt::log_info(
                                                "M*K*N = {}*{}*{} out_subblock_h: {}, out_subblock_w: {}",
                                                m,
                                                k,
                                                n,
                                                out_subblock_h,
                                                out_subblock_w);

                                            std::string in0_storage_type = in0_sharded ? "L1" : "DRAM";
                                            std::string in1_storage_type = "DRAM";
                                            std::string out_storage_type = out_sharded ? "L1" : "DRAM";

                                            const ttnn::MemoryConfig in0_memory_config =
                                                in0_sharded
                                                    ? ttnn::operations::data_movement::create_sharded_memory_config(
                                                          ttnn::Shape{1, 1, m, k},
                                                          ttnn::CoreRangeSet(ttnn::CoreRange(
                                                              CoreCoord(0, 0),
                                                              ttnn::CoreCoord(
                                                                  std::get<0>(grid_size) - 1,
                                                                  std::get<1>(grid_size) - 1))),
                                                          ttnn::operations::data_movement::ShardStrategy::BLOCK,
                                                          tt::tt_metal::ShardOrientation::ROW_MAJOR)
                                                    : ttnn::DRAM_MEMORY_CONFIG;

                                            // In0 is all ones
                                            const std::vector<float> in0_data(m * k, 1.0f);
                                            ttnn::Tensor in0_t = Tensor::from_vector(
                                                in0_data,
                                                ttnn::TensorSpec(
                                                    ttnn::Shape({m, k}),
                                                    tt::tt_metal::TensorLayout(
                                                        dtype, tt::tt_metal::Layout::TILE, in0_memory_config)),
                                                device);

                                            // In1 is random data
                                            std::vector<float> in1_data(k * n);
                                            std::generate(in1_data.begin(), in1_data.end(), []() {
                                                float value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                                                return std::round(value * 15.0f) / 15.0f;
                                            });

                                            ttnn::Tensor in1_t = Tensor::from_vector(
                                                in1_data,
                                                ttnn::TensorSpec(
                                                    ttnn::Shape({k, n}),
                                                    tt::tt_metal::TensorLayout(
                                                        dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
                                                device);

                                            ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig
                                                program_config{
                                                    /* compute_with_storage_grid_size */ {
                                                        std::get<0>(grid_size), std::get<1>(grid_size)},
                                                    /* in0_block_w */ in0_block_w,
                                                    /* out_subblock_h */ out_subblock_h,
                                                    /* out_subblock_w */ out_subblock_w,
                                                    /* out_block_h */ out_block_h,
                                                    /* out_block_w */ out_block_w,
                                                    /* per_core_M */ per_core_M,
                                                    /* per_core_N */ per_core_N,
                                                    /* transpose_mcast */ false,
                                                    /* fused_activation */ std::nullopt};

                                            const ttnn::WormholeComputeKernelConfig compute_kernel_config =
                                                ttnn::WormholeComputeKernelConfig{math_fidelity, true, false, true};

                                            const ttnn::MemoryConfig out_mem_config =
                                                out_sharded
                                                    ? ttnn::
                                                          MemoryConfig{ttnn::TensorMemoryLayout::BLOCK_SHARDED, ttnn::BufferType::L1}
                                                    : ttnn::DRAM_MEMORY_CONFIG;

                                            const Tile output_tile =
                                                out_sharded ? (tile_h <= 16 ? tt::tt_metal::Tile({tile_h, 32})
                                                                            : tt::tt_metal::Tile({tile_h, tile_w}))
                                                            : tt::tt_metal::Tile({tile_h, tile_w});

                                            const ttnn::operations::matmul::Matmul matmul_params(
                                                program_config,
                                                /*bcast_batch*/ std::nullopt,
                                                /* output_mem_config */ out_mem_config,
                                                /* output_dtype */ dtype,
                                                /* compute_kernel_config */ compute_kernel_config,
                                                /* untilize_out */ false,
                                                /* user_core_coord */ std::nullopt,
                                                /* user_fused_activation */ std::nullopt,
                                                /* user_run_batched */ false,
                                                /* transpose_a */ false,
                                                /* transpose_b */ false,
                                                /* output_tile */ output_tile);

                                            ttnn::Tensor output_tensor;
                                            output_tensor = ttnn::operations::matmul::matmul(
                                                in0_t,
                                                in1_t,
                                                /* bias */ std::nullopt,
                                                /* parameters */ matmul_params);
                                            output_tensor.deallocate();
                                            // Deallocate input tensors
                                            in0_t.deallocate();
                                            in1_t.deallocate();
                                            device->synchronize();
                                        }

                                        test_id++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    /*Prefix for the instantiated tests*/ MatmulTests,
    /*Test suite*/ Matmul2DHostPerfTestFixture,
    ::testing::Values(std::make_tuple(
        /* grid_size */ std::make_tuple(1, 2),
        /* tile_h */ 32,
        /* tile_w */ 32,
        /* num_warmup_iterations */ 5,
        /* num_measurement_iterations */ 1,
        /* use_program_cache */ false)));
