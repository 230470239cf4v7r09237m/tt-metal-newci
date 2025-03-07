#include "gtest/gtest.h"
#include "ttnn/device.hpp"
#include <vector>
#include <utility>
#include <boost/chrono.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <tt-metalium/logger.hpp>
#include "ttnn_test_fixtures.hpp"

#include "tools/profiler/op_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/functions.hpp"
#include "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp"

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/trace.hpp"

int get_device_freq() { return 1e6; }

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

std::vector<std::tuple<int, int, int, bool, bool, int, int, int>> matmul_shapes_bfloat16 = {
    {512, 512, 512, true, true, 1, 1, 1},
    {512, 1024, 1024, true, true, 1, 1, 1},
    {512, 1024, 2048, true, true, 1, 1, 1},
    {1024, 1024, 1024, true, true, 1, 1, 1},
    {1024, 1024, 2048, true, true, 1, 1, 1},
    {1024, 2048, 2048, true, true, 1, 1, 1},
    {2048, 2048, 2048, true, true, 1, 1, 1},
    {2048, 2048, 3072, true, true, 1, 1, 1},
    {2048, 3072, 3072, true, true, 2, 1, 1},
    {3072, 3072, 3072, true, true, 4, 1, 1},
    {3072, 3072, 4096, false, false, 2, 1, 1},
    {3072, 4096, 4096, false, false, 2, 1, 1},
    {4096, 4096, 4096, false, false, 1, 2, 2},
    {8192, 8192, 8192, false, false, 2, 4, 4},
    {16384, 16384, 16384, false, false, 4, 8, 8},
};

std::vector<std::tuple<int, int, int, bool, bool, int, int, int>> matmul_shapes_bfloat8_b = {
    {512, 512, 512, true, true, 1, 1, 1},
    {512, 1024, 1024, true, true, 1, 1, 1},
    {512, 1024, 2048, true, true, 1, 1, 1},
    {1024, 1024, 1024, true, true, 1, 1, 1},
    {1024, 1024, 2048, true, true, 1, 1, 1},
    {1024, 2048, 2048, true, true, 1, 1, 1},
    {2048, 2048, 2048, true, true, 1, 1, 1},
    {2048, 2048, 3072, true, true, 1, 1, 1},
    {2048, 3072, 3072, true, true, 1, 1, 1},
    {3072, 3072, 3072, true, true, 2, 1, 1},
    {3072, 3072, 4096, true, true, 2, 1, 1},
    {3072, 4096, 4096, true, true, 1, 2, 2},
    {4096, 4096, 4096, false, false, 1, 2, 2},
    {8192, 8192, 8192, false, false, 2, 4, 4},
    {16384, 16384, 16384, false, false, 4, 8, 8},
};

std::vector<std::tuple<int, int, int, bool, bool, int, int, int>> matmul_shapes_bfloat4_b = {
    {512, 512, 512, true, true, 1, 1, 1},
    {512, 1024, 1024, true, true, 1, 1, 1},
    {512, 1024, 2048, true, true, 1, 1, 1},
    {1024, 1024, 1024, true, true, 1, 1, 1},
    {1024, 1024, 2048, true, true, 1, 1, 1},
    {1024, 2048, 2048, true, true, 1, 1, 1},
    {2048, 2048, 2048, true, true, 1, 1, 1},
    {2048, 2048, 3072, true, true, 1, 1, 1},
    {2048, 3072, 3072, true, true, 1, 1, 1},
    {3072, 3072, 3072, true, true, 1, 1, 1},
    {3072, 3072, 4096, true, true, 1, 1, 1},
    {3072, 4096, 4096, true, true, 2, 1, 1},
    {4096, 4096, 4096, true, true, 2, 1, 1},
    {8192, 8192, 8192, false, false, 2, 2, 2},
    {16384, 16384, 16384, false, false, 4, 4, 4},
};

std::vector<std::tuple<DataType, MathFidelity, bool>> matmul_configs = {
    {DataType::BFLOAT16, MathFidelity::HiFi2, false},
    {DataType::BFLOAT16, MathFidelity::HiFi4, false},
    {DataType::BFLOAT8_B, MathFidelity::HiFi2, false},
    {DataType::BFLOAT8_B, MathFidelity::LoFi, false},
    {DataType::BFLOAT4_B, MathFidelity::LoFi, false},
    {DataType::BFLOAT16, MathFidelity::HiFi2, true},
    {DataType::BFLOAT16, MathFidelity::HiFi4, true},
    {DataType::BFLOAT8_B, MathFidelity::HiFi2, true},
    {DataType::BFLOAT8_B, MathFidelity::LoFi, true},
    {DataType::BFLOAT4_B, MathFidelity::LoFi, true},
};

class Matmul2DHostPerfTestFixture : public ttnn::distributed::test::TTNNFixtureWithTraceEnabledDevice,
                                    public testing::WithParamInterface<std::tuple<
                                        /* grid_size */ std::tuple<int, int>,
                                        /* tile_h */ int,
                                        /* tile_w */ int,
                                        /* num_warmup_iterations */ int,
                                        /* num_measurement_iterations */ int,
                                        /* use_program_cache */ bool>> {
public:
    Matmul2DHostPerfTestFixture() : ttnn::distributed::test::TTNNFixtureWithTraceEnabledDevice(24576, 3855488) {}
};

TEST_P(Matmul2DHostPerfTestFixture, Matmul2DHostPerfTest) {
    const std::tuple<int, int>& grid_size = std::get<0>(GetParam());
    const int& tile_h = std::get<1>(GetParam());
    const int& tile_w = std::get<2>(GetParam());
    const int& num_warmup_iterations = std::get<3>(GetParam());
    const int& num_measurement_iterations = std::get<4>(GetParam());
    const bool& use_program_cache = std::get<5>(GetParam());

    TT_FATAL(std::get<0>(grid_size) > 0 && std::get<1>(grid_size) > 0, "Invalid grid size");
    TT_ASSERT(num_measurement_iterations > 0, "Won't have data without at least one measurement iteration");
    tt::tt_metal::IDevice* device = &getDevice();
    const char* TT_METAL_HOME = std::getenv("TT_METAL_HOME");
    std::string ARTIFACTS_DIR = std::string(TT_METAL_HOME) + "/generated";
    std::string FILE_NAME = ARTIFACTS_DIR + "/matmul_2d_host_perf_report.csv";

    int LoFi_cycle = 16;
    int HiFi2_cycle = LoFi_cycle * 2;
    int HiFi3_cycle = LoFi_cycle * 3;
    int HiFi4_cycle = LoFi_cycle * 4;

    std::ofstream file(FILE_NAME);
    file << "m,k,n,use_trace,grid_size,in0_sharded,out_sharded,in0_storage_type,in1_storage_type,out_storage_type,"
            "dtype,math_fidelity,inference_time_avg (ns),TFLOPs (avg),Utilization (vs user grid),Utilization (vs 8x8 "
            "full grid)\n";

    for (const auto& config : matmul_configs) {
        DataType dtype = std::get<0>(config);
        MathFidelity math_fidelity = std::get<1>(config);
        bool use_trace = std::get<2>(config);

        std::vector<std::tuple<int, int, int, bool, bool, int, int, int>> matmul_shapes;
        if (dtype == DataType::BFLOAT16) {
            matmul_shapes = matmul_shapes_bfloat16;
        } else if (dtype == DataType::BFLOAT8_B) {
            matmul_shapes = matmul_shapes_bfloat8_b;
        } else if (dtype == DataType::BFLOAT4_B) {
            matmul_shapes = matmul_shapes_bfloat4_b;
        }

        for (const auto& shape : matmul_shapes) {
            int m = std::get<0>(shape);
            int k = std::get<1>(shape);
            int n = std::get<2>(shape);
            bool in0_sharded = std::get<3>(shape);
            bool out_sharded = std::get<4>(shape);
            int in0_block_w_div = std::get<5>(shape);
            int num_out_blocks_h = std::get<6>(shape);
            int num_out_blocks_w = std::get<7>(shape);
            std::vector<int64_t> in0_shape = {1, 1, m, k};
            std::vector<int64_t> in1_shape = {1, 1, k, n};

            int in0_block_w = k / std::get<0>(grid_size) / 32 / in0_block_w_div;
            int per_core_M = m / std::get<1>(grid_size) / tile_h;
            int per_core_N = n / std::get<0>(grid_size) / tile_w;
            int out_block_h = per_core_M / num_out_blocks_h;
            int out_block_w = per_core_N / num_out_blocks_w;
            auto [out_subblock_h, out_subblock_w] = get_subblock_sizes(out_block_h, out_block_w, out_sharded);

            tt::log_info(
                "M*K*N = {}*{}*{} out_subblock_h: {}, out_subblock_w: {}", m, k, n, out_subblock_h, out_subblock_w);

            std::string in0_storage_type = in0_sharded ? "L1" : "DRAM";
            std::string in1_storage_type = "DRAM";
            std::string out_storage_type = out_sharded ? "L1" : "DRAM";

            ttnn::MemoryConfig in0_memory_config =
                in0_sharded ? ttnn::operations::data_movement::create_sharded_memory_config(
                                  ttnn::Shape{1, 1, m, k},
                                  ttnn::CoreRangeSet(ttnn::CoreRange(
                                      CoreCoord(0, 0),
                                      ttnn::CoreCoord(std::get<0>(grid_size) - 1, std::get<1>(grid_size) - 1))),
                                  ttnn::operations::data_movement::ShardStrategy::BLOCK,
                                  tt::tt_metal::ShardOrientation::ROW_MAJOR)
                            : ttnn::DRAM_MEMORY_CONFIG;

            ttnn::Tensor in0_t =
                ttnn::ones(ttnn::Shape({m, k}), tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE);
            in0_t = ttnn::operations::core::to_device(in0_t, device, in0_memory_config, ttnn::DefaultQueueId);
            ttnn::Tensor in1_t =
                ttnn::random::random(ttnn::Shape({k, n}), tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE);

            in1_t = ttnn::operations::core::to_device(in1_t, device, ttnn::DRAM_MEMORY_CONFIG, ttnn::DefaultQueueId);
            ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig program_config{
                /* compute_with_storage_grid_size */ {std::get<0>(grid_size), std::get<1>(grid_size)},
                /* in0_block_w */ in0_block_w,
                /* out_subblock_h */ out_subblock_h,
                /* out_subblock_w */ out_subblock_w,
                /* out_block_h */ out_block_h,
                /* out_block_w */ out_block_w,
                /* per_core_M */ per_core_M,
                /* per_core_N */ per_core_N,
                /* transpose_mcast */ false,
                /* fused_activation */ std::nullopt};

            ttnn::WormholeComputeKernelConfig compute_kernel_config =
                ttnn::WormholeComputeKernelConfig{math_fidelity, true, false, true};

            ttnn::MemoryConfig out_mem_config =
                out_sharded ? ttnn::MemoryConfig{ttnn::TensorMemoryLayout::BLOCK_SHARDED, ttnn::BufferType::L1}
                            : ttnn::DRAM_MEMORY_CONFIG;

            Tile output_tile =
                out_sharded ? (tile_h <= 16 ? tt::tt_metal::Tile({tile_h, 32}) : tt::tt_metal::Tile({tile_h, tile_w}))
                            : tt::tt_metal::Tile({tile_h, tile_w});

            ttnn::Tensor output_tensor = ttnn::matmul(
                in0_t,
                in1_t,
                /* transpose_a */ false,
                /* transpose_b */ false,
                /* memory_config */ out_mem_config,
                /* dtype */ dtype,
                /* program_config */ program_config,
                /* activation */ std::nullopt,
                /* compute_kernel_config */ compute_kernel_config,
                /* core_grid */ std::nullopt,
                /* output_tile */ output_tile);
            for (int iter = 0; iter < num_warmup_iterations; ++iter) {
                // output_tensor = ttnn::matmul(in0_t, in1_t, program_config, out_mem_config, dtype,
                // compute_kernel_config, output_tile);
            }

            if (use_trace) {
                auto tid = ttnn::operations::trace::begin_trace_capture(device, ttnn::DefaultQueueId);
                for (int iter = 0; iter < num_measurement_iterations; ++iter) {
                    // ttnn::Tensor output_tensor = ttnn::matmul(
                    //     in0_t,
                    //     in1_t,
                    //     /* bias */ std::nullopt,
                    //     /* parameters */ matmul_config
                    // );
                }
                ttnn::operations::trace::end_trace_capture(device, tid, ttnn::DefaultQueueId);

                // Start timing

                ttnn::operations::trace::execute_trace(device, tid, ttnn::DefaultQueueId, false);
                // End timing

                ttnn::operations::trace::release_trace(device, tid);
            } else {
                // Start timing
                for (int iter = 0; iter < num_measurement_iterations; ++iter) {
                    // ttnn::Tensor output_tensor = ttnn::matmul(
                    //     in0_t,
                    //     in1_t,
                    //     /* bias */ std::nullopt,
                    //     matmul_config,
                    //     ttnn::DefaultQueueId
                    // );
                }
                // ttnn::synchronize_device(device);
                // End timing
            }

            double total_time = 5;
            double inference_time_avg = total_time / num_measurement_iterations;
            double tflops = 2.0 * m * k * n / 1e12 / inference_time_avg;
            int cycle_per_tile = (math_fidelity == MathFidelity::LoFi)    ? LoFi_cycle
                                 : (math_fidelity == MathFidelity::HiFi2) ? HiFi2_cycle
                                 : (math_fidelity == MathFidelity::HiFi3) ? HiFi3_cycle
                                                                          : HiFi4_cycle;
            int num_cores_user_grid = std::get<0>(grid_size) * std::get<1>(grid_size);
            auto compute_grid_size = device->compute_with_storage_grid_size();
            int num_cores_full_grid = compute_grid_size.x * compute_grid_size.y;
            double ideal_cycle_full_grid =
                static_cast<double>(m * k * n) / tile_h / tile_w / 32 * cycle_per_tile / num_cores_full_grid;
            double ideal_cycle_user_grid =
                static_cast<double>(m * k * n) / tile_h / tile_w / 32 * cycle_per_tile / num_cores_user_grid;
            double inference_cycle = inference_time_avg * get_device_freq() * 1e6;
            double utilization_full_grid = ideal_cycle_full_grid / inference_cycle;
            double utilization_user_grid = ideal_cycle_user_grid / inference_cycle;
            std::string utilization_full_grid_percentage = std::to_string(utilization_full_grid * 100) + "%";
            std::string utilization_user_grid_percentage = std::to_string(utilization_user_grid * 100) + "%";
            /*
            tt::log_info( "M*K*N = {}*{}*{} == inference time (avg): %f, tflops (avg): %f, utilization (vs user grid):
            %s, utilization (vs 8x8 grid): %s", m, k, n, inference_time_avg, tflops,
            utilization_user_grid_percentage.c_str(), utilization_full_grid_percentage.c_str());

            file << m << "," << k << "," << n << "," << (use_trace ? "true" : "false") << "," << std::get<0>(grid_size)
            << "x" << std::get<1>(grid_size) << ","
                 << in0_sharded << "," << out_sharded << "," << in0_storage_type << "," << in1_storage_type << "," <<
            out_storage_type << ","
                 << dtype << "," << math_fidelity << "," << inference_time_avg * 1e9 << "," << tflops << "," <<
            utilization_user_grid_percentage << ","
                 << utilization_full_grid_percentage << "\n";
                 */
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    /*Prefix for the instantiated tests*/ MatmulTests,
    /*Test suite*/ Matmul2DHostPerfTestFixture,
    ::testing::Values(std::make_tuple(std::make_tuple(8, 8), 32, 32, 1, 1, false)));
