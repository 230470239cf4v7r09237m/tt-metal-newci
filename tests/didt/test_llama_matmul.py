# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole


class LLamaMatmulTest(OpTestBase):
    def __init__(
        self,
        mesh_device,
        in0_shape,
        in1_shape,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_layout,
        in1_layout,
        program_config,
        compute_config,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_iterations=False,
    ):
        super().__init__(
            mesh_device,
            in0_shape,
            in1_shape,
            in0_mem_config,
            in1_mem_config,
            out_mem_config,
            in0_dtype,
            in1_dtype,
            out_dtype,
            in0_layout,
            in1_layout,
            program_config,
            compute_config,
            loop_count,
            determinism_check_enabled,
            determinism_check_iterations,
        )


def num_to_corerange_set(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_x - 1, num_y - 1),
            ),
        }
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
    ],
    indirect=["mesh_device"],
)
def test_llama_matmul(mesh_device, iterations, determinism_check_iterations, use_program_cache, simulate_bh_harvesting):
    if simulate_bh_harvesting and is_blackhole() == False:
        pytest.skip("Blackhole harvesting simulation is only supported for Blackhole devices")
    # Initialize input configurations
    in0_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(32),
            [
                32,
                8192 // 32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Initialize matmul configurations
    compute_grid = get_blackhole_grid_size(simulate_bh_harvesting) if is_blackhole() else ttnn.CoreCoord(8, 8)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=16,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    in0_shape = [1, 1, 32, 8192]
    in1_shape = [1, 1, 8192, 16 * 1024]

    llama_matmul_test = LLamaMatmulTest(
        mesh_device,
        in0_shape=in0_shape,
        in1_shape=in1_shape,
        in0_mem_config=in0_mem_config,
        in1_mem_config=in1_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT16,
        in1_dtype=ttnn.DataType.BFLOAT8_B,
        out_dtype=ttnn.DataType.BFLOAT16,
        in0_layout=ttnn.TILE_LAYOUT,
        in1_layout=ttnn.TILE_LAYOUT,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
    )

    # Run test
    llama_matmul_test.run_op_test()


@skip_for_blackhole("Multi-chip Blackhole has not been tested")
@pytest.mark.parametrize("logical_chip_id", range(36), ids=[f"logical_chip_{i}_" for i in range(36)])
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
    ],
    indirect=["mesh_device"],
)
def test_specific_chip_llama_matmul(
    mesh_device, logical_chip_id, iterations, determinism_check_iterations, use_program_cache
):
    # Special case for galaxy:
    #   MeshDevice contains 32 chips, but their ids go from 4 - 35
    if len(mesh_device.get_device_ids()) == 32:
        assert (
            logical_chip_id >= 4 and logical_chip_id <= 35
        ), f"For TG configuration, logical chip id needs to be in range [4, 35] inclusive, but is {logical_chip_id}"
    else:
        assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_llama_matmul(
        mesh_device.get_device(logical_chip_id), iterations, determinism_check_iterations, use_program_cache, False
    )


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize(
    "board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["board_mesh_device"],
)
def test_specific_board_lm_head_matmul(board_mesh_device, iterations, determinism_check_iterations, use_program_cache):
    test_llama_matmul(board_mesh_device, iterations, determinism_check_iterations, use_program_cache, False)
