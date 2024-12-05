# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch

from tests.didt.matmul_test_base import MatmulTestBase
import ttnn


class LMHeadTest(MatmulTestBase):
    def __init__(
        self,
        mesh_device,
        seq_len,
        inner_dim,
        weights_n,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        in0_dtype,
        in1_dtype,
        out_dtype,
        program_config,
        compute_config,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_iterations=False,
    ):
        super().__init__(
            mesh_device,
            seq_len,
            inner_dim,
            weights_n,
            in0_mem_config,
            in1_mem_config,
            out_mem_config,
            in0_dtype,
            in1_dtype,
            out_dtype,
            program_config,
            compute_config,
            loop_count,
            determinism_check_enabled,
            determinism_check_iterations,
        )

    def generate_weights(self, shape):
        return torch.randn(shape) - 0.95


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
def test_lm_head_matmul(mesh_device, iterations, determinism_check_iterations, use_program_cache, grid_size=(8, 8)):
    seq_len = 32
    inner_dim = 4544
    weights_n = 65024
    per_core_M = 1
    per_core_N = 32

    # Adjust dimensions if grid size smaller than 8x8;
    # For matmul 1D regardless of whether we remove column or row,
    # we need to reduce the weights width
    if grid_size[0] != 8 or grid_size[1] != 8:
        one_column_or_row = per_core_N * 8 * 32
        if grid_size[0] != 8:
            weights_n -= one_column_or_row * (8 - grid_size[0])
        else:
            weights_n -= one_column_or_row * (8 - grid_size[1])

    # Initialize input configurations
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Initialize matmul configurations
    out_subblock_h = 1
    out_subblock_w = 8

    subblock_1x1 = os.getenv("TT_USE_1X1_SUBBLOCK") == "1"
    if subblock_1x1:
        out_subblock_h = 1
        out_subblock_w = 1

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    fidelity_env = int(os.getenv("TT_MATH_FIDELITY", default=1))
    math_fidelity = ttnn.MathFidelity.LoFi
    if fidelity_env == 2:
        math_fidelity = ttnn.MathFidelity.HiFi2
    elif fidelity_env == 3:
        math_fidelity = ttnn.MathFidelity.HiFi3
    elif fidelity_env == 4:
        math_fidelity = ttnn.MathFidelity.HiFi4

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    lm_head_test = LMHeadTest(
        mesh_device,
        seq_len=seq_len,
        inner_dim=inner_dim,
        weights_n=weights_n,
        in0_mem_config=in0_mem_config,
        in1_mem_config=in1_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT8_B,
        in1_dtype=ttnn.DataType.BFLOAT8_B,
        out_dtype=ttnn.DataType.BFLOAT8_B,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
    )

    # Run test
    lm_head_test.run_matmul()


@pytest.mark.parametrize("logical_chip_id", range(32), ids=[f"logical_chip_{i}_" for i in range(32)])
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
def test_specific_chip_lm_head_matmul(
    mesh_device, logical_chip_id, iterations, determinism_check_iterations, use_program_cache
):
    assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_lm_head_matmul(
        mesh_device.get_device(logical_chip_id), iterations, determinism_check_iterations, use_program_cache
    )


@pytest.mark.parametrize(
    "board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["board_mesh_device"],
)
def test_specific_board_lm_head_matmul(board_mesh_device, iterations, determinism_check_iterations, use_program_cache):
    test_lm_head_matmul(board_mesh_device, iterations, determinism_check_iterations, use_program_cache)


@pytest.mark.parametrize(
    "grid_size",
    [(i, 8) for i in range(1, 9)] + [(8, i) for i in range(1, 8)],
    ids=[f"{i}x8" for i in range(1, 9)] + [f"8x{i}" for i in range(1, 8)],  # 1x8, 2x8 ... 8x1, 8x2...
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
def test_grid_size_lm_head_matmul(mesh_device, grid_size, iterations, determinism_check_iterations, use_program_cache):
    test_lm_head_matmul(mesh_device, iterations, determinism_check_iterations, use_program_cache, grid_size=grid_size)
