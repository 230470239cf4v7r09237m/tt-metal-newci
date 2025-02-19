# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from time import time
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)

from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import (
    num_cores_to_rectangle_grid,
    round_up,
)
from models.perf.benchmarking_utils import BenchmarkProfiler


def run_all_reduce_impl(
    mesh_device,
    output_shapes,
    cluster_axis,
    input_dtypes,
    num_links,
    input_num_cores,
    output_num_cores,
    num_iters=1,
    warmup_iters=0,
    enable_async=False,
    trace_mode=False,
    validate_all=True,
    profiler=BenchmarkProfiler(),
):
    cluster_shape = (8, 4)
    num_shapes = len(output_shapes)

    create_persistent_fabric = True
    teardown_persistent_fabric = True
    enable_persistent_fabric = True
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")

    ##################################
    ##### Set up fabric stuff
    ##################################
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    if create_persistent_fabric:
        mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
            mesh_device, [worker_sub_device], 0, 0, enable_persistent_fabric
        )
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0) for _ in range(8)
    ]

    logger.info(f"Output shape: {output_shapes}")

    try:
        ##################################
        ##### Set up input tensors/configs
        ##################################

        tt_input_tensors = []
        output_mem_configs = []
        output_tensor_goldens_list = []

        for t in range(num_shapes):
            ##### FF2 Case #####
            M, N = output_shapes[t][2:]
            N_per_shard = round_up(math.ceil(N / input_num_cores[t]), ttnn.TILE_SIZE)
            output_N_per_shard = round_up(math.ceil(N / output_num_cores[t]), ttnn.TILE_SIZE)
            input_shape = [*cluster_shape, M, N]

            CORE_RANGE = [(x, y) for y in range(compute_grid_size.y) for x in range(compute_grid_size.x)]
            core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in CORE_RANGE[: input_num_cores[t]]
                ]
            )
            input_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    core_range_set,
                    [M, N_per_shard],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            output_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in CORE_RANGE[: output_num_cores[t]]
                ]
            )
            output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    output_core_range_set,
                    [M, output_N_per_shard],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

            logger.info(f"Input shape: {input_shape[2:]}, Padded shape: {[M, N_per_shard * input_num_cores]}")
            input_tensor = torch.ones(input_shape)
            if len(output_tensor_goldens_list) != 0:
                input_tensor = output_tensor_goldens_list[-1][0].unsqueeze(cluster_axis[t])
                input_tensor = input_tensor @ torch.ones([1, 1, 3584, 2048])
                input_tensor = torch.concat([input_tensor] * cluster_shape[cluster_axis[t]], dim=cluster_axis[t])
            tt_input_tensor = ttnn.from_torch(
                input_tensor,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=input_dtypes[t],
                memory_config=input_mem_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )

            tt_input_tensors.append(tt_input_tensor)
            output_mem_configs.append(output_mem_config)

            # All-Reduce Golden
            output_tensor_goldens_list.append([torch.sum(input_tensor, dim=cluster_axis[t]) for _ in range(num_iters)])

        proj_weight = ttnn.from_torch(
            torch.ones([1, 1, 3584, 2048]),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        ##################################
        ##### Run the op
        ##################################

        def run_op(n_iters, store_all_results=True, proj_weight=proj_weight):
            outs = []
            for i in range(n_iters):
                out = tt_input_tensors[0]
                for t in range(num_shapes):
                    if t == 1:
                        # out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, 32, 2048], memory_config=tt_input_tensors[t].memory_config())
                        out = ttnn.matmul(out, proj_weight, memory_config=tt_input_tensors[t].memory_config())
                    # out = ttnn.reshard(out, tt_input_tensors[t].memory_config())
                    out = ttnn.experimental.all_reduce_async(
                        out,
                        cluster_axis=cluster_axis[t],
                        mesh_device=mesh_device,
                        multi_device_global_semaphore=ccl_semaphore_handles[i % 8],
                        memory_config=output_mem_configs[t],
                        topology=ttnn.Topology.Linear,
                        num_links=num_links[t],
                        subdevice_id=worker_sub_device_id,
                    )
                    # if not trace_mode:
                    #     print("SYNCRING")
                    #     ttnn.synchronize_devices(mesh_device)

                    if store_all_results:
                        outs.append(out)

            if store_all_results:
                return outs
            else:
                return [out]

        if trace_mode:
            ##### Compile Model #####
            logger.info("Compiling model")
            tt_outs = run_op(num_iters, store_all_results=validate_all)

            ##### Capture Trace #####
            logger.info("Capturing trace")
            if warmup_iters > 0:
                trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                tt_outs = run_op(warmup_iters, store_all_results=validate_all)
                ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
                for d in mesh_device.get_devices():
                    ttnn.synchronize_device(d)

            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            tt_outs = run_op(num_iters, store_all_results=validate_all)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            # for d in mesh_device.get_devices():
            #     ttnn.synchronize_device(d)

            ##### Run Trace #####
            logger.info("Starting Trace perf test...")
            profiler.start("all-reduce-async-trace-warmup")
            if warmup_iters > 0:
                ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
                ttnn.release_trace(mesh_device, trace_id_warmup)
                # for d in mesh_device.get_devices():
                #     ttnn.synchronize_device(d)
            profiler.end("all-reduce-async-trace-warmup")

            profiler.start("all-reduce-async-trace")
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.release_trace(mesh_device, trace_id)
            # for d in mesh_device.get_devices():
            #     ttnn.synchronize_device(d)
            profiler.end("all-reduce-async-trace")
            time_taken = profiler.get_duration("all-reduce-async-trace") - profiler.get_duration(
                "all-reduce-async-trace-warmup"
            )
            effective_iter = num_iters - warmup_iters
            logger.info(f"Time taken: {time_taken} s")
            logger.info(f"Time per iter: {time_taken / effective_iter} s")
            logger.info(f"Time per iter: {time_taken / effective_iter * 1e6} us")

        else:
            tt_outs = run_op(num_iters, store_all_results=validate_all)
            # ttnn.synchronize_devices(mesh_device)

        ##################################
        ##### Validation
        ##################################
        def validate(tt_out_tensor, output_tensor, cls_axis):
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
                # get_device_tensors returns row major, so we need to select the correct golden tensor
                if cls_axis == 0:
                    output_tensor_ = output_tensor[i % cluster_shape[not (cls_axis)]].unsqueeze(0).unsqueeze(0)
                else:
                    output_tensor_ = output_tensor[i // cluster_shape[cls_axis]].unsqueeze(0).unsqueeze(0)

                tt_output_tensor = t.cpu().to_torch()
                # logger.info(f"Checking for device {t.device().id()}")

                if input_dtypes[i % num_shapes] == ttnn.bfloat16:
                    eq, output = comp_pcc(tt_output_tensor, output_tensor_)
                else:
                    eq, output = comp_pcc(tt_output_tensor, output_tensor_)
                assert eq, f"{i} FAILED: {output}"
            # breakpoint()
            logger.info(f"PCC output is: {output}")

        if validate_all:
            for tensor_index in range(num_iters):
                for t in range(num_shapes):
                    tt_out_tensor = tt_outs[tensor_index * num_shapes + t]
                    output_tensor = output_tensor_goldens_list[t][tensor_index]
                    validate(tt_out_tensor, output_tensor, cluster_axis[t])
        else:
            tt_out_tensor = tt_outs[-1]
            output_tensor = output_tensor_goldens_list[-1]
            validate(tt_out_tensor, output_tensor, cluster_axis[0])

        # for i in range(mesh_device.get_num_devices()):
        #     assert (
        #         mesh_device.get_devices()[i].num_program_cache_entries() == 1
        #         or mesh_device.get_devices()[i].num_program_cache_entries() == num_iters
        #     ), f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries"

    finally:
        if enable_persistent_fabric and teardown_persistent_fabric:
            mesh_device.reset_sub_device_stall_group()
            t1 = time()
            teardown_fabric_interface(mesh_device)
            t2 = time()
            logger.info(f"Teardown time: {t2 - t1}")


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "output_shapes, cluster_axis, num_links, input_num_cores, output_num_cores",
    [
        # ([1, 1, 32, 2048], 0, 4, 24, 16),  # FF2/DO all reduce
        # ([1, 1, 32, 1280], 1, 3, 24, 40),  # QKV all reduce
        # ([1, 1, 32, 3584], 1, 3, 24, 24),  # FF1 all reduce
        # ([1, 1, 32, 2048], 0, 3, 24, 16),  # FF2/DO all reduce
        # ([1, 1, 32, 1280], 1, 2, 24, 40),  # QKV all reduce
        # ([1, 1, 32, 3584], 1, 2, 24, 24),  # FF1 all reduce
        # ([1, 1, 32, 2048], 0, 2, 24, 16),  # FF2/DO all reduce
        # ([1, 1, 32, 1280], 1, 1, 24, 40),  # QKV all reduce
        # ([1, 1, 32, 3584], 1, 1, 24, 24),  # FF1 all reduce
        # ([1, 1, 32, 2048], 0, 1, 24, 16),  # FF2/DO all reduce
        ([[1, 1, 32, 3584], [1, 1, 32, 2048]], [1, 1], [1, 1], [24, 24], [24, 16]),  # FF1 all reduce
    ],
)
@pytest.mark.parametrize(
    "input_dtypes",
    [
        # ttnn.bfloat16,
        [ttnn.bfloat8_b]
        * 2,
    ],
)
@pytest.mark.parametrize(
    "num_iters, warmup_iters",
    [
        # (1000, 100),
        (5, 0),
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_all_reduce(
    mesh_device,
    output_shapes,
    cluster_axis,
    input_dtypes,
    num_links,
    input_num_cores,
    output_num_cores,
    num_iters,
    warmup_iters,
    enable_async,
    trace_mode,
    use_program_cache,
    function_level_defaults,
):
    profiler = BenchmarkProfiler()

    run_all_reduce_impl(
        mesh_device,
        output_shapes,
        cluster_axis,
        input_dtypes,
        num_links,
        input_num_cores,
        output_num_cores,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        enable_async=enable_async,
        trace_mode=trace_mode,
        validate_all=True,
        profiler=profiler,
    )

    # time_taken = profiler.get_duration("all-reduce-async-trace") - profiler.get_duration(
    #     "all-reduce-async-trace-warmup"
    # )
    # effective_iter = num_iters - warmup_iters
    # latency_us = time_taken / effective_iter * 1e6
    # logger.info(f"Time taken: {time_taken} s")
    # logger.info(f"Time per iter: {latency_us} us")
