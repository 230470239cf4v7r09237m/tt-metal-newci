# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import (
    is_wormhole_b0,
    profiler,
)
from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra
from models.demos.ttnn_resnet.tests.demo_utils import get_data, get_data_loader, get_batch

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


class ResNet50Trace2CQ:
    def __init__(self):
        ...

    def initialize_resnet50_trace_2cqs_inference(
        self,
        device,
        device_batch_size=1,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
    ):
        self.test_infra = create_test_infra(
            device,
            device_batch_size,
            act_dtype,
            weight_dtype,
            ttnn.MathFidelity.LoFi,
            True,
            dealloc_input=True,
            final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.device = device
        self.tt_inputs_host, sharded_mem_config_DRAM, self.input_mem_config = self.test_infra.setup_dram_sharded_input(
            device
        )
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self.op_event = ttnn.create_event(device)
        self.write_event = ttnn.create_event(device)
        # Initialize the op event so we can write
        ttnn.record_event(0, self.op_event)

        # First run configures convs JIT
        profiler.start("compile")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        ttnn.record_event(1, self.write_event)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        spec = self.test_infra.input_tensor.spec
        ttnn.record_event(0, self.op_event)
        self.test_infra.run()
        profiler.end("compile")
        self.test_infra.validate()
        self.test_infra.output_tensor.deallocate(force=True)

        # Optimized run
        profiler.start("cache")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        ttnn.record_event(1, self.write_event)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        ttnn.record_event(0, self.op_event)
        self.test_infra.run()
        profiler.end("cache")
        self.test_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        ttnn.record_event(1, self.write_event)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        ttnn.record_event(0, self.op_event)
        self.test_infra.output_tensor.deallocate(force=True)
        trace_input_addr = ttnn.buffer_address(self.test_infra.input_tensor)
        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.tt_output_res = self.test_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        ttnn.synchronize_devices(self.device)
        assert trace_input_addr == ttnn.buffer_address(self.input_tensor)

    def execute_resnet50_trace_2cqs_inference(self, tt_inputs_host=None):
        # More optimized run with caching
        # if use_signpost:
        #    signpost(header="start")
        outputs = []
        tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = self.test_infra.setup_dram_sharded_input(
            self.device, tt_inputs_host
        )
        self.tt_image_res = tt_inputs_host.to(self.device, sharded_mem_config_DRAM)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        ttnn.wait_for_event(1, self.op_event)
        ttnn.record_event(1, self.write_event)
        ttnn.wait_for_event(0, self.write_event)
        # TODO: Add in place support to ttnn to_memory_config
        ttnn.record_event(0, self.op_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, input_mem_config)
        self.test_infra.run()
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=True)
        outputs = ttnn.from_device(self.test_infra.output_tensor, blocking=True)
        ttnn.synchronize_devices(self.device)

        if use_signpost:
            signpost(header="stop")
        self.test_infra.validate(outputs)
        return outputs

    def release_resnet50_trace_2cqs_inference(self):
        ttnn.release_trace(self.device, self.tid)
