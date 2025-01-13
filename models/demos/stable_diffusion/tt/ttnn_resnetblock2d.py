# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.demos.stable_diffusion.tt.resnetblock2d_utils import (
    get_inputs,
    get_weights,
    get_mask_tensor,
    run_conv,
    run_conv_with_split,
)


def ResnetBlock2D(
    config,
    input_tensor=None,
    temb=None,
    parameters=None,
    device=None,
    eps=1e-5,
    groups=32,
    time_embedding_norm="default",
    non_linearity="silu",
    output_scale_factor=1.0,
    use_torch_conv=False,
):
    hidden_states = input_tensor
    if hidden_states.get_layout() != ttnn.TILE_LAYOUT:
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
    N = input_tensor.shape[0]
    batch_size = N
    C = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    in_channels = C
    input_height = H
    input_width = W
    grid_size = ttnn.CoreGrid(y=4, x=8)

    mem_config = ttnn.L1_MEMORY_CONFIG

    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, input_width * input_height, in_channels))

    gamma_t, beta_t = parameters.norm1.tt_weight, parameters.norm1.tt_bias
    input_mask_tensor = parameters.norm1.input_mask_tensor
    gamma_t = ttnn.to_device(gamma_t, device)
    beta_t = ttnn.to_device(beta_t, device)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    # shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)
    if hidden_states.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

    hidden_states = ttnn.group_norm(
        hidden_states,
        num_groups=groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        epsilon=eps,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )
    ttnn.deallocate(input_mask_tensor)
    ttnn.deallocate(gamma_t)
    ttnn.deallocate(beta_t)

    if non_linearity == "silu":
        DRAM_Flag = False
        if (
            (hidden_states.shape[3] == 960 and hidden_states.shape[2] == 16384)
            or (hidden_states.shape[3] == 640 and hidden_states.shape[2] == 16384)
            or (hidden_states.shape[3] == 1920 and hidden_states.shape[2] == 4096)
        ):
            hidden_states = ttnn.from_device(hidden_states)
            hidden_states = ttnn.to_dtype(hidden_states, ttnn.bfloat8_b)
            hidden_states = ttnn.to_device(hidden_states, device)
            # DRAM_Flag = True
            DRAM_Flag = False
        hidden_states, memory_config = get_inputs(device, hidden_states, grid_size)

        if DRAM_Flag:
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            hidden_states = ttnn.silu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden_states = ttnn.silu(hidden_states, memory_config=memory_config)

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, C))

    batch_size = hidden_states.shape[0]

    if hidden_states.shape[-1] == 1280 and hidden_states.shape[2] == 16:
        weight = ttnn.to_torch(parameters.conv1.weight).to(torch.float)
        bias = ttnn.to_torch(parameters.conv1.bias).to(torch.float).squeeze(0).squeeze(0).squeeze(0)
        conv = nn.Conv2d(
            in_channels=C, out_channels=parameters.conv1.bias.shape[-1], kernel_size=3, stride=1, padding=1
        )
        conv.weight = nn.Parameter(weight)
        conv.bias = nn.Parameter(bias)
        hidden_states = ttnn.to_torch(hidden_states).to(torch.float)
        hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
        print("Conv1 in torch_new_one:", hidden_states.shape)
        hidden_states = conv(hidden_states)
        hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:
        if parameters.conv1.use_split_conv:
            if hidden_states.is_sharded():
                hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))
            """
            if (hidden_states.shape[1] == 1920 and hidden_states.shape[1] == 64) or (
                hidden_states.shape[1] == 960 and hidden_states.shape[1] == 128
            ):
                mem_config = ttnn.DRAM_MEMORY_CONFIG
            """
            hidden_states = run_conv_with_split(
                device,
                hidden_states,
                hidden_states.shape[0],
                parameters,
                kernel_size=3,
                stride=1,
                pad=1,
                split_factor=parameters.conv1.split_factor,
                ttnn_weight=parameters.conv1.weight,
                ttnn_bias=parameters.conv1.bias,
                memory_config=mem_config,
            )
            mem_config = ttnn.L1_MEMORY_CONFIG
        else:
            if hidden_states.is_sharded():
                hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
            hidden_states = run_conv(
                device,
                output_channels=parameters.conv1.bias.shape[-1],
                input_channels=C,
                input_height=H,
                input_width=W,
                filter_height=3,
                stride_h=1,
                pad_h=1,
                tt_input_tensor=hidden_states,
                tt_weight_tensor=parameters.conv1.weight,
                tt_bias_tensor=parameters.conv1.bias,
            )

    if temb is not None:
        temb = ttnn.silu(
            temb,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        temb = ttnn.linear(
            temb,
            parameters.time_emb_proj.weight,
            bias=parameters.time_emb_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
        )

    if temb is not None and time_embedding_norm == "default":
        # temb = ttnn.reshape(temb, (temb.shape[0], temb.shape[1], 1, 1))
        temb = ttnn.permute(temb, (2, 3, 0, 1))
        hidden_states = ttnn.add(
            hidden_states,
            temb,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    ttnn.deallocate(temb)
    ttnn.DumpDeviceProfiler(device)

    N = hidden_states.shape[0]
    C = hidden_states.shape[1]
    H = hidden_states.shape[2]
    W = hidden_states.shape[3]

    gamma_t, beta_t = parameters.norm2.tt_weight, parameters.norm2.tt_bias
    gamma_t = ttnn.to_device(gamma_t, device)
    beta_t = ttnn.to_device(beta_t, device)
    input_mask_tensor = parameters.norm2.input_mask_tensor
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.reshape(hidden_states, (N, 1, W * H, C))

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

    hidden_states = ttnn.group_norm(
        hidden_states,
        num_groups=groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        epsilon=eps,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )
    ttnn.deallocate(input_mask_tensor)
    ttnn.deallocate(gamma_t)
    ttnn.deallocate(beta_t)

    if non_linearity == "silu":
        hidden_states, memory_config = get_inputs(device, hidden_states, grid_size)
        hidden_states = ttnn.silu(hidden_states, memory_config=memory_config)

    hidden_states = ttnn.reshape(hidden_states, (N, H, W, C))
    batch_size = hidden_states.shape[0]
    # if use_torch_conv:

    if parameters.conv2.use_split_conv:
        if hidden_states.is_sharded():
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

        hidden_states = run_conv_with_split(
            device,
            hidden_states,
            hidden_states.shape[0],
            parameters,
            kernel_size=3,
            stride=1,
            pad=1,
            split_factor=parameters.conv2.split_factor,
            ttnn_weight=parameters.conv2.weight,
            ttnn_bias=parameters.conv2.bias,
        )
    else:
        if hidden_states.is_sharded():
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = run_conv(
            device,
            output_channels=parameters.conv2.bias.shape[-1],
            input_channels=C,
            input_height=H,
            input_width=W,
            filter_height=3,
            stride_h=1,
            pad_h=1,
            tt_input_tensor=hidden_states,
            tt_weight_tensor=parameters.conv2.weight,
            tt_bias_tensor=parameters.conv2.bias,
        )
    ttnn.DumpDeviceProfiler(device)
    if parameters.conv2.conv_shortcut:
        if parameters.conv_shortcut.use_split_conv:
            if input_tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            # print("Shape of input conv shortcut :", input_tensor.shape, " ", parameters.conv_shortcut.split_factor)
            input_tensor = run_conv_with_split(
                device,
                input_tensor,
                input_tensor.shape[0],
                parameters,
                kernel_size=1,
                stride=1,
                pad=0,
                split_factor=parameters.conv_shortcut.split_factor,
                ttnn_weight=parameters.conv_shortcut.weight,
                ttnn_bias=parameters.conv_shortcut.bias,
            )
        else:
            input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
            if input_tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)

            # weight = ttnn.squeeze(parameters.conv_shortcut.weight, -1)
            # weight = ttnn.squeeze(weight, -1)
            input_tensor = run_conv(
                device,
                output_channels=parameters.conv_shortcut.bias.shape[-1],
                input_channels=C,
                input_height=H,
                input_width=W,
                filter_height=1,
                stride_h=1,
                pad_h=0,
                tt_input_tensor=input_tensor,
                tt_weight_tensor=parameters.conv_shortcut.weight,
                tt_bias_tensor=parameters.conv_shortcut.bias,
            )

    output_tensor = ttnn.add(
        input_tensor,
        hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output_tensor = ttnn.mul(
        output_tensor,
        (1 / output_scale_factor),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    # print("Dtype in resnet :", output_tensor.get_dtype())
    return output_tensor
