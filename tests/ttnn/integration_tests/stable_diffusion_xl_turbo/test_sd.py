# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from diffusers import AutoPipelineForText2Image
from models.demos.stabe_diffusion_xl_turbo.tt.resnetblock2d_utils import update_params, run_conv_with_split, run_conv
from models.demos.stabe_diffusion_xl_turbo.tt.sd_transformer2d import (
    sd_geglu,
    sd_feed_forward,
    sd_attention,
    sd_basic_transformer_block,
    sd_transformer_2d,
)
from models.demos.stabe_diffusion_xl_turbo.tt.resnetblock2d import ResnetBlock2D
from models.demos.stabe_diffusion_xl_turbo.tt.sd_unetmidblock2dcrossattn import sd_unetmidblock2dcrossattn
from models.demos.stabe_diffusion_xl_turbo.tt.sd_crossattnupblock2d import sd_crossattnupblock2d
from models.demos.stabe_diffusion_xl_turbo.tt.utils import custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters

from models.demos.stabe_diffusion_xl_turbo.tt import tt_upsample_2d
from models.demos.stabe_diffusion_xl_turbo.tt import custom_preprocessing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_geglu(device, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    geglu_model = model.down_blocks[1].attentions[0].transformer_blocks[0].ff.net[0]

    hidden_states = torch.randn((1, 4096, 640), dtype=torch.float16)

    torch_output = geglu_model(hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: geglu_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn_output = sd_geglu(
        ttnn_input_tensor,
        parameters,
        device,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99947)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_feed_forward(device, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    feed_forward_model = model.down_blocks[1].attentions[0].transformer_blocks[0].ff
    hidden_states = torch.randn((1, 4096, 640), dtype=torch.float16)

    torch_output = feed_forward_model(hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: feed_forward_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn_output = sd_feed_forward(
        ttnn_input_tensor,
        parameters,
        device,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.9994)


@pytest.mark.parametrize(
    "N, C, H, W, has_encoder_hidden_states, index",
    [
        (1, 2, 4096, 640, False, 1),
        (1, 2, 4096, 640, True, 1),
        (1, 2, 1024, 1280, False, 2),
        (1, 2, 1024, 1280, True, 2),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_attention_down_blocks(device, N, C, H, W, has_encoder_hidden_states, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    encoder_hidden_states = None
    ttnn_encoder_hidden_states = None
    attention_model = model.down_blocks[index].attentions[0].transformer_blocks[0].attn1
    heads = attention_model.heads

    if has_encoder_hidden_states:
        attention_model = model.down_blocks[index].attentions[0].transformer_blocks[0].attn2
        encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float16)
        ttnn_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    hidden_states = torch.randn((N, H, W), dtype=torch.float16)

    torch_output = attention_model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: attention_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = sd_attention(
        ttnn_input_tensor,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        heads=heads,
        parameters=parameters,
        device=device,
        query_dim=hidden_states.shape[-1] // 8,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.9987)


@pytest.mark.parametrize(
    "N, C, H, W, has_encoder_hidden_states, index",
    [
        (1, 2, 4096, 640, False, 1),
        (1, 2, 4096, 640, True, 1),
        (1, 2, 1024, 1280, False, 0),
        (1, 2, 1024, 1280, True, 0),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_attention_up_blocks(device, N, C, H, W, has_encoder_hidden_states, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    encoder_hidden_states = None
    ttnn_encoder_hidden_states = None
    attention_model = model.up_blocks[index].attentions[0].transformer_blocks[0].attn1
    heads = attention_model.heads

    if has_encoder_hidden_states:
        attention_model = model.up_blocks[index].attentions[0].transformer_blocks[0].attn2
        encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float16)
        ttnn_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    hidden_states = torch.randn((N, H, W), dtype=torch.float16)

    torch_output = attention_model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: attention_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = sd_attention(
        ttnn_input_tensor,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        heads=heads,
        parameters=parameters,
        device=device,
        query_dim=hidden_states.shape[-1] // 8,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "N, C, H, W, has_encoder_hidden_states, index",
    [
        (1, 2, 1024, 1280, False, 0),
        (1, 2, 1024, 1280, True, 0),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_attention_mid_blocks(device, N, C, H, W, has_encoder_hidden_states, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    encoder_hidden_states = None
    ttnn_encoder_hidden_states = None
    attention_model = model.mid_block.attentions[0].transformer_blocks[0].attn1
    heads = attention_model.heads

    if has_encoder_hidden_states:
        attention_model = model.mid_block.attentions[0].transformer_blocks[0].attn2
        encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float16)
        ttnn_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    hidden_states = torch.randn((N, H, W), dtype=torch.float16)

    torch_output = attention_model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: attention_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = sd_attention(
        ttnn_input_tensor,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        heads=heads,
        parameters=parameters,
        device=device,
        query_dim=hidden_states.shape[-1] // 8,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "N, C, H, W, attention_head_dim, index",
    [
        (1, 2, 4096, 640, 40, 1),
        (1, 2, 1024, 1280, 40, 2),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_basic_transformer_block(device, N, C, H, W, attention_head_dim, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config

    basic_transformer = model.down_blocks[index].attentions[0].transformer_blocks[0]

    hidden_states = torch.randn((N, H, W), dtype=torch.float32)
    encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float32)

    torch_output = basic_transformer(hidden_states, encoder_hidden_states=encoder_hidden_states)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None

    parameters = preprocess_model_parameters(initialize_model=lambda: basic_transformer, device=device)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = sd_basic_transformer_block(
        hidden_states=ttnn_hidden_states,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        timestep=timestep,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        class_labels=class_labels,
        config=config,
        attention_head_dim=attention_head_dim,
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.996)


@pytest.mark.parametrize(
    "input_shape, index1, index2, attention_head_dim, block, num_layers",
    [
        # ((1, 640, 64, 64), 1, 0, 10, "down", 2),
        # ((1, 1280, 32, 32), 2, 0, 20, "down", 10),
        # ((1, 640, 64, 64), 1, 0, 10, "up", 2),
        # ((1, 1280, 32, 32), 0, 0, 20, "up", 10),
        ((1, 1280, 32, 32), 0, 0, 20, "mid", 10),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_transformer_2d_model(input_shape, index1, index2, block, attention_head_dim, num_layers, device, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config

    num_attention_heads = 8
    norm_num_groups = 32

    if block == "down":
        transformer_2d_model = model.down_blocks[index1].attentions[index2]
    elif block == "up":
        transformer_2d_model = model.up_blocks[index1].attentions[index2]
    else:
        transformer_2d_model = model.mid_block.attentions[index1]

    hidden_states = torch.randn(input_shape, dtype=torch.float32)
    encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float32)

    torch_output = transformer_2d_model(hidden_states, encoder_hidden_states=encoder_hidden_states)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None
    return_dict = False

    parameters = preprocess_model_parameters(initialize_model=lambda: transformer_2d_model, device=device)

    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device)

    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_output = sd_transformer_2d(
        hidden_states=ttnn_hidden_states,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        parameters=parameters,
        device=device,
        timestep=timestep,
        class_labels=class_labels,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        attention_head_dim=attention_head_dim,
        num_layers=num_layers,
        norm_num_groups=norm_num_groups,
        attention_mask=attention_mask,
        config=config,
        eps=1e-06,
    )
    ttnn_output = ttnn.to_torch(ttnn_output[0])

    assert_with_pcc(torch_output.sample, ttnn_output, 0.98)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index1, index2, block_name",
    [
        (1, 320, 128, 128, 0, 0, "down"),
        (1, 320, 128, 128, 0, 1, "down"),
        (1, 320, 64, 64, 1, 0, "down"),
        (1, 640, 64, 64, 1, 1, "down"),
        (1, 640, 32, 32, 2, 0, "down"),
        (1, 1280, 32, 32, 2, 1, "down"),
        (1, 1280, 32, 32, 0, 0, "mid"),
        (1, 1280, 32, 32, 1, 0, "mid"),  # 0.988
        (1, 2560, 32, 32, 0, 0, "up"),  #  0.96
        (1, 2560, 32, 32, 0, 1, "up"),  #  0.97
        (1, 1920, 32, 32, 0, 2, "up"),  #  0.979
        (1, 1920, 64, 64, 1, 0, "up"),  # 0.94
        (1, 1280, 64, 64, 1, 1, "up"),
        (1, 960, 64, 64, 1, 2, "up"),  # 0.94
        (1, 960, 128, 128, 2, 0, "up"),  # 0.94
        (1, 640, 128, 128, 2, 1, "up"),  # 0.94
        (1, 640, 128, 128, 2, 2, "up"),  # 0.98
    ],
)
def test_resnet_block_2d_1024x1024(
    device,
    batch_size,
    in_channels,
    input_height,
    input_width,
    index1,
    index2,
    block_name,
    reset_seeds,
):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
    model = pipe.unet
    model.eval()
    config = model.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    print("update_param")
    parameters = update_params(parameters)

    if block_name == "up":
        parameters = parameters.up_blocks[index1].resnets[index2]
        resnet = model.up_blocks[index1].resnets[index2]
    elif block_name == "down":
        parameters = parameters.down_blocks[index1].resnets[index2]
        resnet = model.down_blocks[index1].resnets[index2]
    else:
        parameters = parameters.mid_block.resnets[index1]
        resnet = model.mid_block.resnets[index1]

    temb_channels = 1280
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    temb_shape = [1, temb_channels]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)
    torch_output = resnet(input, temb)
    input = ttnn.from_torch(
        input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        # memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    temb = ttnn.from_torch(
        temb,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    resnet_block = ResnetBlock2D(
        config,
        input,
        temb,
        parameters,
        device,
    )
    resnet_block = ttnn.to_torch(resnet_block)  # .permute(0,3,1,2)
    print(torch_output.shape, resnet_block.shape)
    assert_with_pcc(torch_output, resnet_block, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unetmidblock2dcrossattn_1024x1024(
    device,
):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
    model = pipe.unet
    model.eval()
    config = model.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters = update_params(parameters)

    model = model.mid_block
    parameters = parameters.mid_block
    hidden_states_shape = [1, 1280, 32, 32]
    temb_shape = [1, 1280]
    encoder_hidden_states = [1, 77, 2048]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states)

    torch_out = model(input, temb=temb, encoder_hidden_states=encoder_hidden_states)

    tt_input = ttnn.from_torch(
        input,
        dtype=ttnn.DataType.BFLOAT16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_temb = ttnn.from_torch(temb, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_out = sd_unetmidblock2dcrossattn(device, tt_input, tt_temb, ttnn_encoder_hidden_states, parameters, config)

    ttnn_output = ttnn.to_torch(ttnn_out)

    assert_with_pcc(torch_out, ttnn_output, 0.98)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "in_channels, res_channel1, res_channel2, res_channel3, input_height, input_width, index, num_layers, attention_head_dim",
    [
        (1280, 640, 1280, 1280, 32, 32, 0, 20, 10),
        (1280, 320, 640, 640, 64, 64, 1, 10, 2),
    ],
)
def test_crossattnupblock2d_1024x1024(
    device,
    in_channels,
    res_channel1,
    res_channel2,
    res_channel3,
    input_height,
    input_width,
    index,
    num_layers,
    attention_head_dim,
):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
    model = pipe.unet
    model.eval()
    config = model.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters = update_params(parameters)

    model = model.up_blocks[index]
    parameters = parameters.up_blocks[index]

    hidden_states_shape = [1, in_channels, input_height, input_width]
    res_hidden_states_tuple1 = [1, res_channel1, input_height, input_width]
    res_hidden_states_tuple2 = [1, res_channel2, input_height, input_width]
    res_hidden_states_tuple3 = [1, res_channel3, input_height, input_width]
    temb_shape = [1, 1280]
    encoder_hidden_states = [1, 77, 2048]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states)
    res_hidden_states_tuple = (
        torch.randn(res_hidden_states_tuple1),
        torch.randn(res_hidden_states_tuple2),
        torch.randn(res_hidden_states_tuple3),
    )

    torch_out = model(
        input, temb=temb, encoder_hidden_states=encoder_hidden_states, res_hidden_states_tuple=res_hidden_states_tuple
    )

    tt_input = ttnn.from_torch(
        input,
        dtype=ttnn.DataType.BFLOAT16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_temb = ttnn.from_torch(temb, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_res_hidden_states_tuple1 = ttnn.from_torch(
        res_hidden_states_tuple[0],
        dtype=ttnn.DataType.BFLOAT16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_res_hidden_states_tuple2 = ttnn.from_torch(
        res_hidden_states_tuple[1],
        dtype=ttnn.DataType.BFLOAT16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_res_hidden_states_tuple3 = ttnn.from_torch(
        res_hidden_states_tuple[2],
        dtype=ttnn.DataType.BFLOAT16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_res_hidden_states_tuple = (tt_res_hidden_states_tuple1, tt_res_hidden_states_tuple2, tt_res_hidden_states_tuple3)

    ttnn_out = sd_crossattnupblock2d(
        device,
        tt_input,
        tt_res_hidden_states_tuple,
        tt_temb,
        ttnn_encoder_hidden_states,
        parameters,
        config,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
    )

    ttnn_output = ttnn.to_torch(ttnn_out)

    assert_with_pcc(torch_out, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, index",
    [
        (1, 1280, 32, 32, 0),
        (1, 640, 64, 64, 1),
    ],
)
def test_upsample(device, N, C, H, W, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config
    model = model.up_blocks[index].upsamplers[0]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=custom_preprocessing.custom_preprocessor
    )
    input_tensor = torch.randn((N, C, H, W), dtype=torch.float32)
    torch_output = model(input_tensor)
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    ttnn_hidden_state = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device)

    output = tt_upsample_2d.upsample(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.94 if index == 0 else 0.93)  # -0.000740


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_split_conv(device):
    torch.manual_seed(9)
    weigts_shape = [320, 320, 3, 3]
    bias_shape = [320]
    input_shape = [1, 320, 128, 128]

    bias = torch.randn(bias_shape, dtype=torch.bfloat16).float()
    weights = torch.randn(weigts_shape, dtype=torch.bfloat16).float()
    input = torch.randn(input_shape, dtype=torch.bfloat16).float()

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        input,
        weights,
        bias=bias,
        stride=(1, 1),
        padding=(1, 1),
    )

    tt_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
    tt_weights = ttnn.from_torch(weights, dtype=ttnn.bfloat16)
    tt_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)

    parameters = {}

    tt_out = run_conv_with_split(device, tt_input, 1, parameters, 3, 1, 1, 2, ttnn_weight=tt_weights, ttnn_bias=tt_bias)
    # tt_out = run_conv(device, 320, 320,128,128, 3,1,1,tt_input,tt_weights,tt_bias )

    ttnn_output = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out_golden_tensor, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_conv(device):
    torch.manual_seed(9)
    weigts_shape = [1280, 1280, 3, 3]
    bias_shape = [1280]
    input_shape = [1, 1280, 32, 32]

    bias = torch.randn(bias_shape, dtype=torch.bfloat16).float()
    weights = torch.randn(weigts_shape, dtype=torch.bfloat16).float()
    input = torch.randn(input_shape, dtype=torch.bfloat16).float()

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        input,
        weights,
        bias=bias,
        stride=(1, 1),
        padding=(1, 1),
    )

    tt_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
    tt_weights = ttnn.from_torch(weights, dtype=ttnn.bfloat16)
    tt_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)
    tt_input = ttnn.permute(tt_input, (0, 2, 3, 1))

    parameters = {}

    tt_out = run_conv(device, 1280, 1280, 32, 32, 3, 1, 1, tt_input, tt_weights, tt_bias)
    # tt_out = run_conv(device, 320, 320,128,128, 3,1,1,tt_input,tt_weights,tt_bias )

    ttnn_output = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out_golden_tensor, ttnn_output, 0.99)
