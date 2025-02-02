# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from diffusers import StableDiffusion3Pipeline
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.clip_text_model_with_projection import (
    CLIPTextModelWithProjection,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_text_model_with_projection import (
    ttnn_CLIPTextModelWithProjection,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_clip_text_model_with_projection(device, reset_seeds):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )

    config = pipe.text_encoder.config

    reference_model = CLIPTextModelWithProjection(config)

    reference_model.eval()

    input_ids = torch.normal(0.0, 30.0, size=(1, 77))
    input_ids = input_ids.abs()
    input_ids = input_ids.to(torch.int64)

    torch_output = reference_model(input_ids, None, None, None, True, None)

    parameters = preprocess_model_parameters(initialize_model=lambda: reference_model, device=device)

    parameters["text_model"]["position_ids"] = ttnn.from_torch(
        reference_model.text_model.embeddings.position_ids, device=device
    )

    ttnn_input_ids = ttnn.from_torch(
        input_ids,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_model = ttnn_CLIPTextModelWithProjection(config, parameters=parameters)

    ttnn_output = ttnn_model(ttnn_input_ids, None, None, None, True, None, parameters=parameters)

    assert_with_pcc(torch_output.text_embeds, ttnn.to_torch(ttnn_output.text_embeds), pcc=0.99)

    assert_with_pcc(torch_output.last_hidden_state, ttnn.to_torch(ttnn_output.last_hidden_state), pcc=0.98)

    for i in range(len(torch_output.hidden_states)):
        assert_with_pcc(torch_output.hidden_states[i], ttnn.to_torch(ttnn_output.hidden_states[i]), pcc=0.98)
