# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.patch_embed import PatchEmbed
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_patch_embed import ttnn_PatchEmbed


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, PatchEmbed):
            parameters["proj"] = {}
            parameters["proj"]["weight"] = ttnn.from_torch(model.proj.weight, dtype=ttnn.bfloat16)
            parameters["proj"]["bias"] = ttnn.from_torch(
                torch.reshape(model.proj.bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )
            parameters["pos_embed"] = ttnn.from_torch(model.pos_embed, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "h,w",
    (
        # (128, 128),
        (64, 64),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_patch_embed(device, reset_seeds, h, w):
    torch_model = PatchEmbed(
        height=h, width=w, patch_size=2, in_channels=16, embed_dim=1536, pos_embed_max_size=384
    ).to(dtype=torch.bfloat16)
    torch_model.eval()
    torch_input = torch.randn(2, 16, h, w, dtype=torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    parameters["pos_embed"] = ttnn.to_device(parameters["pos_embed"], device=device)

    torch_output = torch_model(torch_input)

    ttnn_model = ttnn_PatchEmbed(
        height=h,
        width=w,
        patch_size=2,
        in_channels=16,
        embed_dim=1536,
        pos_embed_max_size=384,
        parameters=parameters,
    )

    torch_input = torch_input.permute(0, 2, 3, 1)  # NCHW to NHWC
    ttnn_input = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_output = ttnn_model(device, ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output.unsqueeze(1), ttnn_output, pcc=0.9995)
