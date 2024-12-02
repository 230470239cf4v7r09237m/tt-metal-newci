# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 70

parameters = {
    "default": {
        "params": [
            (1, 1024, 1024, 1024),
            (1, 1024, 1024, 3072),
            (1, 1024, 1024, 32128),
            (1, 1024, 1024, 4096),
            (1, 1024, 1024, 512),
            (1, 10, 10, 128),
            (1, 128, 128, 9216),
            (1, 2048, 2048, 512),
            (1, 3072, 3072, 768),
            (1, 384, 384, 512),
            (1, 4096, 4096, 1024),
            (1, 512, 512, 1024),
            (1, 512, 512, 2048),
            (1, 512, 512, 32128),
            (1, 512, 512, 384),
            (1, 512, 512, 50272),
            (1, 512, 512, 512),
            (1, 768, 768, 3072),
            (1, 768, 768, 32128),
            (1, 768, 768, 50257),
            (1, 768, 768, 512),
            (1, 768, 768, 51865),
            (1, 768, 768, 768),
            (10, 1024, 1024, 1024),
            (10, 1024, 1024, 4096),
            (10, 1, 1, 128),
            (10, 2048, 2048, 512),
            (10, 3072, 3072, 768),
            (10, 4096, 4096, 1024),
            (10, 512, 512, 2048),
            (10, 512, 512, 512),
            (10, 768, 768, 3072),
            (10, 768, 768, 768),
            (1024, 160, 160, 256),
            (1024, 384, 384, 192),
            (1024, 512, 512, 256),
            (1024, 640, 640, 640),
            (128, 1, 1, 9216),
            (15, 1024, 1024, 512),
            (15, 384, 384, 512),
            (15, 512, 512, 1024),
            (15, 512, 512, 384),
            (1500, 768, 768, 768),
            (16384, 32, 32, 256),
            (19, 1024, 1024, 256008),
            (196, 1024, 1024, 512),
            (196, 768, 768, 384),
            (197, 1024, 1024, 1024),
            (197, 768, 768, 768),
            (2, 512, 512, 1),
            (2, 512, 512, 512),
            (2048, 768, 768, 262),
            (225, 512, 512, 12),
            (225, 512, 512, 16),
            (225, 512, 512, 24),
            (225, 512, 512, 32),
            (225, 512, 512, 3),
            (225, 512, 512, 4),
            (225, 512, 512, 6),
            (225, 512, 512, 8),
            (256, 1024, 1024, 512),
            (256, 1280, 1280, 1280),
            (256, 256, 256, 256),
            (256, 768, 768, 384),
            (32, 1536, 1536, 250880),
            (4, 768, 768, 51865),
            (4, 768, 768, 768),
            (4096, 320, 320, 320),
            (4096, 64, 64, 256),
            (45, 768, 768, 50257),
            (45, 768, 768, 768),
            (49, 1536, 1536, 768),
            (49, 2048, 2048, 1024),
            (5, 1024, 1024, 1024),
            (5, 1024, 1024, 3072),
            (59, 1024, 1024, 512),
            (59, 512, 512, 1024),
            (59, 512, 512, 50272),
            (64, 1280, 1280, 1280),
            (64, 1536, 1536, 768),
            (64, 2048, 2048, 1024),
            (7, 18176, 18176, 4544),
            (7, 4544, 4544, 18176),
            (7, 4544, 4544, 4544),
            (7, 4544, 4544, 4672),
            (7, 4544, 4544, 65024),
            (7, 768, 768, 2),
            (768, 196, 196, 384),
            (784, 384, 384, 192),
            (784, 512, 512, 256),
            (9, 768, 768, 1280),
            (9, 768, 768, 320),
            (9, 768, 768, 640),
            (920, 256, 256, 256),
        ],
    }
}


def run(
    params,
    *,
    device,
) -> list:
    [in0_h, in0_w, in1_h, in1_w] = params
    torch_input_tensor0 = torch.rand([in0_h, in0_w], dtype=torch.float32)
    torch_input_tensor1 = torch.rand([in1_h, in1_w], dtype=torch.float32)
    torch_output_tensor = torch.matmul(torch_input_tensor0, torch_input_tensor1)

    input_tensor0 = ttnn.from_torch(torch_input_tensor0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    output_tensor = ttnn.matmul(input_tensor0, input_tensor1)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.99
    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
