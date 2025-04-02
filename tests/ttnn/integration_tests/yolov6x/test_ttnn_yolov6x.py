# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.functional_yolov6x.reference.yolov6x import SPPF, Detect, Yolov6x_model
from models.experimental.functional_yolov6x.tt.ttnn_yolov6x import Ttnn_Detect, Ttnn_Sppf, Ttnn_Yolov6x
from models.experimental.functional_yolov6x.tt.model_preprocessing import (
    create_yolov6x_model_parameters_sppf,
    create_yolov6x_model_parameters_detect,
    create_yolov6x_model_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov6x_sppf(device, reset_seeds):
    torch_model = SPPF(640, 640)
    torch_model.eval()
    torch_input = torch.randn(1, 640, 20, 20)

    parameters = create_yolov6x_model_parameters_sppf(torch_model, torch_input, device)

    ttnn_model = Ttnn_Sppf(device, parameters, parameters.model_args)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn_model(device, ttnn_input)

    torch_output = torch_model(torch_input)

    output = ttnn.to_torch(output)
    output = output.permute(0, 3, 1, 2)
    output = output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, output, pcc=0.999)  # PCC: 0.9998703105881518


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_yolov6x_detect(device, reset_seeds):
    torch_input1 = torch.randn(1, 160, 80, 80)
    torch_input2 = torch.randn(1, 320, 40, 40)
    torch_input3 = torch.randn(1, 640, 20, 20)

    torch_model = Detect(nc=80, ch=(160, 320, 640))
    torch_model.eval()

    parameters = create_yolov6x_model_parameters_detect(torch_model, torch_input1, torch_input2, torch_input3, device)

    ttnn_model = Ttnn_Detect(device, parameters, parameters.model_args)

    input_tensor1 = torch.permute(torch_input1, (0, 2, 3, 1))
    input_tensor2 = torch.permute(torch_input2, (0, 2, 3, 1))
    input_tensor3 = torch.permute(torch_input3, (0, 2, 3, 1))
    input_tensor1 = input_tensor1.reshape(
        1,
        1,
        input_tensor1.shape[0] * input_tensor1.shape[1] * input_tensor1.shape[2],
        input_tensor1.shape[3],
    )
    input_tensor2 = input_tensor2.reshape(
        1,
        1,
        input_tensor2.shape[0] * input_tensor2.shape[1] * input_tensor2.shape[2],
        input_tensor2.shape[3],
    )
    input_tensor3 = input_tensor3.reshape(
        1,
        1,
        input_tensor3.shape[0] * input_tensor3.shape[1] * input_tensor3.shape[2],
        input_tensor3.shape[3],
    )
    ttnn_input1 = ttnn.from_torch(input_tensor1, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input2 = ttnn.from_torch(input_tensor2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input3 = ttnn.from_torch(input_tensor3, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn_model(device, ttnn_input1, ttnn_input2, ttnn_input3)

    torch_output = torch_model([torch_input1, torch_input2, torch_input3])

    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output[0], output, pcc=0.999)  # PCC: 0.9999919166110536 0.9999920682983192 (auto shard all)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov6x(device, reset_seeds):
    torch_model = Yolov6x_model()
    torch_model.eval()
    torch_input = torch.randn(1, 3, 640, 640)

    parameters = create_yolov6x_model_parameters(torch_model, torch_input, device)

    ttnn_model = Ttnn_Yolov6x(device, parameters, parameters.model_args)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn_model(device, ttnn_input)

    torch_output = torch_model(torch_input)

    output = ttnn.to_torch(output)
    # output = output.permute(0, 3, 1, 2)
    # output = output.reshape(torch_output[0].shape)
    assert_with_pcc(torch_output[0], output, pcc=0.999)  # 0.9999890988262345
