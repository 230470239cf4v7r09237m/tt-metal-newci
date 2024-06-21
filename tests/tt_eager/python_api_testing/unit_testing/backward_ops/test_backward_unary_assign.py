# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_unary_assign(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = tt_lib.tensor.unary_assign_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y = torch.clone(in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("are_required_outputs", [[True], [False]])
def test_bw_unary_assign_opt_out(input_shapes, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    input_grad = None
    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    tt_output_tensor_on_device = tt_lib.tensor.unary_assign_bw(
        grad_tensor, input_tensor, are_required_outputs=are_required_outputs, input_grad=input_grad
    )

    in_data.retain_grad()

    pyt_y = torch.clone(in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = True
    if are_required_outputs[0]:
        status = status & compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("are_required_outputs", [[True], [False]])
def test_bw_unary_assign_opt_out_cq_id(input_shapes, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    input_grad = None
    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = tt_lib.tensor.unary_assign_bw(
        grad_tensor, input_tensor, are_required_outputs=are_required_outputs, input_grad=input_grad, queue_id=cq_id
    )

    in_data.retain_grad()

    pyt_y = torch.clone(in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = True
    if are_required_outputs[0]:
        status = status & compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
