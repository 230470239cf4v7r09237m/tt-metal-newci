# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.common.rmsnorm import RMSNorm as TtRMSNorm
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import RMSNorm as RefRMSNorm
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    skip_for_parallelism,
    skip_for_batch_parallelism,
    skip_for_model_parallelism,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3.tt.distributed_norm import DistributedNorm


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"),
            len(ttnn.get_device_ids())
            # "N150"
        )
    ],
    indirect=True,
)
# @pytest.mark.parametrize(
#     "batch_dp_tp",
#     [
#         # (1, 1, 8),
#         # (8, 8, 1),
#         (1, 1, 2),
#         # (32, 2, 1),
#         # (64, 2, 1),
#         # (32, 2, 1),
#         # (32, 1, 2),
#         # (64, 1, 2),
#         # (2, 2, 1),
#     ],
#     ids=lambda args: "batch_{}_dp_{}_tp_{}".format(*args),
# )
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("dp", [True, False])
@pytest.mark.parametrize(
    "max_seq_len",
    (128,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize(
    "mode",
    ["prefill", "decode"],
)
def test_llama_rms_norm_inference(
    batch_size,
    dp,
    max_seq_len,
    mode,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
):
    batch_size = batch_size * mesh_device.get_num_devices() if dp else batch_size
    data_parallel = mesh_device.get_num_devices() if dp else 1
    tensor_parallel = mesh_device.get_num_devices() if not dp else 1

    skip, reason = skip_for_batch_parallelism(batch_size, data_parallel)
    if skip:
        pytest.skip(reason)

    skip, reason = skip_for_parallelism(
        mesh_device.get_num_devices() if mesh_device else 0, data_parallel, tensor_parallel
    )
    if skip:
        pytest.skip(reason)

    skip, reason = skip_for_model_parallelism(data_parallel)
    if skip:
        pytest.skip(reason)

    dtype = ttnn.bfloat16

    mesh_device.enable_async(True)

    model_args = TtModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        data_parallel=data_parallel,
        tensor_parallel=tensor_parallel,
    )

    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", 0)
    first_layer_prefix = state_dict_prefix + "attention_norm."

    # Create the inner RMSNormxw
    tt_inner_norm = TtRMSNorm(
        device=mesh_device,
        dim=model_args.dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_key="attention_norm",
        weight_dtype=dtype,
        is_distributed=model_args.is_distributed_norm,
        sharded_program_config=model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
        sharded_output_config=model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
    )

    # Wrap it in DistributedNorm
    tt_model = DistributedNorm(tt_inner_norm, model_args, TG=model_args.is_galaxy)

    # Create reference model (unchanged)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = RefRMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
    reference_model.load_state_dict(partial_state_dict)

    input = torch.rand(1, 1, model_args.num_devices_dp * 32, model_args.dim)
    reference_output = reference_model(input)

    if data_parallel > 1:
        input_shard_dims = (2, None)  # shard across batch dimension
    else:
        input_shard_dims = (None, -1)  # shard across width dimension

    # DistributedNorm inputs are fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
    tt_input = ttnn.from_torch(
        input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=input_shard_dims, mesh_shape=model_args.cluster_shape),
        memory_config=(
            model_args.get_model_config()["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        ),
    )

    tt_output = tt_model(tt_input, mode=mode)

    # DistributedNorm outputs are replicated across devices
    if data_parallel > 1:
        # Data parallel is not running distributed norm.
        # Data parallel per chip batch runs on dim 0. dim 3 is not utilized.
        output_shard_dims = (2, 3)
    elif model_args.is_galaxy:
        output_shard_dims = (0, 3)
    else:
        output_shard_dims = (3, 0)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=output_shard_dims, mesh_shape=model_args.cluster_shape
        ),
    )
    if tensor_parallel > 1:
        tt_output_torch = tt_output_torch[:1, :, :, :]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("Llama_rms_norm Passed!")
    else:
        logger.warning("Llama_rms_norm Failed!")

    assert passing, f"Llama_rms_norm output does not meet PCC requirement {0.99}."
