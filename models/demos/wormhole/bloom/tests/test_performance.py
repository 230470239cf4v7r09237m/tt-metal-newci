# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger
import pytest
from transformers import BloomConfig, BloomForCausalLM, BloomForQuestionAnswering, BloomTokenizerFast

from models.demos.wormhole.bloom.tt import ttnn_optimized_bloom


from models.utility_functions import is_grayskull, is_wormhole_b0, skip_for_grayskull
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters


def get_expected_times_qa(functional_bloom):
    return {
        ttnn_optimized_bloom: (12, 0.85),
    }[functional_bloom]


def get_expected_times_causal_lm(functional_bloom):
    return {
        ttnn_optimized_bloom: (9.0, 7.8),
    }[functional_bloom]


@skip_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("functional_bloom", [ttnn_optimized_bloom])
def test_performance_of_bloom_for_question_answering(
    mesh_device,
    use_program_cache,
    functional_bloom,
    batch_size=8,
    max_length=384,
):
    disable_persistent_kernel_cache()

    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    torch_model = BloomForQuestionAnswering.from_pretrained(model_name).eval()

    num_heads = config.n_head

    question = "What is my name?"
    context = "My name is John."
    inputs = tokenizer(question, context, return_tensors="pt")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=f"ttnn_functional_bloom_for_question_answering",
            initialize_model=lambda: torch_model,
            device=mesh_device,
            custom_preprocessor=ttnn_optimized_bloom.custom_preprocessor,
        )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    num_tokens = input_ids.shape[-1]
    input_ids = input_ids.expand((batch_size, num_tokens))
    attention_mask = attention_mask.expand((batch_size, num_tokens))

    input_ids, alibi, causal_mask = functional_bloom.preprocess_inputs(
        input_ids=input_ids,
        device=mesh_device,
        num_heads=num_heads,
        attention_mask=attention_mask,
        max_length=max_length,
        mesh_mapper=inputs_mesh_mapper,
    )

    durations = []
    for _ in range(2):
        start = time.time()
        tt_output = functional_bloom.bloom_for_question_answering(
            config, input_ids, alibi, causal_mask, parameters=parameters
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()

        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times_qa(functional_bloom)
    prep_perf_report(
        model_name=f"ttnn_{model_name}_optimized",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("functional_bloom", [ttnn_optimized_bloom])
def test_performance_of_causal_lm(mesh_device, use_program_cache, functional_bloom, batch_size=8, max_length=128):
    disable_persistent_kernel_cache()

    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    torch_model = BloomForCausalLM.from_pretrained(model_name).eval()

    num_heads = config.n_head

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=f"ttnn_functional_bloom_for_question_answering",
            initialize_model=lambda: torch_model,
            custom_preprocessor=functional_bloom.custom_preprocessor,
            device=mesh_device,
            # convert_to_ttnn=lambda model, name: name != "lm_head",
        )
    num_heads = config.n_head

    context = "Hello, my dog is cute"
    inputs = tokenizer.encode_plus(context, return_tensors="pt")

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    num_tokens = input_ids.shape[-1]
    input_ids = input_ids.expand((batch_size, num_tokens))
    attention_mask = attention_mask.expand((batch_size, num_tokens))

    input_ids, alibi, causal_mask = functional_bloom.preprocess_inputs(
        input_ids=input_ids,
        device=mesh_device,
        num_heads=num_heads,
        attention_mask=attention_mask,
        max_length=max_length,
        mesh_mapper=inputs_mesh_mapper,
    )

    durations = []
    for _ in range(2):
        start = time.time()
        tt_output = functional_bloom.bloom_for_causal_lm(config, input_ids, alibi, causal_mask, parameters=parameters)
        tt_output = ttnn.from_device(tt_output)

        end = time.time()

        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times_causal_lm(functional_bloom)
    prep_perf_report(
        model_name=f"ttnn_{model_name}_optimized",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size",
    [8],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_bloom_for_question_answering(batch_size, reset_seeds):
    subdir = ""
    num_iterations = 1
    margin = 0.03

    if is_wormhole_b0():
        expected_perf = 49.37

    command = f"pytest tests/ttnn/integration_tests/bloom/test_ttnn_bloom_wh.py::test_bloom_for_question_answering_real"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"tt_bloom{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


@pytest.mark.parametrize(
    "batch_size",
    [8],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_bloom_for_causal_lm(batch_size, reset_seeds):
    subdir = ""
    num_iterations = 1
    margin = 0.03
    if is_wormhole_b0():
        expected_perf = 49.37

    command = f"pytest tests/ttnn/integration_tests/bloom/test_ttnn_bloom_wh.py::test_bloom_for_causal_lm_real"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"tt_bloom{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
