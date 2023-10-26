# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json
from pathlib import Path
from models.experimental.mistral.tt.mistral_transformer import TtTransformer
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.demo.model import Transformer
from models.experimental.mistral.demo.tokenizer import Tokenizer
from models.utility_functions import tt_to_torch_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.95),),
)
def test_mistral_transformer_inference(pcc, model_location_generator, device, reset_seeds):
    prompts = [
        "This is a sample text for single layer execution ",
    ]

    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    tokenizer = Tokenizer(str(Path(mistral_path) / "tokenizer.model"))
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    base_address = f""
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    state_dict = {
        k[0:]: v
        for k, v in state_dict.items()
        if (
            k.startswith("layers.0.")
            or k.startswith("tok_embeddings")
            or k.startswith("norm.weight")
            or k.startswith("output.weight")
        )
    }

    model_args.max_batch_size = 1
    model_args.n_layers = 1
    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)

    tt_model = TtTransformer(
        args=model_args,
        state_dict=state_dict,
        device=device,
        base_address=base_address,
    )

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="cpu")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)  # .to(input_tokens)

    positions = torch.arange(0, min_prompt_len)

    tt_output = tt_model.forward(input_tokens[:, :min_prompt_len], positions)

    reference_output = reference_model(input_tokens[:, :min_prompt_len], positions)

    tt_output_torch = tt_to_torch_tensor(tt_output).squeeze(0).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_Transformer Passed!")
    else:
        logger.warning("Mistral_Transformer Failed!")

    assert passing, f"Mistral_Transformer output does not meet PCC requirement {pcc}."
