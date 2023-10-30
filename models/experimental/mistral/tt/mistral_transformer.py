# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import torch
import torch.nn as nn
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.tt.mistral_transformer_block import TtTransformerBlock
from models.experimental.mistral.tt.mistral_rms_norm import TtRMSNorm
from models.experimental.mistral.mistral_helper_funcs import Linear as TtLinear
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from typing import Optional


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


class TtTransformer(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        device=None,
        state_dict=None,
        base_address=None,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address
        assert self.vocab_size > 0

        embedding_weights = state_dict["tok_embeddings.weight"]
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim, _weight=embedding_weights)

        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    args=args, state_dict=self.state_dict, base_address=f"layers.{i}.", device=self.device
                )
                for i in range(args.n_layers)
            ]
        )
        self.norm = TtRMSNorm(args.dim, base_address=f"norm.", state_dict=state_dict, device=device, eps=args.norm_eps)
        self.output_weight = torch_to_tt_tensor_rm(state_dict["output.weight"], self.device)
        self.output = TtLinear(
            args.dim,
            args.vocab_size,
            self.output_weight,
        )
        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        h = self.tok_embeddings(input_ids)
        input_ids = torch_to_tt_tensor_rm(input_ids, self.device, put_on_device=False)
        freqs_cis = self.freqs_cis[positions]
        mask: Optional[torch.Tensor] = None
        if input_ids.shape()[-1] > 1:
            seqlen = input_ids.shape()[-1]
            tensor = tt_lib.tensor.full(
                (1, 1, seqlen, seqlen),
                fill_value=1.0,
            )
            diagonal = 0

            mask = tt_lib.tensor.tril(tensor, diagonal)
            # make the mask banded to account for sliding window
            diagonal = -self.args.sliding_window
            mask = tt_lib.tensor.triu(mask, diagonal)
            mask = tt_lib.tensor.log(mask)
            mask = tt_to_torch_tensor(mask)

        positions = torch_to_tt_tensor_rm(positions, self.device, put_on_device=False)
        h = torch_to_tt_tensor_rm(h, self.device, put_on_device=False)
        for layer in self.layers:
            h = layer(h, freqs_cis, positions, mask)
        return self.output(self.norm(h))
