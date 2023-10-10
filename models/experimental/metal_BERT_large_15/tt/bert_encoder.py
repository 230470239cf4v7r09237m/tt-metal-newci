# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch


import tt_lib
from models.experimental.metal_BERT_large_15.tt.mha import TtMultiHeadAttentionModel
from models.experimental.metal_BERT_large_15.tt.ffn import TtFeedForwardModel
from tt_lib.utils import pad_weight



class TtBertEncoder(torch.nn.Module):
    def __init__(self, config, encoder_idx, state_dict, device, model_config):
        super().__init__()
        self.device = device
        self.model_config = model_config

        # MHA sub-graph
        self.mha = TtMultiHeadAttentionModel(
            config, encoder_idx, state_dict, device, model_config
        )

        self.attention_output_weight = pad_weight(
            torch.transpose(
                state_dict[
                    f"bert.encoder.layer.{encoder_idx}.attention.output.dense.weight"
                ],
                -2,
                -1,
            )
        )
        self.attention_output_weight = (
            tt_lib.tensor.Tensor(
                self.attention_output_weight.reshape(-1).tolist(),
                self.attention_output_weight.shape,
                model_config["OP11_SELFOUT_WEIGHTS_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            )
            .to(tt_lib.tensor.Layout.TILE)
            .to(device, model_config["OP11_SELFOUT_WEIGHTS_MEMCFG"])
        )
        self.attention_output_bias = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.dense.bias"]
        )
        self.attention_output_bias = (
            tt_lib.tensor.Tensor(
                self.attention_output_bias.reshape(-1).tolist(),
                self.attention_output_bias.shape,
                model_config["OP11_SELFOUT_BIAS_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            )
            .to(tt_lib.tensor.Layout.TILE)
            .to(device, model_config["OP11_SELFOUT_BIAS_MEMCFG"])
        )

        # Weights pre-transposed on host​. No on-the fly transpose of W.
        # self.attention_output_weight = tt_lib.tensor.transpose(
        #     self.attention_output_weight
        # )

        # MHA layernorm
        gamma0 = state_dict[
            f"bert.encoder.layer.{encoder_idx}.attention.output.LayerNorm.weight"
        ]
        beta0 = state_dict[
            f"bert.encoder.layer.{encoder_idx}.attention.output.LayerNorm.bias"
        ]
        mha_gamma = gamma0.reshape(1, 1, -1, 32)
        self.mha_gamma = tt_lib.tensor.Tensor(
            mha_gamma.reshape(-1).tolist(),
            mha_gamma.shape,
            model_config["OP12_LAYERNORM_GAMMA_DTYPE"],
            tt_lib.tensor.Layout.ROW_MAJOR,
        ).to(device, model_config["OP12_LAYERNORM_GAMMA_MEMCFG"])
        mha_beta = beta0.reshape(1, 1, -1, 32)
        self.mha_beta = tt_lib.tensor.Tensor(
            mha_beta.reshape(-1).tolist(),
            mha_beta.shape,
            model_config["OP12_LAYERNORM_BETA_DTYPE"],
            tt_lib.tensor.Layout.ROW_MAJOR,
        ).to(device, model_config["OP12_LAYERNORM_BETA_MEMCFG"])

        # FFN sub-graph
        self.ffn = TtFeedForwardModel(encoder_idx, state_dict, device, model_config)

        # FFN layernorm
        gamma1 = state_dict[f"bert.encoder.layer.{encoder_idx}.output.LayerNorm.weight"]
        beta1 = state_dict[f"bert.encoder.layer.{encoder_idx}.output.LayerNorm.bias"]
        ffn_gamma = gamma1.reshape(1, 1, -1, 32)
        self.ffn_gamma = tt_lib.tensor.Tensor(
            ffn_gamma.reshape(-1).tolist(),
            ffn_gamma.shape,
            model_config["OP15_LAYERNORM_GAMMA_DTYPE"],
            tt_lib.tensor.Layout.ROW_MAJOR,
        ).to(device, model_config["OP15_LAYERNORM_GAMMA_MEMCFG"])
        ffn_beta = beta1.reshape(1, 1, -1, 32)
        self.ffn_beta = tt_lib.tensor.Tensor(
            ffn_beta.reshape(-1).tolist(),
            ffn_beta.shape,
            model_config["OP15_LAYERNORM_BETA_DTYPE"],
            tt_lib.tensor.Layout.ROW_MAJOR,
        ).to(device, model_config["OP15_LAYERNORM_BETA_MEMCFG"])

        self.layer_norm_eps = config.layer_norm_eps

    def op11_mm_plus_bias(
        self, mha_res, attention_output_weight, attention_output_bias
    ):
        mha_out = tt_lib.tensor.bert_large_selfout_matmul(
            mha_res,
            attention_output_weight,
            attention_output_bias,
            output_mem_config=self.model_config["OP11_SELFOUT_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["OP11_SELFOUT_OUTPUT_DTYPE"],
        )
        return mha_out

    def op12_add_layernorm(self, activation, mha_out):
        mha_out_add_and_norm = tt_lib.operations.primary.add_layernorm(
            activation,
            mha_out,
            self.layer_norm_eps,
            self.mha_gamma,
            self.mha_beta,
            output_mem_config=self.model_config["OP12_LAYERNORM_OUTPUT_MEMCFG"],
        )
        return mha_out_add_and_norm

    def op15_add_layernorm(self, mha_out_add_and_norm, ffn_out):
        ffn_out_add_and_norm = tt_lib.operations.primary.add_layernorm(
            mha_out_add_and_norm,
            ffn_out,
            self.layer_norm_eps,
            self.ffn_gamma,
            self.ffn_beta,
            output_mem_config=self.model_config["OP15_LAYERNORM_OUTPUT_MEMCFG"],
        )
        return ffn_out_add_and_norm

    def forward(self, activation, attention_mask=None):
        activation_shape = activation.shape()
        assert activation_shape == [activation_shape[0], 1, 384, 1024]

        # MHA - OP1 - OP10 ------------------------------->
        mha_res = self.mha(activation, attention_mask)
        # Don't deallocate activations here since it is used by more ops

        mha_out = self.op11_mm_plus_bias(
            mha_res, self.attention_output_weight, self.attention_output_bias
        )
        mha_res.deallocate()
        mha_out_add_and_norm = self.op12_add_layernorm(activation, mha_out)
        activation.deallocate()
        mha_out.deallocate()

        # FFN - OP13 - OP14 ----------------------------->
        ffn_out = self.ffn(mha_out_add_and_norm)

        ffn_out_add_and_norm = self.op15_add_layernorm(mha_out_add_and_norm, ffn_out)
        mha_out_add_and_norm.deallocate()
        ffn_out.deallocate()
        return ffn_out_add_and_norm
