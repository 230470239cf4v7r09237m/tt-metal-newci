# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Optional

import torch

import ttnn

from ..common.attention_mask_functions import get_extended_attention_mask, invert_attention_mask


def t5_layer_norm(hidden_states, *, weight, eps=1e-6):
    # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
    # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
    # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
    # half-precision inputs is done in fp32

    # import tt_lib as ttl

    # original_shape = tuple(hidden_states.shape)
    # hidden_states = ttnn.core._reshape_to_4D(hidden_states)

    # ttl_hidden_states = hidden_states._tensor
    # ttl_weight = weight._tensor
    # ttl_hidden_states = ttl.tensor.rmsnorm(ttl_hidden_states, eps, ttl_weight)

    # hidden_states = ttnn.Tensor(ttl_hidden_states)
    # hidden_states = ttnn.reshape(hidden_states, original_shape)

    # return hidden_states

    original_shape = tuple(hidden_states.shape)
    hidden_states = ttnn.core._reshape_to_4D(hidden_states)

    squared_hidden_states = ttnn.pow(hidden_states, 2)
    averaged_squared_hidden_states = ttnn.mean(
        squared_hidden_states,
        dim=-1,
        keepdim=True,
    )

    variance = averaged_squared_hidden_states + eps
    std = ttnn.rsqrt(variance)

    hidden_states = hidden_states * std
    hidden_states = hidden_states * weight
    hidden_states = ttnn.reshape(hidden_states, original_shape)

    return hidden_states


def t5_dense_gated_act_dense(hidden_states, parameters):
    hidden_gelu = hidden_states @ parameters.wi_0.weight
    hidden_gelu = ttnn.gelu(hidden_gelu)
    hidden_linear = hidden_states @ parameters.wi_1.weight
    hidden_states = hidden_gelu * hidden_linear

    hidden_states = hidden_states @ parameters.wo.weight
    return hidden_states


def t5_layer_ff(hidden_states, parameters):
    forwarded_states = t5_layer_norm(hidden_states, weight=parameters.layer_norm.weight, eps=1e-6)
    forwarded_states = t5_dense_gated_act_dense(forwarded_states, parameters.DenseReluDense)
    hidden_states = hidden_states + forwarded_states
    return hidden_states


def t5_attention(
    hidden_states,
    key_value_states=None,
    mask=None,
    layer_head_mask=None,
    *,
    parameters,
    num_heads,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length, _ = hidden_states.shape

    def shape(states, num_heads, head_size, is_key=False):
        """projection"""
        states = ttnn.reshape(states, (batch_size, seq_length, num_heads, head_size))
        if is_key:
            states = ttnn.permute(states, (0, 2, 3, 1))
        else:
            states = ttnn.permute(states, (0, 2, 1, 3))
        return states

    def unshape(states, hidden_size):
        """reshape"""
        states = ttnn.permute(states, (0, 2, 1, 3))
        states = ttnn.reshape(states, (batch_size, seq_length, hidden_size))
        return states

    def project(hidden_states, weight, is_key=False):
        hidden_size = weight.shape[-1]
        head_size = hidden_size // num_heads
        """projects hidden states correctly to key/query states"""
        # self-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(hidden_states @ weight, num_heads, head_size, is_key=is_key)
        return hidden_states

    # get query states
    hidden_size = parameters.q.weight.shape[-1]
    query_states = project(hidden_states, parameters.q.weight)  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states if key_value_states is None else key_value_states,
        parameters.k.weight,
        is_key=True,
    )
    value_states = project(
        hidden_states if key_value_states is None else key_value_states,
        parameters.v.weight,
    )

    # compute scores
    scores = ttnn.matmul(query_states, key_states)
    if mask is not None:
        scores += mask

    attn_weights = ttnn.softmax(scores, dim=-1)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(ttnn.matmul(attn_weights, value_states), hidden_size)  # (batch_size, seq_length, dim)
    attn_output = attn_output @ parameters.o.weight

    return attn_output


def t5_layer_self_attention(
    hidden_states,
    attention_mask=None,
    *,
    parameters,
    num_heads,
):
    normed_hidden_states = t5_layer_norm(hidden_states, weight=parameters.layer_norm.weight, eps=1e-06)
    attention_output = t5_attention(
        normed_hidden_states,
        mask=attention_mask,
        parameters=parameters.SelfAttention,
        num_heads=num_heads,
    )
    hidden_states = hidden_states + attention_output
    return hidden_states


def t5_layer_cross_attention(hidden_states, key_value_states, attention_mask=None, *, parameters, num_heads):
    normed_hidden_states = t5_layer_norm(hidden_states, weight=parameters.layer_norm.weight, eps=1e-06)
    attention_output = t5_attention(
        normed_hidden_states,
        key_value_states,
        mask=attention_mask,
        parameters=parameters.EncDecAttention,
        num_heads=num_heads,
    )
    layer_output = hidden_states + attention_output
    return layer_output


def t5_block(
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    *,
    parameters,
    num_heads,
):
    hidden_states = t5_layer_self_attention(
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters.layer[0],
        num_heads=num_heads,
    )

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    do_cross_attention = encoder_hidden_states is not None
    if do_cross_attention:
        hidden_states = t5_layer_cross_attention(
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            parameters=parameters.layer[1],
            num_heads=num_heads,
        )

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    # Apply Feed Forward layer
    hidden_states = t5_layer_ff(hidden_states, parameters.layer[-1])

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    return hidden_states  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


def t5_stack(
    input_ids,
    shared_embedding_weight,
    encoder_hidden_states=None,
    *,
    parameters,
    num_heads,
):
    input_shape = tuple(input_ids.shape)

    hidden_states = ttnn.embedding(input_ids, shared_embedding_weight)

    attention_mask = create_attention_mask(
        input_shape, num_heads, input_ids.device, is_decoder=encoder_hidden_states is not None
    )
    if encoder_hidden_states is not None:
        encoder_attention_mask = create_encoder_attention_mask(input_shape, num_heads, input_ids.device)
    else:
        encoder_attention_mask = None

    for block_parameters in parameters.block:
        hidden_states = t5_block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            parameters=block_parameters,
            num_heads=num_heads,
        )

    hidden_states = t5_layer_norm(hidden_states, weight=parameters.final_layer_norm.weight, eps=1e-06)

    return hidden_states


def t5_for_conditional_generation(
    input_ids: Optional[torch.LongTensor],
    decoder_input_ids: Optional[torch.LongTensor],
    parameters,
    *,
    num_heads,
) -> torch.FloatTensor:
    # Encode
    hidden_states = t5_stack(
        input_ids=input_ids,
        shared_embedding_weight=parameters.shared.weight,
        parameters=parameters.encoder,
        num_heads=num_heads,
    )

    # Decode
    sequence_output = t5_stack(
        input_ids=decoder_input_ids,
        encoder_hidden_states=hidden_states,
        shared_embedding_weight=parameters.shared.weight,
        parameters=parameters.decoder,
        num_heads=num_heads,
    )

    lm_logits = sequence_output @ parameters.lm_head.weight

    return lm_logits


@functools.lru_cache
def create_attention_mask(input_shape, num_heads, device, is_decoder):
    batch_size, seq_length = input_shape

    attention_mask = torch.ones(batch_size, seq_length)

    extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, is_decoder=is_decoder)

    extended_attention_mask = extended_attention_mask.expand((-1, num_heads, seq_length, -1))
    extended_attention_mask = ttnn.from_torch(extended_attention_mask)
    extended_attention_mask = ttnn.to_layout(extended_attention_mask, ttnn.TILE_LAYOUT)
    extended_attention_mask = ttnn.to_device(extended_attention_mask, device)
    return extended_attention_mask


@functools.lru_cache
def create_encoder_attention_mask(input_shape, num_heads, device):
    batch_size, seq_length = input_shape

    encoder_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)

    encoder_extended_attention_mask = encoder_extended_attention_mask.expand((-1, num_heads, seq_length, -1))
    encoder_extended_attention_mask = ttnn.from_torch(encoder_extended_attention_mask)
    encoder_extended_attention_mask = ttnn.to_layout(encoder_extended_attention_mask, ttnn.TILE_LAYOUT)
    encoder_extended_attention_mask = ttnn.to_device(encoder_extended_attention_mask, device)
    return encoder_extended_attention_mask


def custom_preprocessor(model, name):
    import transformers
    from ttnn.model_preprocessing import preprocess_layernorm_parameter

    if isinstance(model, transformers.models.t5.modeling_t5.T5LayerNorm):
        return {"weight": preprocess_layernorm_parameter(model.weight, dtype=ttnn.bfloat16)}
    return {}
