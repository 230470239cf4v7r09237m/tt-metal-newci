# Model Updates

> [!NOTE]
>
> Please refer to the front-page [README](../README.md) for the latest verified release for each model.

## August 12, 2024

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
- Added support for flash decoding

### [Mistral7B](demos/wormhole/mistral7b)
- Updated the demo to support multiple batches of users

### [Mamba-2.8B](demos/wormhole/mamba) 
- Updated the demo to use the full prefill graph instead of processing a single token of the prompt at a time using decode

### [Mixtral7Bx8](demos/t3000/mixtral8x7b)
- Added support for decode with 32K context length using flash decoding
- Fused mixture of experts into a single operation using `ttnn.moe`

## July 29, 2024

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
- Added support for LLaMA 3.1 - 8B
- Runs fast prefill for sequence lengths of up to 512 tokens
- Supports a maximum context length of 8K tokens

### [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Added support for LLaMA 3.1 70B (new scaled rotary position embeddings)
- Prefill and decode now support 8K context length with batch size 16

### [Mistral7B](demos/wormhole/mistral7b)
- Added prefill support for 4K context length, using scaled dot product attention
