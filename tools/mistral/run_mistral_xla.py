"""
This script loads a converted Mistral checkpoint (produced by
`export_mistral_to_torch_xla.py`) onto an XLA device and runs text generation.

Please ensure that `mistral-inference` and `torch_xla` are installed:

`pip install mistral-inference`
`pip install torch_xla`

Note that this script can take several minutes to run on CPU.

Sample usage:

Run with the output directory produced by the export script:
```
python tools/mistral/run_mistral_xla.py \
  --checkpoint_dir mistral_xla \
  --prompt "Inception is about"
```

This should produce something like:
```
======================================
PROMPT: Inception is about
RESULT: a thief who steals corporate secrets through the use of
======================================
```
"""

import json
import os
import sys
from typing import List

import sentencepiece as spm
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from absl import app
from absl import flags
from mistral_inference.args import ModelArgs
from mistral_inference.model import Transformer

"""
Sample usage:

```
python tools/mistral/run_mistral_xla.py \
    --checkpoint_dir mistral_xla \
    --prompt "The capital of France is"
```
"""

PAD_TOKEN_ID = -1

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint_dir",
    "mistral_xla",
    "Directory containing `mistral.ckpt`, `params.json`, and `tokenizer.model`.",
)
flags.DEFINE_string(
    "prompt",
    "The capital of France is",
    "A test prompt for verifying functionality of the converted model.",
)
flags.DEFINE_integer(
    "output_len",
    10,
    "Number of new tokens to generate.",
)
flags.DEFINE_float(
    "temperature",
    0.0,
    "Sampling temperature. Use 0.0 for greedy decoding.",
)


def generate(
    index: int,
    checkpoint_dir: str,
    prompts: List[str],
    output_len: int,
    temperature: float,
):
    """Generate text on an XLA device from a converted Mistral checkpoint."""
    device = xm.xla_device()

    # Suppress output from non-primary workers in multi-process mode
    if index > 0:
        sys.stdout = open(os.devnull, "w")

    # ------------------------------------------------------------------ #
    # 1. Load model config
    # ------------------------------------------------------------------ #
    params_path = os.path.join(checkpoint_dir, "params.json")
    with open(params_path) as f:
        params = json.load(f)
    model_args = ModelArgs(**params)

    # ------------------------------------------------------------------ #
    # 2. Create model on XLA device and load converted weights
    # ------------------------------------------------------------------ #
    model = Transformer(model_args).to(device).eval()

    checkpoint_path = os.path.join(checkpoint_dir, "mistral.ckpt")
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    xm.mark_step()

    # ------------------------------------------------------------------ #
    # 3. Tokenise prompts
    # ------------------------------------------------------------------ #
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(os.path.join(checkpoint_dir, "tokenizer.model"))

    prompt_tokens = [tokenizer.Encode(p, out_type=int) for p in prompts]
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_seq_len = max(len(p) + output_len for p in prompt_tokens)
    batch_size = len(prompts)

    assert max_seq_len <= model_args.max_position_embeddings, (
        f"max_seq_len {max_seq_len} exceeds model max "
        f"{model_args.max_position_embeddings}"
    )

    n_kv_heads = model_args.n_kv_heads or model_args.n_heads
    head_dim = model_args.head_dim or (model_args.dim // model_args.n_heads)

    # ------------------------------------------------------------------ #
    # 4. Build initial token tensors and KV caches
    # ------------------------------------------------------------------ #
    token_ids_tensor = torch.full(
        (batch_size, max_seq_len), PAD_TOKEN_ID, dtype=torch.int64
    )
    input_token_ids_tensor = torch.full(
        (batch_size, min_prompt_len), PAD_TOKEN_ID, dtype=torch.int64
    )
    for i, p in enumerate(prompt_tokens):
        token_ids_tensor[i, : len(p)] = torch.tensor(p)
        input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
            p[:min_prompt_len]
        )

    token_ids_tensor = token_ids_tensor.to(device)
    prompt_mask_tensor = token_ids_tensor != PAD_TOKEN_ID
    input_token_ids_tensor = input_token_ids_tensor.to(device)

    # Causal mask: −∞ above the diagonal, 0 on/below
    mask_tensor = torch.full(
        (1, 1, max_seq_len, max_seq_len), float("-inf")
    ).to(torch.float32)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)

    input_positions_tensor = torch.arange(
        0, min_prompt_len, dtype=torch.int64
    ).to(device)
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)

    # KV cache buffers
    kv_caches = []
    for _ in range(model_args.n_layers):
        k_cache = torch.zeros(
            (batch_size, max_seq_len, n_kv_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        v_cache = torch.zeros(
            (batch_size, max_seq_len, n_kv_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        kv_caches.append((k_cache, v_cache))

    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    xm.mark_step()

    # ------------------------------------------------------------------ #
    # 5. Auto-regressive generation loop
    # ------------------------------------------------------------------ #
    temperatures_tensor = torch.FloatTensor([temperature] * batch_size).to(
        device
    )

    for _ in range(max_seq_len - min_prompt_len):
        logits = model(
            input_token_ids=input_token_ids_tensor,
            input_positions=input_positions_tensor,
            kv_write_indices=None,
            kv_caches=kv_caches,
            mask=curr_mask_tensor,
            output_positions=output_positions_tensor,
            temperatures=temperatures_tensor,
            top_ps=torch.FloatTensor([1.0] * batch_size).to(device),
            top_ks=torch.LongTensor([1] * batch_size).to(device),
        )

        # Pick next token (logits already argmax-ed / sampled inside model)
        next_token_ids = logits

        # If this position was already part of the prompt, keep the prompt token
        curr_prompt_mask = prompt_mask_tensor.index_select(
            1, output_index
        ).squeeze(dim=1)
        curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(
            dim=1
        )
        output_token_ids = torch.where(
            curr_prompt_mask, curr_token_ids, next_token_ids
        ).unsqueeze(dim=1)
        token_ids_tensor.index_copy_(1, output_index, output_token_ids)

        input_token_ids_tensor = output_token_ids
        input_positions_tensor = output_index
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(device)
        output_index = output_index + 1
        xm.mark_step()

    # ------------------------------------------------------------------ #
    # 6. Decode and print
    # ------------------------------------------------------------------ #
    eos_id = tokenizer.eos_id()
    token_ids = token_ids_tensor.tolist()
    for prompt, tokens in zip(prompts, token_ids):
        # Trim to the generated portion only
        generated = tokens[len(tokenizer.Encode(prompt, out_type=int)):]
        generated = generated[:output_len]
        if eos_id in generated:
            generated = generated[: generated.index(eos_id)]
        result = tokenizer.Decode(generated)

        print("=" * 38)
        print(f"PROMPT: {prompt}")
        print(f"RESULT: {result}")
        print("=" * 38)


def flag_error_handler():
    checkpoint_dir = FLAGS.checkpoint_dir
    for required in ("mistral.ckpt", "params.json", "tokenizer.model"):
        path = os.path.join(checkpoint_dir, required)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: `{path}`. "
                f"Please run `export_mistral_to_torch_xla.py` first."
            )


def main(_):
    flag_error_handler()
    xmp.spawn(
        generate,
        args=(
            FLAGS.checkpoint_dir,
            [FLAGS.prompt],
            FLAGS.output_len,
            FLAGS.temperature,
        ),
    )


if __name__ == "__main__":
    app.run(main)
