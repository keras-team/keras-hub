# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a modified version of `run_xla.py` script in the PyTorch Gemma repo
to ensure proper functionality after porting checkpoints from Keras. Please
run `export_gemma_to_torch_xla.py` prior to running this verification script.

As with the conversion script, ensure that `torch_xla` and the PyTorch
implementation of Gemma are properly installed:

`pip install git+https://github.com/google/gemma_pytorch.git`
`pip install torch_xla`

Note that this verification script can take several minutes to run.
"""

import contextlib
import os
import random
import sys
from typing import List

import gemma.xla_model_parallel as xla_model_parallel
import numpy as np
import torch
import torch.multiprocessing
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from absl import app
from absl import flags
from gemma.config import GemmaConfig
from gemma.config import get_config_for_2b
from gemma.config import get_config_for_7b
from gemma.model_xla import GemmaForCausalLM
from gemma.tokenizer import Tokenizer

"""
Sample usage:

Run the verification script supplying your model size, converted checkpoint file,
vocabulary file, and test prompt.

```
python keras-nlp-gemma/tools/gemma/run_gemma_xla.py \
  --size 2b \
  --checkpoint_file fine_tuned_imdb.ckpt \
  --vocab_file gemma_tokenizer/vocabulary.spm \
  --prompt "Three Billboards"
```

After a delay (a couple minutes if running on CPU), this should produce:
```
======================================
PROMPT: Three Billboards
RESULT: Outside Ebbing, Missouri is a film in the tradition of Hollywood westerns
======================================
```

If running from a preset, instead provide your converted checkpoint file and
the associated preset name:

```
python keras-nlp-gemma/tools/gemma/run_gemma_xla.py \
    --preset gemma_2b_en \
    --checkpoint_file gemma_2b.ckpt \
    --prompt "California is the largest"
```

After a delay (a couple minutes if running on CPU), this should produce:
```
======================================
PROMPT: California is the largest
RESULT:  producer of strawberries in the world, and is a
======================================
```
"""

PAD_TOKEN_ID = -1

FILE_PATH = "gemma.ckpt"
TOKENIZER_DIR = "gemma_tokenizer"

PRESET_MAP = {
    "gemma_2b_en": get_config_for_2b(),
    "gemma_instruct_2b_en": get_config_for_2b(),
    "gemma_7b_en": get_config_for_7b(),
    "gemma_instruct_7b_en": get_config_for_7b(),
}

SIZE_MAP = {
    "2b": get_config_for_2b(),
    "7b": get_config_for_7b(),
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)
flags.DEFINE_string(
    "size",
    None,
    "Size of model. Must be passed if `preset` is not passed. "
    "This should be either `2b` or `7b`.",
)
flags.DEFINE_string(
    "checkpoint_file",
    "gemma.ckpt",
    "A PyTorch checkpoint file containing the converted weights.",
)
flags.DEFINE_string(
    "vocab_file",
    "gemma_tokenizer/vocabulary.spm",
    "The file containing the vocabulary for the tokenizer.",
)
flags.DEFINE_string(
    "prompt",
    "The capital of France is",
    "A test prompt for verifying functionality of the PyTorch Gemma model.",
)


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def generate(
    i: int,
    model_config: GemmaConfig,
    checkpoint_file: str,
    vocab_file: str,
    prompts: List[str],
    output_lens: List[int],
    temperatures: List[float],
    top_ps: List[float],
    top_ks: List[int],
):
    # Set seed from config
    seed = model_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = xm.xla_device()
    xm.set_rng_state(seed, device)

    rank = xla_model_parallel.get_model_parallel_rank()
    world_size = xla_model_parallel.get_model_parallel_world_size()
    if rank > 0:
        sys.stdout = open(os.devnull, "w")

    # Load model with ported weights and place on device
    with _set_default_tensor_type(model_config.get_dtype()):
        model = GemmaForCausalLM(model_config, world_size, rank, device)
        model.load_weights(checkpoint_file)
        model = model.to(device).eval()

    # Create tokenizer with saved Keras tokenizer state
    tokenizer = Tokenizer(vocab_file)

    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    min_prompt_len = min(len(p) for p in prompt_tokens)

    batch_size = len(prompts)
    assert batch_size == len(temperatures)
    assert batch_size == len(top_ps)
    assert batch_size == len(top_ks)
    max_seq_len = max([len(p) + o for p, o in zip(prompt_tokens, output_lens)])
    assert max_seq_len <= model_config.max_position_embeddings
    if model_config.num_key_value_heads < world_size:
        assert world_size % model_config.num_key_value_heads == 0
        n_local_heads = 1
    else:
        assert model_config.num_key_value_heads % world_size == 0
        n_local_heads = model_config.num_key_value_heads // world_size

    # build KV caches
    kv_caches = []
    for _ in range(model_config.num_hidden_layers):
        k_cache = torch.zeros(
            size=(
                batch_size,
                max_seq_len,
                n_local_heads,
                model_config.head_dim,
            ),
            dtype=model_config.get_dtype(),
            device=device,
        )
        v_cache = torch.zeros(
            size=(
                batch_size,
                max_seq_len,
                n_local_heads,
                model_config.head_dim,
            ),
            dtype=model_config.get_dtype(),
            device=device,
        )
        kv_caches.append((k_cache, v_cache))

    # prepare inputs
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
    input_positions_tensor = torch.arange(
        0, min_prompt_len, dtype=torch.int64
    ).to(device)
    mask_tensor = torch.full(
        (1, 1, max_seq_len, max_seq_len), -2.3819763e38
    ).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = torch.FloatTensor(temperatures).to(device)
    top_ps_tensor = torch.FloatTensor(top_ps).to(device)
    top_ks_tensor = torch.LongTensor(top_ks).to(device)
    output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)
    xm.mark_step()

    # Prefill up to min_prompt_len tokens, then treat other prefill as decode and ignore output.
    for i in range(max_seq_len - min_prompt_len):
        next_token_ids = model(
            input_token_ids=input_token_ids_tensor,
            input_positions=input_positions_tensor,
            kv_write_indices=None,
            kv_caches=kv_caches,
            mask=curr_mask_tensor,
            output_positions=output_positions_tensor,
            temperatures=temperatures_tensor,
            top_ps=top_ps_tensor,
            top_ks=top_ks_tensor,
        )
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

    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
        trimmed_output = tokens[
            len(prompt_tokens[i]) : len(prompt_tokens[i]) + output_lens[i]
        ]
        if tokenizer.eos_id in trimmed_output:
            eos_index = trimmed_output.index(tokenizer.eos_id)
            trimmed_output = trimmed_output[:eos_index]
        results.append(tokenizer.decode(trimmed_output))

    for prompt, result in zip(prompts, results):
        print("======================================")
        print(f"PROMPT: {prompt}")
        print(f"RESULT: {result}")
        print("======================================")


def flag_error_handler():
    if not FLAGS.preset and not FLAGS.size:
        raise ValueError(
            "Please pass either a valid Keras preset to `--preset`"
            " or supply a model size (`2b` or `7b`) to `--size`."
        )
    if FLAGS.size and FLAGS.size.lower() not in ["2b", "7b"]:
        raise ValueError(
            "Invalid `size`. Please pass the appropriate size (`2b` or `7b`) "
            "for your model to the `--size` flag."
        )


def main(_):
    flag_error_handler()
    if FLAGS.preset:
        model_config = PRESET_MAP[FLAGS.preset]
    else:
        model_config = SIZE_MAP[FLAGS.size.lower()]
    prompts = [
        FLAGS.prompt,
    ]
    n = len(prompts)
    output_lengths = [10] * n
    temperatures = [0.95] * n
    top_ps = [1.0] * n
    top_ks = [100] * n
    xmp.spawn(
        generate,
        args=(
            model_config,
            FLAGS.checkpoint_file,
            FLAGS.vocab_file,
            prompts,
            output_lengths,
            temperatures,
            top_ps,
            top_ks,
        ),
    )


if __name__ == "__main__":
    app.run(main)
