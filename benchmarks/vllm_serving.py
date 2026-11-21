"""Benchmark KerasHub presets served through vLLM.

Measures output token throughput for one of three configurations, all serving
the same weights:

    keras_hub       `CausalLM.generate()`, static full-batch decoding.
    vllm_native     vLLM's own model implementation, loaded from the
                    equivalent Hugging Face checkpoint.
    keras_hub_vllm  The KerasHub integration, `keras_hub.vllm.KerasHubLLM`.

Keras reads its backend at import, so set these before running. The last one
is required: vLLM otherwise forks a worker process, which deadlocks against an
already-initialized JAX.

```
export KERAS_BACKEND=jax
export KERAS_NNX_ENABLED=true
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

Run one configuration per invocation:

```
python3 benchmarks/vllm_serving.py \
    --config keras_hub_vllm \
    --preset gemma3_instruct_1b
```

One config per run. A vLLM engine holds its memory until the process exits.

Runs 32 and 512 word prompts at 1, 32 and 64 concurrent requests, 128
generated tokens each, greedy. One warmup pass then 20 timed passes per cell.
Throughput counts generated tokens only.
"""

import time

from absl import app
from absl import flags

import keras_hub

# The presets served by the merged integration, mapped to the Hugging Face
# checkpoint holding the same weights for the vllm_native configuration.
PRESETS = {
    "gpt2_base_en": "openai-community/gpt2",
    "gpt2_large_en": "openai-community/gpt2-large",
    "qwen2.5_coder_0.5b": "Qwen/Qwen2.5-Coder-0.5B",
    "llama3.2_instruct_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma_2b_en": "google/gemma-2b",
    "gemma2_2b_en": "google/gemma-2-2b",
    "gemma3_instruct_1b": "google/gemma-3-1b-it",
}

CONFIGS = ("keras_hub", "vllm_native", "keras_hub_vllm")

INPUT_WORDS = (32, 512)
CONCURRENCY = (1, 32, 64)
OUTPUT_TOKENS = 128
WARMUP_RUNS = 1
TIMED_RUNS = 20
MAX_MODEL_LEN = 1024
# Caps how much prefill the engine batches, and so how many shapes it compiles
# at startup.
MAX_NUM_BATCHED_TOKENS = 512
# Matches the highest concurrency measured, so no request waits on a slot.
MAX_NUM_SEQS = 64
DTYPE = "bfloat16"

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "config",
    "keras_hub_vllm",
    CONFIGS,
    "Which configuration to measure.",
)
flags.DEFINE_enum(
    "preset",
    "gemma3_instruct_1b",
    list(PRESETS),
    "Which preset to measure.",
)
flags.DEFINE_string(
    "output",
    None,
    "CSV output path. Defaults to <config>_<preset>.csv.",
)


def build_prompt(word_count):
    """Returns a prompt of the requested length in words."""
    words = "The future of artificial intelligence is".split()
    return " ".join((words * (word_count // len(words) + 1))[:word_count])


def keras_hub_generate(preset):
    """Returns generate and count functions for `CausalLM.generate()`."""
    from keras import ops

    model = keras_hub.models.CausalLM.from_preset(preset, dtype=DTYPE)
    # `CausalLM.compile` defaults to top_k, which is stochastic. The vLLM
    # configurations use temperature=0.0, so this has to be greedy to match.
    model.compile(sampler="greedy")
    tokenizer = model.preprocessor.tokenizer

    # The prompt length comes from generate_preprocess, which counts the <bos>
    # the tokenizer alone would miss.
    max_lengths = {}
    for word_count in INPUT_WORDS:
        prompt = build_prompt(word_count)
        preprocessed = model.preprocessor.generate_preprocess([prompt])
        prompt_tokens = int(ops.sum(preprocessed["padding_mask"][0]))
        max_lengths[prompt] = prompt_tokens + OUTPUT_TOKENS

    def generate(prompts):
        return model.generate(
            prompts,
            max_length=max_lengths[prompts[0]],
            strip_prompt=True,
        )

    def count(outputs):
        return sum(len(tokenizer(output)) for output in outputs)

    return generate, count


def vllm_generate(model, keras_hub_integration):
    """Returns generate and count functions for a vLLM engine."""
    from vllm import SamplingParams

    if keras_hub_integration:
        from keras_hub.vllm import KerasHubLLM

        engine = KerasHubLLM(
            f"keras_hub:{model}",
            dtype=DTYPE,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
        )
    else:
        from vllm import LLM

        engine = LLM(
            model=model,
            dtype=DTYPE,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
        )

    params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS)

    def generate(prompts):
        return engine.generate(prompts, params)

    def count(outputs):
        return sum(len(output.outputs[0].token_ids) for output in outputs)

    return generate, count


def measure(generate, count, prompts):
    """Returns tokens generated over time taken. Only generation is timed."""
    for _ in range(WARMUP_RUNS):
        generate(prompts)

    total_tokens = 0
    total_time = 0.0
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        outputs = generate(prompts)
        total_time += time.perf_counter() - start
        total_tokens += count(outputs)
    return total_tokens / total_time


def main(_):
    preset = FLAGS.preset
    config = FLAGS.config

    if config == "keras_hub":
        generate, count = keras_hub_generate(preset)
    elif config == "vllm_native":
        generate, count = vllm_generate(
            PRESETS[preset], keras_hub_integration=False
        )
    else:
        generate, count = vllm_generate(preset, keras_hub_integration=True)

    path = FLAGS.output or f"{config}_{preset}.csv"
    with open(path, "w") as results:
        results.write(
            "preset,config,input_words,concurrency,tokens_per_second\n"
        )
        for input_words in INPUT_WORDS:
            prompt = build_prompt(input_words)
            for concurrency in CONCURRENCY:
                throughput = measure(generate, count, [prompt] * concurrency)
                print(
                    f"{preset} {config} input_words={input_words} "
                    f"concurrency={concurrency} "
                    f"{throughput:.1f} tokens/s"
                )
                results.write(
                    f"{preset},{config},{input_words},{concurrency},"
                    f"{throughput:.2f}\n"
                )
                results.flush()

    print(f"Wrote {path}")


if __name__ == "__main__":
    app.run(main)
