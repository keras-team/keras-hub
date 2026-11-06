"""Convert FLUX.2 [klein] from Hugging Face diffusers to a KerasHub preset.

This script builds a KerasHub `Flux2TextToImage` from the diffusers repo with
the library converters (`from_preset("hf://...")` runs `convert_flux2.py` for
the transformer and the VAE, and `convert_qwen3.py` for the text encoder and
the tokenizer), checks it against diffusers' `Flux2KleinPipeline` starting
from a raw prompt, and saves it as a KerasHub preset.

Setup (from the root of a keras-hub checkout):
```shell
pip install -e .
pip install "diffusers>=0.37" transformers accelerate
```

Usage:
```shell
python tools/checkpoint_conversion/convert_flux2_checkpoints.py \
    --preset flux2_klein_4b
```

The Keras backend defaults to torch and can be set with `KERAS_BACKEND`
(e.g. `KERAS_BACKEND=jax`). Both models run in `--dtype`, one after the
other: ~32GB of host RAM at float32, ~16GB at bfloat16.
"""

import gc
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from diffusers import Flux2KleinPipeline
from keras import ops

from keras_hub.src.models.flux2.flux2_text_to_image import Flux2TextToImage

FLAGS = flags.FLAGS

PRESET_MAP = {
    "flux2_klein_4b": "black-forest-labs/FLUX.2-klein-4B",
}

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)
flags.DEFINE_enum(
    "dtype",
    "float32",
    ["float32", "bfloat16"],
    "The dtype both models run in, and the dtype of the saved preset.",
)

PROMPT = "A cat holding a sign that says hello world"
SEED = 42
NUM_STEPS = 4
IMAGE_SIZE = 512

# The maximum relative L2 error of the prompt embeddings. In bfloat16 every
# op rounds, so two correct implementations drift apart.
EMBEDDING_TOLERANCE = {"float32": 1e-4, "bfloat16": 5e-2}
# The maximum mean absolute difference of the generated uint8 images.
IMAGE_TOLERANCE = {"float32": 0.5, "bfloat16": 8.0}


def relative_error(actual, expected):
    actual = np.asarray(actual, "float64")
    expected = np.asarray(expected, "float64")
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def reference_outputs(hf_id, dtype):
    """Run `Flux2KleinPipeline` on `PROMPT` from shared initial noise."""
    pipe = Flux2KleinPipeline.from_pretrained(
        hf_id, torch_dtype=getattr(torch, dtype)
    )
    # The noise `Flux2TextToImage.generate(..., seed=SEED)` starts from.
    latent_size = IMAGE_SIZE // (pipe.vae_scale_factor * 2)
    channels = pipe.transformer.config.in_channels
    noise = keras.random.normal(
        (1, latent_size, latent_size, channels), dtype="float32", seed=SEED
    )
    noise = torch.from_numpy(ops.convert_to_numpy(noise)).permute(0, 3, 1, 2)
    text = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = pipe.tokenizer(
        text, padding="max_length", truncation=True, max_length=512
    )["input_ids"]
    with torch.no_grad():
        embeddings, _ = pipe.encode_prompt(PROMPT)
        image = pipe(
            prompt=PROMPT,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=NUM_STEPS,
            latents=noise.to(pipe.transformer.dtype),
            output_type="np",
        ).images[0]
    outputs = {
        "token_ids": np.asarray(token_ids),
        "embeddings": embeddings[0].float().numpy(),
        # Quantized as `Flux2TextToImage.generate()` does.
        "image": np.round(np.clip(image, 0, 1) * 255).astype("uint8"),
        "num_params": sum(p.numel() for p in pipe.transformer.parameters()),
    }
    del pipe
    gc.collect()
    return outputs


def check_parameter_count(keras_model, reference):
    keras_params = keras_model.backbone.count_params()
    print(f"diffusers transformer params: {reference['num_params']:>14,}")
    print(f"KerasHub backbone params:     {keras_params:>14,}")
    return keras_params == reference["num_params"]


def check_tokenizer(keras_model, reference):
    """The raw prompt must give the same 512 token ids."""
    inputs = keras_model.preprocessor.generate_preprocess([PROMPT])
    token_ids = ops.convert_to_numpy(inputs["token_ids"])[0]
    num_tokens = int(ops.convert_to_numpy(inputs["padding_mask"]).sum())
    print(f"Prompt tokens: {num_tokens}, padded to {len(token_ids)}")
    return np.array_equal(token_ids, reference["token_ids"])


def check_prompt_embeddings(keras_model, reference, dtype):
    """Raw prompt -> preprocessor -> text encoder, padding included."""
    inputs = keras_model.preprocessor.generate_preprocess([PROMPT])
    embeddings, _ = keras_model.encode_text_step(inputs)
    actual = ops.convert_to_numpy(ops.cast(embeddings, "float32"))[0]
    expected = reference["embeddings"]
    real = ops.convert_to_numpy(inputs["padding_mask"])[0].astype(bool)
    for name, positions in (("real", real), ("padded", ~real)):
        error = relative_error(actual[positions], expected[positions])
        print(f"{name:>6} positions: relative L2 {error:.2e}")
    error = relative_error(actual, expected)
    max_abs = np.abs(actual - expected).max()
    print(f"   all positions: relative L2 {error:.2e}, max abs {max_abs:.2e}")
    return error <= EMBEDDING_TOLERANCE[dtype]


def check_generate(image, reference, dtype):
    """`generate()` against `Flux2KleinPipeline`, from the same noise."""
    diff = np.abs(image.astype("int32") - reference["image"].astype("int32"))
    print(f"Image shape: {image.shape}")
    print(f"Mean abs pixel diff: {diff.mean():.3f} (max {diff.max()})")
    return diff.mean() <= IMAGE_TOLERANCE[dtype]


def check_saved_preset(image, preset):
    """Reload the saved preset and compare its generated image."""
    # Use a path, a bare name would resolve to the registered Kaggle preset.
    reloaded = Flux2TextToImage.from_preset(os.path.abspath(preset))
    restored = reloaded.generate(PROMPT, num_steps=NUM_STEPS, seed=SEED)
    diff = np.abs(image.astype("int32") - restored.astype("int32")).max()
    print(f"Max abs pixel diff after reload: {diff}")
    # GPU kernels are not bitwise reproducible across compiled models.
    return diff <= 1


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {preset}. Must be one of "
            f"{','.join(PRESET_MAP.keys())}"
        )
    hf_id = PRESET_MAP[preset]
    dtype = FLAGS.dtype
    print(f"Converting {hf_id} -> {preset} ({keras.config.backend()}, {dtype})")

    print("\n--- Running Flux2KleinPipeline ---")
    reference = reference_outputs(hf_id, dtype)

    print("\n--- Converting ---")
    keras_model = Flux2TextToImage.from_preset(
        f"hf://{hf_id}", image_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=dtype
    )
    image = keras_model.generate(PROMPT, num_steps=NUM_STEPS, seed=SEED)

    checks = {
        "Parameter count": lambda: check_parameter_count(
            keras_model, reference
        ),
        "Tokenizer": lambda: check_tokenizer(keras_model, reference),
        "Prompt embeddings": lambda: check_prompt_embeddings(
            keras_model, reference, dtype
        ),
        "generate()": lambda: check_generate(image, reference, dtype),
    }
    results = {}
    for name, check in checks.items():
        print(f"\n--- {name} ---")
        results[name] = bool(check())

    print("\n--- Summary ---")
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if not all(results.values()):
        raise RuntimeError("Verification failed. The preset was not saved.")

    print("\n--- Saved preset ---")
    keras_model.save_to_preset(preset)
    # Free the converted model before reloading the preset, so that only one
    # copy of the weights is in memory at a time.
    keras_model = None
    gc.collect()
    if not check_saved_preset(image, preset):
        raise RuntimeError("The saved preset does not reproduce the outputs.")
    print(f"Saved and verified preset: ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
