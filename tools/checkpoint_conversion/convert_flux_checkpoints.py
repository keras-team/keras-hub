"""Convert FLUX.1 [schnell] from Hugging Face diffusers to a KerasHub preset.

FLUX.1 generation is not supported yet (the CLIP-L and T5-XXL text encoders
and the VAE are not ported), so this script converts the transformer. It
builds a KerasHub `FluxBackbone` from the diffusers repo with the library
converter (`from_preset("hf://...")` runs `convert_flux.py`), checks it
against diffusers' `FluxTransformer2DModel` on the inputs `FluxPipeline`
builds from a raw prompt, and saves it as a KerasHub preset.

FLUX.1 [schnell] is gated on the Hub: accept its license on the model page
and log in with `huggingface-cli login` first.

Setup (from the root of a keras-hub checkout):
```shell
pip install -e .
pip install "diffusers>=0.32" transformers accelerate sentencepiece
```

Usage:
```shell
python tools/checkpoint_conversion/convert_flux_checkpoints.py \
    --preset flux1_schnell
```

The Keras backend defaults to torch and can be set with `KERAS_BACKEND`
(e.g. `KERAS_BACKEND=jax`). Both transformers run in `--dtype`, one after
the other: ~48GB of host RAM at float32, ~24GB at bfloat16.
"""

import gc
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel

from keras_hub.src.models.flux.flux_backbone import FluxBackbone

FLAGS = flags.FLAGS

PRESET_MAP = {
    "flux1_schnell": "black-forest-labs/FLUX.1-schnell",
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
IMAGE_SIZE = 512
# The first step of the schnell schedule (pure noise) and a later one.
TIMESTEPS = (1.0, 0.5)

# The maximum relative L2 error of the transformer output. In bfloat16
# every op rounds, so two correct implementations drift apart.
TOLERANCE = {"float32": 1e-4, "bfloat16": 5e-2}


def relative_error(actual, expected):
    actual = np.asarray(actual, "float64")
    expected = np.asarray(expected, "float64")
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def pipeline_inputs(hf_id, dtype):
    """The transformer inputs `FluxPipeline` builds from `PROMPT`."""
    torch_dtype = getattr(torch, dtype)
    # The text encoders only; the transformer is loaded on its own below.
    pipe = FluxPipeline.from_pretrained(
        hf_id, transformer=None, vae=None, torch_dtype=torch_dtype
    )
    config = FluxTransformer2DModel.load_config(hf_id, subfolder="transformer")
    with torch.no_grad():
        text, pooled, text_ids = pipe.encode_prompt(
            PROMPT, prompt_2=None, max_sequence_length=256
        )
        latents, image_ids = pipe.prepare_latents(
            1,
            config["in_channels"] // 4,
            IMAGE_SIZE,
            IMAGE_SIZE,
            torch_dtype,
            "cpu",
            torch.Generator().manual_seed(SEED),
        )
    del pipe
    gc.collect()

    def to_numpy(x):
        return np.ascontiguousarray(x.float().numpy(), "float32")

    return {
        "image": to_numpy(latents),
        "image_ids": to_numpy(image_ids)[None],
        "text": to_numpy(text),
        "text_ids": to_numpy(text_ids)[None],
        "y": to_numpy(pooled),
    }


def reference_outputs(hf_id, dtype, inputs):
    """`FluxTransformer2DModel` outputs at every timestep of `TIMESTEPS`."""
    torch_dtype = getattr(torch, dtype)
    transformer = FluxTransformer2DModel.from_pretrained(
        hf_id, subfolder="transformer", torch_dtype=torch_dtype
    ).eval()

    def tensor(name):
        return torch.from_numpy(inputs[name]).to(torch_dtype)

    outputs = []
    with torch.no_grad():
        for timestep in TIMESTEPS:
            output = transformer(
                hidden_states=tensor("image"),
                encoder_hidden_states=tensor("text"),
                pooled_projections=tensor("y"),
                timestep=torch.tensor([timestep], dtype=torch_dtype),
                # Position ids stay float32, as in the pipeline.
                img_ids=torch.from_numpy(inputs["image_ids"][0]),
                txt_ids=torch.from_numpy(inputs["text_ids"][0]),
                return_dict=False,
            )[0]
            outputs.append(output.float().numpy())
    num_params = sum(p.numel() for p in transformer.parameters())
    del transformer
    gc.collect()
    return outputs, num_params


def keras_outputs(backbone, inputs):
    return [
        np.asarray(
            backbone.predict(
                {**inputs, "timesteps": np.array([t], "float32")}, verbose=0
            ),
            "float32",
        )
        for t in TIMESTEPS
    ]


def check_parameter_count(backbone, reference_params):
    keras_params = backbone.count_params()
    print(f"diffusers transformer params: {reference_params:>16,}")
    print(f"KerasHub backbone params:     {keras_params:>16,}")
    return keras_params == reference_params


def check_outputs(outputs, expected, dtype):
    """Compare the outputs on the pipeline's real inputs."""
    passed = True
    for timestep, actual, want in zip(TIMESTEPS, outputs, expected):
        error = relative_error(actual, want)
        max_abs = np.abs(actual - want).max()
        print(
            f"timestep {timestep}: relative L2 {error:.2e}, "
            f"max abs {max_abs:.2e}"
        )
        passed &= error <= TOLERANCE[dtype]
    return passed


def check_saved_preset(outputs, inputs, preset):
    """Reload the saved preset and compare its outputs."""
    # Use a path, a bare name would resolve to the registered Kaggle preset.
    reloaded = FluxBackbone.from_preset(os.path.abspath(preset))
    max_diff = max(
        np.abs(a - b).max()
        for a, b in zip(outputs, keras_outputs(reloaded, inputs))
    )
    print(f"Max abs diff after reload: {max_diff:.2e}")
    # GPU kernels are not bitwise reproducible across compiled models.
    return max_diff <= 1e-5


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

    print("\n--- Running FluxPipeline and FluxTransformer2DModel ---")
    inputs = pipeline_inputs(hf_id, dtype)
    expected, reference_params = reference_outputs(hf_id, dtype, inputs)

    print("\n--- Converting ---")
    backbone = FluxBackbone.from_preset(f"hf://{hf_id}", dtype=dtype)
    outputs = keras_outputs(backbone, inputs)

    checks = {
        "Parameter count": lambda: check_parameter_count(
            backbone, reference_params
        ),
        "Transformer output": lambda: check_outputs(outputs, expected, dtype),
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
    backbone.save_to_preset(preset)
    # Free the converted model before reloading the preset, so that only one
    # copy of the weights is in memory at a time.
    backbone = None
    gc.collect()
    if not check_saved_preset(outputs, inputs, preset):
        raise RuntimeError("The saved preset does not reproduce the outputs.")
    print(f"Saved and verified preset: ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
