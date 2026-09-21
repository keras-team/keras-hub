r"""Convert the official FLUX.1 checkpoints to KerasHub presets.

Usage:
    python tools/checkpoint_conversion/convert_flux_checkpoints.py \
        --preset flux1_schnell

"""

import argparse
import os
import tempfile

import keras
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from keras_hub.src.models.flux.flux_backbone import FluxBackbone
from keras_hub.src.utils.transformers import convert_flux

PRESETS = {
    "flux1_schnell": {
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "filename": "flux1-schnell.safetensors",
        "guidance_embed": False,
    },
    "flux1_dev": {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "filename": "flux1-dev.safetensors",
        "guidance_embed": True,
    },
}

# Shared across both FLUX.1 variants.
INPUT_CHANNELS = 64
HIDDEN_SIZE = 3072
MLP_RATIO = 4.0
NUM_HEADS = 24
DEPTH = 19
DEPTH_SINGLE_BLOCKS = 38
AXES_DIM = [16, 56, 56]
THETA = 10_000
USE_BIAS = True
TEXT_EMBEDDING_DIM = 4096
Y_DIM = 768


class TorchSafetensorLoader:
    """A `SafetensorLoader`-compatible reader for bfloat16 checkpoints.

    `keras_hub`'s `SafetensorLoader` opens files with `framework="np"`, and
    numpy has no bfloat16 dtype, so it cannot read the official FLUX
    checkpoints. This reads through the torch backend and upcasts one tensor
    at a time, which also keeps the ~24GB checkpoint from ever being fully
    resident in memory.

    It implements just the surface `convert_flux` needs, so the real weight
    mapping can be reused verbatim.
    """

    def __init__(self, path):
        self._path = path
        self._file = None
        self._keys = None

    def __enter__(self):
        self._file = safe_open(self._path, framework="pt", device="cpu")
        self._file.__enter__()
        self._keys = set(self._file.keys())
        return self

    def __exit__(self, *exc_info):
        return self._file.__exit__(*exc_info)

    def has_tensor(self, hf_weight_key):
        return hf_weight_key in self._keys

    def get_tensor(self, hf_weight_key):
        tensor = self._file.get_tensor(hf_weight_key)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.detach().cpu().numpy()

    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        if hf_weight_key not in self._keys:
            raise KeyError(f"Checkpoint is missing tensor `{hf_weight_key}`.")
        tensor = self.get_tensor(hf_weight_key)
        if hook_fn is not None:
            tensor = hook_fn(tensor, list(keras_variable.shape))
        keras_variable.assign(tensor)
        del tensor


def download_checkpoint(repo_id, filename):
    """Download the checkpoint into the shared HF cache and return its path.

    Deliberately uses the cache rather than the working directory so that a
    re-run does not re-download ~24GB, and so we never delete a file the user
    may want to keep.
    """
    print(f"Downloading {repo_id}/{filename} (this is ~24GB)...")
    return hf_hub_download(repo_id=repo_id, filename=filename, token=True)


def build_backbone(guidance_embed):
    """Build a FLUX backbone with dynamic sequence axes.

    The sequence axes must stay `None`: they are serialized into
    `get_config()`, so pinning them here would produce a preset that can only
    ever run at one resolution and prompt length.
    """
    return FluxBackbone(
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        mlp_ratio=MLP_RATIO,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        depth_single_blocks=DEPTH_SINGLE_BLOCKS,
        axes_dim=AXES_DIM,
        theta=THETA,
        use_bias=USE_BIAS,
        guidance_embed=guidance_embed,
        image_shape=(None, INPUT_CHANNELS),
        text_shape=(None, TEXT_EMBEDDING_DIM),
        image_ids_shape=(None, 3),
        text_ids_shape=(None, 3),
        y_shape=(Y_DIM,),
    )


# The reference is loaded in whatever dtype the preset is being converted
# to. That halves peak memory at the bfloat16 default (23.8GB rather than
# 47.6GB for 11.9B parameters) and, more importantly, compares at the
# precision the preset will actually ship in.
_REFERENCE_DTYPES = {
    "float32": (torch.float32, 1e-4),
    "float16": (torch.float16, 1e-2),
    "bfloat16": (torch.bfloat16, 2e-2),
}

# The guidance embedding evaluates cos(guidance * 1000 * omega), so at
# guidance=3.5 the argument reaches 3500 -- where float32 has only about
# 1e-4 of relative resolution left. Two correct implementations then
# disagree by a few e-4 purely on conditioning. Measured error in the
# timestep embedding alone: 1.5e-5 at an argument of 250, 2.4e-4 at 3500,
# 4.8e-4 at 7000. Only float32 needs this: for the low precision policies
# the effect is already far below their own resolution.
_GUIDANCE_FLOAT32_TOLERANCE = 1e-3


def _reference_dtype_and_tolerance(dtype, guidance_embed):
    """Map a Keras dtype policy name onto a torch dtype and a tolerance.

    Mixed policies keep their weights in the low-precision dtype, which is
    what governs the achievable agreement, so `mixed_bfloat16` is treated
    exactly like `bfloat16`.
    """
    name = dtype.removeprefix("mixed_")
    if name not in _REFERENCE_DTYPES:
        raise SystemExit(
            f"Cannot validate under dtype `{dtype}`. Expected one of "
            f"{sorted(_REFERENCE_DTYPES)} (optionally `mixed_` prefixed)."
        )
    reference_dtype, tolerance = _REFERENCE_DTYPES[name]
    if guidance_embed and name == "float32":
        tolerance = _GUIDANCE_FLOAT32_TOLERANCE
    return reference_dtype, tolerance


def _relative_error(reference, actual):
    """Max absolute difference, normalised by the reference magnitude.

    An absolute tolerance would be meaningless here: the output magnitude
    depends on the weights, and the floating point error grows with it.
    """
    import numpy as np

    reference = np.asarray(reference, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    if reference.shape != actual.shape:
        raise AssertionError(
            f"shape mismatch: reference {reference.shape} vs {actual.shape}"
        )
    denominator = max(float(np.abs(reference).max()), 1.0)
    return float(np.abs(reference - actual).max()) / denominator


def _align_reference_epsilon(reference_model):
    """Match the reference's RMSNorm epsilon to the original BFL value.

    diffusers defaults `FluxAttention` q/k RMSNorm to `eps=1e-5`; the Black
    Forest Labs implementation that produced these weights uses `1e-6`, and
    KerasHub follows BFL. This is a real upstream divergence, not a porting
    error, so normalise it rather than let it pollute the comparison.
    """
    for module in reference_model.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.eps = 1e-6


def validate_output(backbone, spec, guidance_embed, dtype):
    """Compare the converted backbone against the real diffusers model.

    Uses the diffusers-format copy of the same checkpoint as an independent
    reference: if both the architecture and the weight mapping are right,
    the two must agree to floating point noise.

    This reliably catches artifact-level faults -- a mis-mapped, transposed
    or missing tensor, which land around 1e-1. It is a weaker detector of
    subtly wrong formulas: an incorrect activation function moves the
    end-to-end error only to ~1e-4. That class is covered instead by the
    unit tests in `keras_hub/src/models/flux/flux_layers_test.py`, which
    check the layers individually.
    """
    import numpy as np
    from diffusers import FluxTransformer2DModel

    reference_dtype, tolerance = _reference_dtype_and_tolerance(
        dtype, guidance_embed
    )

    print(f"Loading the diffusers reference in {reference_dtype}...")
    reference = FluxTransformer2DModel.from_pretrained(
        spec["repo_id"], subfolder="transformer", torch_dtype=reference_dtype
    )
    reference.eval()
    _align_reference_epsilon(reference)

    image_sequence, text_sequence = 32, 8
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(1, image_sequence, INPUT_CHANNELS, generator=generator)
    text = torch.randn(
        1, text_sequence, TEXT_EMBEDDING_DIM, generator=generator
    )
    pooled = torch.randn(1, Y_DIM, generator=generator)
    timestep = torch.full((1,), 0.25)
    image_ids = torch.zeros(image_sequence, 3)
    image_ids[:, 1] = torch.arange(image_sequence).float()
    text_ids = torch.zeros(text_sequence, 3)
    guidance = torch.full((1,), 3.5) if guidance_embed else None

    with torch.no_grad():
        expected = reference(
            hidden_states=image.to(reference_dtype),
            encoder_hidden_states=text.to(reference_dtype),
            pooled_projections=pooled.to(reference_dtype),
            timestep=timestep.to(reference_dtype),
            # Position ids stay float32: diffusers computes the RoPE
            # frequencies from them in float64 regardless.
            img_ids=image_ids,
            txt_ids=text_ids,
            guidance=None if guidance is None else guidance.to(reference_dtype),
            return_dict=False,
        )[0]
    # Free the reference before running Keras, so the two 11.9B parameter
    # copies are never resident at the same time.
    del reference
    expected = expected.float().numpy()

    inputs = {
        "image": image.numpy(),
        "text": text.numpy(),
        "y": pooled.numpy(),
        "timesteps": timestep.numpy(),
        "image_ids": image_ids.numpy()[None],
        "text_ids": text_ids.numpy()[None],
    }
    if guidance_embed:
        inputs["guidance"] = guidance.numpy()
    actual = np.asarray(backbone.predict(inputs, verbose=0), dtype="float32")

    error = _relative_error(expected, actual)
    print(f"  reference shape: {expected.shape}")
    print(f"  max relative error: {error:.3e}  (tol {tolerance:.0e})")
    if error > tolerance:
        raise SystemExit(
            f"Numerical validation FAILED: {error:.3e} > {tolerance:.0e}. "
            "Do not upload this preset."
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        # The docstring is mostly copy-pasteable commands, which the
        # default formatter reflows into one unreadable paragraph.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        default="flux1_schnell",
        choices=sorted(PRESETS),
        help="Which FLUX.1 variant to convert.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write the preset. Defaults to ./<preset>.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(_REFERENCE_DTYPES) + ["mixed_bfloat16", "mixed_float16"],
        help="Dtype policy for the converted model.",
    )
    args = parser.parse_args()

    spec = PRESETS[args.preset]
    output_dir = args.output_dir or args.preset

    keras.config.set_dtype_policy(args.dtype)

    checkpoint_path = download_checkpoint(spec["repo_id"], spec["filename"])

    backbone = build_backbone(spec["guidance_embed"])

    # `SafetensorLoader` resolves files relative to a preset directory, and
    # `convert_flux` is written against that interface. Point a temporary
    # directory at the cached checkpoint with a symlink so we can reuse the
    # shared mapping without copying 24GB.
    with tempfile.TemporaryDirectory() as tmp_dir:
        linked = os.path.join(tmp_dir, "model.safetensors")
        os.symlink(checkpoint_path, linked)

        print("Converting weights...")
        with TorchSafetensorLoader(linked) as loader:
            convert_flux.convert_weights(backbone, loader, {})

    # Deliberately before the save: a preset that does not match the
    # reference implementation must never reach disk.
    validate_output(backbone, spec, spec["guidance_embed"], args.dtype)
    print("✅ Output validated")

    print(f"Saving preset to {output_dir}")
    backbone.save_to_preset(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
