r"""Convert the official FLUX.1 checkpoints to KerasHub presets.

Usage:
    python tools/checkpoint_conversion/convert_flux_checkpoints.py \
        --preset flux1_schnell

FLUX.1 is ~11.9B parameters, so at the bfloat16 default the backbone alone
needs ~24GB of host RAM, and validation briefly holds the diffusers
reference alongside it for ~48GB. On a machine that cannot fit both, split
it into two phases so only one model is ever resident:

    python tools/checkpoint_conversion/convert_flux_checkpoints.py \
        --preset flux1_schnell --skip_validation
    python tools/checkpoint_conversion/convert_flux_checkpoints.py \
        --preset flux1_schnell --validate_only flux1_schnell

Conversion runs on the host; set FLUX_CONVERT_ALLOW_GPU=1 to allow a GPU.
"""

import argparse
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if os.environ.get("FLUX_CONVERT_ALLOW_GPU") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import keras  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import constants  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from huggingface_hub.errors import GatedRepoError  # noqa: E402
from huggingface_hub.errors import LocalTokenNotFoundError  # noqa: E402
from safetensors import safe_open  # noqa: E402

try:
    from keras_hub.src.models.flux.flux_backbone import (  # noqa: E402
        FluxBackbone,
    )
    from keras_hub.src.utils.transformers import convert_flux  # noqa: E402
except ModuleNotFoundError as e:
    import keras_hub  # noqa: E402

    raise ModuleNotFoundError(
        f"{e}\n\n"
        f"`keras_hub` was imported from {os.path.dirname(keras_hub.__file__)}, "
        f"which does not contain the FLUX backbone conversion code.\n"
        "Expected it to resolve inside "
        f"{os.path.join(_REPO_ROOT, 'keras_hub')}.\n"
        "Make sure you are on a checkout that has "
        "`keras_hub/src/models/flux/flux_backbone.py` (older revisions name it "
        "`flux_model.py`), and that no other `keras-hub` install shadows it "
        "(`pip uninstall keras-hub` or `pip install -e .` from the repo root)."
    ) from e

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


# Both FLUX.1 checkpoints are ~23.8GB of bfloat16 weights.
_CHECKPOINT_SIZE_GB = 24


def _free_gb(path):
    """Free space on the filesystem holding `path`, in GB.

    The cache directory may not exist yet, so walk up to the nearest
    ancestor that does. Returns None if nothing along the path resolves,
    in which case the caller should just attempt the download.
    """
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent
    return shutil.disk_usage(path).free / 1024**3


def download_checkpoint(repo_id, filename):
    """Download the checkpoint into the shared HF cache and return its path.

    Deliberately uses the cache rather than the working directory so that a
    re-run does not re-download ~24GB, and so we never delete a file the user
    may want to keep.
    """
    # The checkpoint plus the Xet chunk cache both land on the cache
    # filesystem, and a full disk surfaces as an opaque Xet writer error
    # rather than ENOSPC, so check up front.
    free_gb = _free_gb(constants.HF_HUB_CACHE)
    if free_gb is not None and free_gb < _CHECKPOINT_SIZE_GB:
        raise SystemExit(
            f"Only {free_gb:.1f}GB free on the filesystem holding "
            f"{constants.HF_HUB_CACHE}, but {repo_id}/{filename} needs about "
            f"{_CHECKPOINT_SIZE_GB}GB (more while Xet caches chunks).\n"
            "Free up space, or point HF_HOME at a larger volume."
        )

    print(f"Downloading {repo_id}/{filename} (this is ~24GB)...")
    # No explicit `token`: the default picks up `HF_TOKEN` or a cached
    # `hf auth login`. `token=True` instead raised a bare
    # `LocalTokenNotFoundError` before contacting the Hub, which hid the
    # real requirement. Both FLUX.1 repos are gated, so access must be
    # granted to the account behind the token.
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except (GatedRepoError, LocalTokenNotFoundError) as e:
        raise SystemExit(
            f"Could not download {repo_id}/{filename}: {e}\n\n"
            f"`{repo_id}` needs authentication. Accept the model terms at "
            f"https://huggingface.co/{repo_id}, then authenticate with "
            "`hf auth login` or by setting the `HF_TOKEN` environment "
            "variable."
        ) from e
    except RuntimeError as e:
        # `hf_xet` raises plain RuntimeErrors out of its Rust extension
        # ("Background writer channel closed" and friends). They are
        # transfer-layer faults, not something the checkpoint or the
        # credentials can fix, and the plain HTTPS path usually succeeds.
        # `is_xet_available()` reads this constant per call, so flipping it
        # here is enough to take that path on the retry.
        print(f"\nXet transfer failed: {e}")
        print("Retrying over plain HTTPS (slower, no chunk cache)...")
        constants.HF_HUB_DISABLE_XET = True
        return hf_hub_download(repo_id=repo_id, filename=filename)


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


def _validation_inputs(guidance_embed):
    """Build the deterministic inputs both implementations are fed."""
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
    return {
        "image": image,
        "text": text,
        "pooled": pooled,
        "timestep": timestep,
        "image_ids": image_ids,
        "text_ids": text_ids,
        "guidance": guidance,
    }


def _reference_output(spec, reference_dtype, tensors):
    """Run the diffusers reference and free it before returning.

    The reference is local to this call so that callers can sequence it
    against the Keras model rather than holding both.
    """
    from diffusers import FluxTransformer2DModel

    print(f"Loading the diffusers reference in {reference_dtype}...")
    reference = FluxTransformer2DModel.from_pretrained(
        spec["repo_id"], subfolder="transformer", torch_dtype=reference_dtype
    )
    reference.eval()
    _align_reference_epsilon(reference)

    guidance = tensors["guidance"]
    with torch.no_grad():
        expected = reference(
            hidden_states=tensors["image"].to(reference_dtype),
            encoder_hidden_states=tensors["text"].to(reference_dtype),
            pooled_projections=tensors["pooled"].to(reference_dtype),
            timestep=tensors["timestep"].to(reference_dtype),
            # Position ids stay float32: diffusers computes the RoPE
            # frequencies from them in float64 regardless.
            img_ids=tensors["image_ids"],
            txt_ids=tensors["text_ids"],
            guidance=None if guidance is None else guidance.to(reference_dtype),
            return_dict=False,
        )[0]
    # Free the reference before running Keras, so the two 11.9B parameter
    # copies are never resident at the same time.
    del reference
    return expected.float().numpy()


def _keras_output(backbone, tensors, guidance_embed):
    """Run the converted backbone on the same inputs as the reference."""
    import numpy as np

    inputs = {
        "image": tensors["image"].numpy(),
        "text": tensors["text"].numpy(),
        "y": tensors["pooled"].numpy(),
        "timesteps": tensors["timestep"].numpy(),
        "image_ids": tensors["image_ids"].numpy()[None],
        "text_ids": tensors["text_ids"].numpy()[None],
    }
    if guidance_embed:
        inputs["guidance"] = tensors["guidance"].numpy()
    return np.asarray(backbone.predict(inputs, verbose=0), dtype="float32")


def _report(expected, actual, tolerance):
    """Print the agreement and refuse to continue if it is too poor."""
    error = _relative_error(expected, actual)
    print(f"  reference shape: {expected.shape}")
    print(f"  max relative error: {error:.3e}  (tol {tolerance:.0e})")
    if error > tolerance:
        raise SystemExit(
            f"Numerical validation FAILED: {error:.3e} > {tolerance:.0e}. "
            "Do not upload this preset."
        )


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
    reference_dtype, tolerance = _reference_dtype_and_tolerance(
        dtype, guidance_embed
    )
    tensors = _validation_inputs(guidance_embed)
    expected = _reference_output(spec, reference_dtype, tensors)
    actual = _keras_output(backbone, tensors, guidance_embed)
    _report(expected, actual, tolerance)


def validate_preset(preset_dir, spec, guidance_embed, dtype):
    """Validate an already saved preset, at roughly half the peak memory.

    The same comparison as `validate_output`, but the reference runs and is
    freed *before* the preset is read back, so only one 11.9B parameter
    model is ever resident (~24GB rather than ~48GB at bfloat16). During
    conversion the backbone must already be in memory, which is why that
    path cannot do this and why low-memory machines need the two phases.
    """
    reference_dtype, tolerance = _reference_dtype_and_tolerance(
        dtype, guidance_embed
    )
    # Checked before the reference loads, which otherwise spends ~24GB and
    # several minutes before failing. `save_backbone` writes the config
    # first, so its absence means the conversion phase never reached the
    # save at all -- usually because it was OOM-killed, which leaves no
    # traceback behind.
    if not os.path.isfile(os.path.join(preset_dir, "config.json")):
        existing = (
            sorted(os.listdir(preset_dir))
            if os.path.isdir(preset_dir)
            else "the directory does not exist"
        )
        raise SystemExit(
            f"`{preset_dir}` is not a complete preset: no config.json.\n"
            f"Found: {existing}\n\n"
            "The conversion phase never reached `save_to_preset`. If it "
            "ended without a traceback it was almost certainly killed for "
            "running out of host memory -- conversion needs ~24GB at "
            "bfloat16. Check `free -g`, then re-run:\n"
            f"    python {os.path.relpath(__file__)} --skip_validation"
        )
    tensors = _validation_inputs(guidance_embed)
    expected = _reference_output(spec, reference_dtype, tensors)
    print(f"Loading the converted preset from {preset_dir}...")
    backbone = FluxBackbone.from_preset(preset_dir)
    actual = _keras_output(backbone, tensors, guidance_embed)
    _report(expected, actual, tolerance)


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
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help=(
            "Save without comparing against the reference. Halves peak "
            "memory. The preset is UNVALIDATED until --validate_only is run."
        ),
    )
    parser.add_argument(
        "--validate_only",
        default=None,
        metavar="PRESET_DIR",
        help=(
            "Skip conversion and validate an already saved preset. Phase "
            "two of the low-memory flow; run in a fresh process."
        ),
    )
    args = parser.parse_args()

    if args.validate_only and args.skip_validation:
        parser.error("--validate_only and --skip_validation are exclusive.")

    spec = PRESETS[args.preset]
    output_dir = args.output_dir or args.preset

    keras.config.set_dtype_policy(args.dtype)

    if args.validate_only:
        validate_preset(
            args.validate_only, spec, spec["guidance_embed"], args.dtype
        )
        print("✅ Output validated")
        return

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

    if not args.skip_validation:
        # Deliberately before the save: a preset that does not match the
        # reference implementation must never reach disk.
        validate_output(backbone, spec, spec["guidance_embed"], args.dtype)
        print("✅ Output validated")

    print(f"Saving preset to {output_dir}")
    backbone.save_to_preset(output_dir)
    print("Done.")

    if args.skip_validation:
        print(
            f"\n⚠️  {output_dir} is UNVALIDATED. Do not upload it until "
            f"the following passes:\n"
            f"    python {os.path.relpath(__file__)} "
            f"--preset {args.preset} --dtype {args.dtype} "
            f"--validate_only {output_dir}"
        )


if __name__ == "__main__":
    main()
