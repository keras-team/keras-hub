r"""Convert the official FLUX.1 checkpoints to KerasHub presets.

Usage:
    python tools/checkpoint_conversion/convert_flux_checkpoints.py \
        --preset flux1_schnell

This converts the weights at `--dtype` (bfloat16 by default, ~24GB of host
RAM for the 11.9B parameter backbone), saves the preset to a staging
directory, then numerically validates the *saved* preset against the
diffusers `FluxTransformer2DModel` of the same repo, and only then moves it
to `--output_dir`. Validation runs in `--validation_dtype` (float32 by
default) for both models, one after the other, so only one 11.9B parameter
model is ever resident: ~48GB of host RAM at float32, ~24GB at bfloat16.

The checkpoint weights are bfloat16, so a bfloat16 preset holds exactly the
same values as the checkpoint, and validating it in float32 checks the
architecture and the weight mapping to float32 noise rather than to
bfloat16 rounding.

To convert and validate in separate processes:

    python tools/checkpoint_conversion/convert_flux_checkpoints.py \
        --preset flux1_schnell --skip_validation
    python tools/checkpoint_conversion/convert_flux_checkpoints.py \
        --preset flux1_schnell --validate_only flux1_schnell

Conversion runs on the host; set FLUX_CONVERT_ALLOW_GPU=1 to allow a GPU.
"""

import argparse
import gc
import importlib.metadata
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
import numpy as np  # noqa: E402
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
    mapping can be reused verbatim. It also records which tensors were read
    and which Keras variables were written, for `check_conversion`.
    """

    def __init__(self, path):
        self._path = path
        self._file = None
        self.keys = None
        self.used = set()
        self.assigned = {}

    def __enter__(self):
        self._file = safe_open(self._path, framework="pt", device="cpu")
        self._file.__enter__()
        self.keys = set(self._file.keys())
        return self

    def __exit__(self, *exc_info):
        return self._file.__exit__(*exc_info)

    def has_tensor(self, hf_weight_key):
        return hf_weight_key in self.keys

    def get_tensor(self, hf_weight_key):
        tensor = self._file.get_tensor(hf_weight_key)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.detach().cpu().numpy()

    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        if hf_weight_key not in self.keys:
            raise KeyError(f"Checkpoint is missing tensor `{hf_weight_key}`.")
        tensor = self.get_tensor(hf_weight_key)
        if hook_fn is not None:
            tensor = hook_fn(tensor, list(keras_variable.shape))
        keras_variable.assign(tensor)
        del tensor
        self.used.add(hf_weight_key)
        entry = self.assigned.setdefault(
            id(keras_variable), [keras_variable.path, 0]
        )
        entry[1] += 1

    def num_params(self):
        """Total element count of the checkpoint, read from the header."""
        return sum(
            int(np.prod(self._file.get_slice(key).get_shape()))
            for key in self.keys
        )


def check_conversion(backbone, loader):
    """Fail unless the weight mapping was complete and one-to-one.

    `convert_weights` raises on a *missing* tensor, but a Keras weight the
    mapping forgets would silently keep its random initialization, and a
    checkpoint tensor it never reads would go unnoticed.
    """
    weights = backbone.weights
    never = [w.path for w in weights if id(w) not in loader.assigned]
    twice = [path for path, count in loader.assigned.values() if count > 1]
    unused = sorted(loader.keys - loader.used)
    keras_params = backbone.count_params()
    checkpoint_params = loader.num_params()
    problems = []
    if never:
        problems.append(
            f"{len(never)} Keras weights were never assigned, e.g. {never[:3]}"
        )
    if twice:
        problems.append(
            f"{len(twice)} Keras weights were assigned more than once, e.g. "
            f"{twice[:3]}"
        )
    if unused:
        problems.append(
            f"{len(unused)} checkpoint tensors were never read, e.g. "
            f"{unused[:3]}"
        )
    if keras_params != checkpoint_params:
        problems.append(
            f"the backbone has {keras_params:,} parameters but the checkpoint "
            f"has {checkpoint_params:,}"
        )
    if problems:
        raise SystemExit(
            "Weight conversion is incomplete:\n  - " + "\n  - ".join(problems)
        )
    print(
        f"  All {len(weights)} Keras weights were assigned exactly once from "
        f"the checkpoint's {len(loader.keys)} tensors "
        f"({keras_params:,} parameters)."
    )


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


# Validation runs both models in `--validation_dtype`, independently of the
# dtype the preset is saved in. float32, the default, is the meaningful
# check: the checkpoint is bfloat16, so a bfloat16 preset holds exactly the
# checkpoint's values, and comparing in float32 resolves the architecture and
# the weight mapping down to float32 noise.
#
# In the low precision dtypes every op output is rounded (bfloat16 keeps 8
# mantissa bits), so two correct implementations drift apart block after
# block. Measured in bfloat16 on just the first 1 double + 1 single block,
# KerasHub and diffusers already differed by 1.9e-2, while each was equally
# far (8e-3 relative L2) from a float32 run. Over all 57 blocks that noise is
# of the same order as a single mis-mapped tensor (~1e-1), so validating in
# low precision only catches gross faults. On those same 1 + 1 blocks, a
# preset with the q/k norm scales swapped scored 8.8e-2 in bfloat16 and
# passed, while float32 rejected it at 9.1e-2 against 1.4e-6 when correct.
_REFERENCE_DTYPES = {
    "float32": (torch.float32, 1e-4),
    "float16": (torch.float16, 5e-2),
    "bfloat16": (torch.bfloat16, 1e-1),
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
    reference = np.asarray(reference, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    if reference.shape != actual.shape:
        raise AssertionError(
            f"shape mismatch: reference {reference.shape} vs {actual.shape}"
        )
    denominator = max(float(np.abs(reference).max()), 1.0)
    return float(np.abs(reference - actual).max()) / denominator


def _align_reference_epsilon(reference_model):
    """Make sure the reference's q/k RMSNorm epsilon is BFL's `1e-6`.

    The Black Forest Labs implementation that produced these weights uses
    `eps=1e-6` in its q/k RMSNorm, and KerasHub follows it. diffusers'
    `FluxAttention` defaults to `1e-5`, but its Flux transformer blocks pass
    `1e-6`, so this is normally a no-op. It stays as a guard: if a diffusers
    release changes the value, the difference is reported and normalised
    rather than polluting the comparison.

    Modules are matched by class name: diffusers has used both
    `torch.nn.RMSNorm` and its own `RMSNorm` class here.
    """
    norms = [
        module
        for module in reference_model.modules()
        if type(module).__name__ == "RMSNorm" and hasattr(module, "eps")
    ]
    changed = sorted({module.eps for module in norms if module.eps != 1e-6})
    for module in norms:
        module.eps = 1e-6
    if changed:
        print(
            f"  Set the diffusers q/k RMSNorm eps to 1e-6 (was {changed}) in "
            f"{len(norms)} modules."
        )
    else:
        print(
            f"  diffusers q/k RMSNorm eps is 1e-6 in all {len(norms)} modules."
        )


# Validation inputs: two samples at different timesteps (1.0 is
# FLUX.1-schnell's first sampling step), image tokens on a latent grid with
# `(0, row, col)` ids exactly as the diffusers pipeline builds them, and
# zero text ids. The batch of two catches anything that mixes samples, and
# the non-square grid drives the two spatial RoPE axes differently.
_VALIDATION_TIMESTEPS = (1.0, 0.25)
_VALIDATION_GRID = (8, 12)  # latent rows x columns: 96 image tokens.
_VALIDATION_TEXT_TOKENS = 16
_VALIDATION_GUIDANCE = 3.5


def _validation_inputs(guidance_embed):
    """Build the deterministic inputs both implementations are fed."""
    batch = len(_VALIDATION_TIMESTEPS)
    rows, cols = _VALIDATION_GRID
    image_sequence, text_sequence = rows * cols, _VALIDATION_TEXT_TOKENS
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(
        batch, image_sequence, INPUT_CHANNELS, generator=generator
    )
    text = torch.randn(
        batch, text_sequence, TEXT_EMBEDDING_DIM, generator=generator
    )
    pooled = torch.randn(batch, Y_DIM, generator=generator)
    timestep = torch.tensor(_VALIDATION_TIMESTEPS)
    image_ids = torch.zeros(rows, cols, 3)
    image_ids[..., 1] = torch.arange(rows).float()[:, None]
    image_ids[..., 2] = torch.arange(cols).float()[None, :]
    image_ids = image_ids.reshape(image_sequence, 3)
    text_ids = torch.zeros(text_sequence, 3)
    guidance = (
        torch.full((batch,), _VALIDATION_GUIDANCE) if guidance_embed else None
    )
    return {
        "image": image,
        "text": text,
        "pooled": pooled,
        "timestep": timestep,
        "image_ids": image_ids,
        "text_ids": text_ids,
        "guidance": guidance,
    }


def _describe_inputs(tensors):
    batch, image_sequence, _ = tensors["image"].shape
    rows, cols = _VALIDATION_GRID
    timesteps = ", ".join(f"{t:g}" for t in tensors["timestep"].tolist())
    description = (
        f"batch {batch} (timesteps {timesteps}), {image_sequence} image "
        f"tokens on a {rows}x{cols} grid, {tensors['text'].shape[1]} text "
        "tokens"
    )
    if tensors["guidance"] is not None:
        description += f", guidance {_VALIDATION_GUIDANCE:g}"
    return description


def _use_full_float32_matmuls():
    """Keep float32 matmuls in float32 on GPUs (no TF32), on every backend.

    Only matters with FLUX_CONVERT_ALLOW_GPU=1: TF32 rounds matmul inputs to
    10 mantissa bits, which alone exceeds the float32 tolerance.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backend = keras.config.backend()
    if backend == "jax":
        import jax

        jax.config.update("jax_default_matmul_precision", "highest")
    elif backend == "tensorflow":
        import tensorflow as tf

        tf.config.experimental.enable_tensor_float_32_execution(False)


def _load_reference(spec, reference_dtype):
    """The diffusers `FluxTransformer2DModel` of the same repo.

    Its diffusers-format weights are a separate copy of the checkpoint,
    converted by diffusers' own code, so it is an independent reference for
    both the architecture and the weight mapping.
    """
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel.from_pretrained(
        spec["repo_id"], subfolder="transformer", torch_dtype=reference_dtype
    )


def _reference_output(spec, reference_dtype, tensors):
    """Run the diffusers reference and free it before returning.

    The reference is local to this call so that callers can sequence it
    against the Keras model rather than holding both.
    """
    print(f"Loading the diffusers reference in {reference_dtype}...")
    reference = _load_reference(spec, reference_dtype)
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
    gc.collect()
    return expected.float().numpy()


def _keras_output(backbone, tensors, guidance_embed):
    """Run the converted backbone on the same inputs as the reference."""
    batch = tensors["image"].shape[0]

    def per_sample(ids):
        # diffusers takes one `(sequence, 3)` id table for the whole batch,
        # the Keras backbone one per sample.
        return np.repeat(ids.numpy()[None], batch, axis=0)

    inputs = {
        "image": tensors["image"].numpy(),
        "text": tensors["text"].numpy(),
        "y": tensors["pooled"].numpy(),
        "timesteps": tensors["timestep"].numpy(),
        "image_ids": per_sample(tensors["image_ids"]),
        "text_ids": per_sample(tensors["text_ids"]),
    }
    if guidance_embed:
        inputs["guidance"] = tensors["guidance"].numpy()
    return np.asarray(
        backbone.predict(inputs, batch_size=batch, verbose=0), dtype="float32"
    )


def _report(expected, actual, tolerance, dtype):
    """Print the agreement and refuse to continue if it is too poor."""
    error = _relative_error(expected, actual)
    expected = np.asarray(expected, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    difference = np.abs(expected - actual)
    cosine = float(
        np.sum(expected * actual)
        / max(np.linalg.norm(expected) * np.linalg.norm(actual), 1e-30)
    )
    print(
        f"  output shape {expected.shape}, max |reference| "
        f"{np.abs(expected).max():.4g}"
    )
    print(
        f"  max abs error {difference.max():.3e}, mean abs error "
        f"{difference.mean():.3e}, cosine similarity {cosine:.8f}"
    )
    print(f"  max relative error {error:.3e}  (tol {tolerance:.0e})")
    # `not error <= tolerance` rather than `error > tolerance`: a NaN or inf
    # anywhere in the output must fail, not slip through the comparison.
    if not error <= tolerance:
        hint = ""
        if dtype.removeprefix("mixed_") != "float32":
            hint = (
                f" In {dtype} this comparison is dominated by rounding noise; "
                "re-run with `--validation_dtype float32` before concluding "
                "the conversion is wrong."
            )
        raise SystemExit(
            f"Numerical validation FAILED: {error:.3e} > {tolerance:.0e}. "
            "Do not upload this preset." + hint
        )
    if dtype.removeprefix("mixed_") != "float32":
        print(
            f"  ⚠️  A {dtype} pass only rules out gross faults (a missing, "
            "transposed or random tensor); subtler mapping errors, such as "
            "swapped q/k norm scales, pass it too. Re-run with "
            "`--validation_dtype float32` before uploading."
        )


def validate_preset(preset_dir, spec, guidance_embed, dtype, tolerance=None):
    """Compare a saved preset against the real diffusers model.

    Uses the diffusers-format copy of the same checkpoint as an independent
    reference: if both the architecture and the weight mapping are right,
    the two must agree to floating point noise.

    This reliably catches artifact-level faults -- a mis-mapped, transposed
    or missing tensor, which land around 1e-1. It is a weaker detector of
    subtly wrong formulas: an incorrect activation function moves the
    end-to-end error only to ~1e-4. That class is covered instead by the
    unit tests in `keras_hub/src/models/flux/flux_layers_test.py`, which
    check the layers individually.

    Both models run in `dtype`, whatever dtype the preset was saved in, and
    the reference runs and is freed *before* the preset is read back, so
    only one 11.9B parameter model is ever resident (~48GB at float32, ~24GB
    at bfloat16). Validating the saved preset rather than the model in
    memory also covers the save / load round trip.
    """
    reference_dtype, default_tolerance = _reference_dtype_and_tolerance(
        dtype, guidance_embed
    )
    if tolerance is None:
        tolerance = default_tolerance
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
    if dtype.removeprefix("mixed_") == "float32":
        _use_full_float32_matmuls()
    tensors = _validation_inputs(guidance_embed)
    print(
        f"Validating against diffusers `FluxTransformer2DModel` "
        f"({spec['repo_id']}), both in {dtype}, on {_describe_inputs(tensors)}."
    )
    print(
        f"  keras {keras.__version__} ({keras.config.backend()} backend), "
        f"torch {torch.__version__}, diffusers "
        f"{importlib.metadata.version('diffusers')}"
    )
    expected = _reference_output(spec, reference_dtype, tensors)
    print(f"Loading the converted preset from {preset_dir} in {dtype}...")
    keras.config.set_dtype_policy(dtype)
    backbone = FluxBackbone.from_preset(preset_dir, dtype=dtype)
    actual = _keras_output(backbone, tensors, guidance_embed)
    del backbone
    gc.collect()
    _report(expected, actual, tolerance, dtype)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        # The docstring is mostly copy-pasteable commands, which the
        # default formatter reflows into one unreadable paragraph.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dtypes = sorted(_REFERENCE_DTYPES) + ["mixed_bfloat16", "mixed_float16"]
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
        choices=dtypes,
        help="Dtype policy the preset is converted and saved in.",
    )
    parser.add_argument(
        "--validation_dtype",
        default="float32",
        choices=dtypes,
        help=(
            "Dtype both models run in for validation, independent of "
            "--dtype. float32 (default) needs ~48GB of host RAM; bfloat16 "
            "needs ~24GB but only catches gross errors."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=(
            "Override the max relative error tolerance (default: 1e-4 at "
            "float32, 1e-3 for guidance-distilled models, 5e-2 at float16, "
            "1e-1 at bfloat16)."
        ),
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help=(
            "Save without comparing against the reference. The preset is "
            "UNVALIDATED until --validate_only is run."
        ),
    )
    parser.add_argument(
        "--validate_only",
        default=None,
        metavar="PRESET_DIR",
        help=(
            "Skip conversion and validate an already saved preset. Phase "
            "two of the two-process flow; run in a fresh process."
        ),
    )
    args = parser.parse_args()

    if args.validate_only and args.skip_validation:
        parser.error("--validate_only and --skip_validation are exclusive.")

    spec = PRESETS[args.preset]
    guidance_embed = spec["guidance_embed"]
    output_dir = os.path.normpath(args.output_dir or args.preset)

    if args.validate_only:
        validate_preset(
            args.validate_only,
            spec,
            guidance_embed,
            args.validation_dtype,
            args.tolerance,
        )
        print(f"✅ Output validated in {args.validation_dtype}")
        return

    # A preset that does not match the reference implementation must never
    # reach `output_dir`. It is saved to a staging directory next to it,
    # validated as saved, and only then moved into place. Both paths are
    # checked now rather than after a 24GB download and conversion.
    staging_dir = f"{output_dir}.unvalidated"
    if not args.skip_validation:
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            raise SystemExit(
                f"`{output_dir}` already exists. Validate it with "
                f"`--validate_only {output_dir}`, or pass another "
                "--output_dir."
            )
        if os.path.exists(staging_dir):
            raise SystemExit(
                f"`{staging_dir}` is left over from an earlier run that did "
                "not pass validation. Inspect it (or re-validate it with "
                f"`--validate_only {staging_dir}`), delete it, and re-run."
            )

    keras.config.set_dtype_policy(args.dtype)

    checkpoint_path = download_checkpoint(spec["repo_id"], spec["filename"])

    backbone = build_backbone(guidance_embed)

    # `SafetensorLoader` resolves files relative to a preset directory, and
    # `convert_flux` is written against that interface. Point a temporary
    # directory at the cached checkpoint with a symlink so we can reuse the
    # shared mapping without copying 24GB.
    with tempfile.TemporaryDirectory() as tmp_dir:
        linked = os.path.join(tmp_dir, "model.safetensors")
        # Absolute: a relative target (e.g. from a relative HF_HOME) would
        # resolve against `tmp_dir` and leave a dangling link.
        os.symlink(os.path.abspath(checkpoint_path), linked)

        print("Converting weights...")
        with TorchSafetensorLoader(linked) as loader:
            convert_flux.convert_weights(backbone, loader, {})
            check_conversion(backbone, loader)

    if args.skip_validation:
        print(f"Saving preset to {output_dir}")
        backbone.save_to_preset(output_dir)
        print("Done.")
        print(
            f"\n⚠️  {output_dir} is UNVALIDATED. Do not upload it until "
            f"the following passes:\n"
            f"    python {os.path.relpath(__file__)} "
            f"--preset {args.preset} --validate_only {output_dir}"
        )
        return

    print(f"Saving preset to {staging_dir} until it is validated")
    backbone.save_to_preset(staging_dir)
    # Free the converted model: validation re-loads it from disk.
    del backbone
    gc.collect()

    try:
        validate_preset(
            staging_dir,
            spec,
            guidance_embed,
            args.validation_dtype,
            args.tolerance,
        )
    except SystemExit:
        print(f"\nThe unvalidated preset was left in {staging_dir}.")
        raise
    os.replace(staging_dir, output_dir)
    print(f"✅ Output validated in {args.validation_dtype}")
    print(f"Saved the validated preset to {output_dir}")


if __name__ == "__main__":
    main()
