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

Validation compares the preset with the reference on:

  - the config and the parameter count;
  - every embedding, double block and single block, and the final layer,
    each fed the reference's own inputs, so that a fault is reported at the
    layer that has it rather than everywhere downstream of it;
  - the output of `FluxBackbone.predict`, at two different input shapes;
  - a 4-step Euler sampling trajectory, where each step is fed the previous
    step's output (skip it with `--skip_sampling`).

The checkpoint weights are bfloat16, so a bfloat16 preset holds exactly the
same values as the checkpoint, and validating it in float32 checks the
architecture and the weight mapping to float32 noise rather than to
bfloat16 rounding.

Validation needs `diffusers` and `accelerate` (`pip install
"diffusers>=0.32" accelerate`); they are checked before anything is
downloaded.

To convert and validate in separate processes:

    python tools/checkpoint_conversion/convert_flux_checkpoints.py \
        --preset flux1_schnell --skip_validation
    python tools/checkpoint_conversion/convert_flux_checkpoints.py \
        --preset flux1_schnell --validate_only flux1_schnell

Conversion runs on the host; set FLUX_CONVERT_ALLOW_GPU=1 to allow a GPU.
"""

import argparse
import functools
import gc
import importlib.metadata
import importlib.util
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
    from keras_hub.src.models.flux.flux_presets import (  # noqa: E402
        presets as REGISTERED_PRESETS,
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


# Only validation needs these. `diffusers` is the reference implementation;
# without `accelerate` it falls back to its slow, memory-hungry loading path
# for the 11.9B parameter reference.
_VALIDATION_PACKAGES = ("diffusers", "accelerate")


def check_validation_dependencies():
    """Fail now if validation could not run, not after the conversion.

    Otherwise a missing package only surfaces once the 24GB download and
    the conversion have finished, when validation starts.
    """
    missing = [
        name
        for name in _VALIDATION_PACKAGES
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(
            f"Validation needs {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed for "
            f"{sys.executable}. Install into the same environment:\n"
            f"    {sys.executable} -m pip install 'diffusers>=0.32' "
            "accelerate\n"
            "or convert now and validate later with --validate_only:\n"
            f"    python {os.path.relpath(__file__)} --skip_validation"
        )


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


def _reference_dtype_and_tolerances(dtype, guidance_embed):
    """Map a Keras dtype policy name onto a torch dtype and tolerances.

    Mixed policies keep their weights in the low-precision dtype, which is
    what governs the achievable agreement, so `mixed_bfloat16` is treated
    exactly like `bfloat16`.

    Returns `(reference_dtype, layer_tolerance, sinusoid_tolerance,
    output_tolerance)`. In float32, the sinusoidal embeddings get
    `_GUIDANCE_FLOAT32_TOLERANCE` too: at t=1 the timestep's argument
    reaches 1000, where correct backends already disagree by ~6e-5 (one
    float32 ulp), and a sinusoid has no weights, so its faults (the cos /
    sin order, the time factor, the frequencies) all show at O(1). A layer
    fed the reference's own intermediates never sees KerasHub's guidance
    embedding, so of the other checks only those downstream of that
    embedding need the looser tolerance.
    """
    name = dtype.removeprefix("mixed_")
    if name not in _REFERENCE_DTYPES:
        raise SystemExit(
            f"Cannot validate under dtype `{dtype}`. Expected one of "
            f"{sorted(_REFERENCE_DTYPES)} (optionally `mixed_` prefixed)."
        )
    reference_dtype, tolerance = _REFERENCE_DTYPES[name]
    sinusoid_tolerance = output_tolerance = tolerance
    if name == "float32":
        sinusoid_tolerance = _GUIDANCE_FLOAT32_TOLERANCE
        if guidance_embed:
            output_tolerance = _GUIDANCE_FLOAT32_TOLERANCE
    return reference_dtype, tolerance, sinusoid_tolerance, output_tolerance


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

# `predict` is also compared on a second input set that changes every
# dynamic axis -- one sample, a portrait grid and an odd text length -- so
# that a preset tied to the first set's shapes cannot pass.
_SHAPE_CHECK_TIMESTEPS = (0.5,)
_SHAPE_CHECK_GRID = (6, 4)  # 24 image tokens, the transposed aspect.
_SHAPE_CHECK_TEXT_TOKENS = 7

# FLUX samples by integrating the predicted velocity with Euler steps from
# noise at t=1 to t=0, and FLUX.1-schnell is distilled for 4 of them.
# Comparing whole trajectories is the diffusion counterpart of the
# generation checks in the LLM conversion scripts: every step is fed the
# previous step's output, so an error that compounds shows up.
_SAMPLING_STEPS = 4


def _validation_inputs(
    guidance_embed,
    timesteps=_VALIDATION_TIMESTEPS,
    grid=_VALIDATION_GRID,
    text_tokens=_VALIDATION_TEXT_TOKENS,
    seed=0,
):
    """Build the deterministic inputs both implementations are fed."""
    batch = len(timesteps)
    rows, cols = grid
    image_sequence, text_sequence = rows * cols, text_tokens
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(
        batch, image_sequence, INPUT_CHANNELS, generator=generator
    )
    text = torch.randn(
        batch, text_sequence, TEXT_EMBEDDING_DIM, generator=generator
    )
    pooled = torch.randn(batch, Y_DIM, generator=generator)
    timestep = torch.tensor(timesteps)
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
        "grid": grid,
    }


def _describe_inputs(tensors):
    batch, image_sequence, _ = tensors["image"].shape
    rows, cols = tensors["grid"]
    timesteps = ", ".join(f"{t:g}" for t in tensors["timestep"].tolist())
    description = (
        f"batch {batch} (timestep{'s' if batch > 1 else ''} {timesteps}), "
        f"{image_sequence} image tokens on a {rows}x{cols} grid, "
        f"{tensors['text'].shape[1]} text tokens"
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


def _to_numpy(value):
    """A torch tensor, or a tuple of them, as float32 numpy."""
    if isinstance(value, (tuple, list)):
        return tuple(_to_numpy(item) for item in value)
    return value.detach().float().cpu().numpy()


def _reference_forward(reference, reference_dtype, tensors):
    """One forward pass of the diffusers reference, as float32 numpy."""
    guidance = tensors["guidance"]
    output = reference(
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
    return output.float().numpy()


def _fused_sequence(output):
    """A single block's output as one `[text, image]` sequence.

    Older diffusers releases return the fused sequence itself; newer ones
    split it and return `(text, image)`.
    """
    if isinstance(output, (tuple, list)):
        return torch.cat(tuple(output), dim=1)
    return output


def _trace_reference(reference, trace):
    """Record the output of every diffusers module KerasHub mirrors.

    Each output is appended to `trace[name]`. Returns the hook handles;
    remove them to stop recording.
    """

    def record(name, transform=lambda output: output):
        def hook(module, args, output):
            trace.setdefault(name, []).append(_to_numpy(transform(output)))

        return hook

    embed = reference.time_text_embed
    hooks = [
        # The sinusoidal projection runs for the timestep, then once more
        # for the guidance strength of guidance-distilled models.
        (embed.time_proj, record("sinusoids")),
        (embed.timestep_embedder, record("time_in")),
        (embed.text_embedder, record("vector_in")),
        (embed, record("vec")),
        (reference.x_embedder, record("img_in")),
        (reference.context_embedder, record("txt_in")),
        (reference.pos_embed, record("rope")),
    ]
    if hasattr(embed, "guidance_embedder"):
        hooks.append((embed.guidance_embedder, record("guidance_in")))
    # Double blocks return `(text, image)`; recorded as `(image, text)`.
    hooks += [
        (block, record("double", lambda output: output[::-1]))
        for block in reference.transformer_blocks
    ]
    hooks += [
        (block, record("single", _fused_sequence))
        for block in reference.single_transformer_blocks
    ]
    return [module.register_forward_hook(hook) for module, hook in hooks]


def _sample(forward, tensors, steps):
    """Integrate the flow from the inputs' noise with Euler steps.

    `forward` maps an input set to the predicted velocity. Each step is fed
    the previous step's latents, as in a sampler, and the update is done in
    float32, as diffusers' `FlowMatchEulerDiscreteScheduler` does.
    """
    sigmas = np.linspace(1.0, 0.0, steps + 1, dtype="float32")
    latents = tensors["image"].numpy()
    batch = latents.shape[0]
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        velocity = forward(
            dict(
                tensors,
                image=torch.from_numpy(latents),
                timestep=torch.full((batch,), float(sigma)),
            )
        )
        latents = latents + (next_sigma - sigma) * velocity
    return latents


def _reference_run(spec, reference_dtype, input_sets, sampling_steps):
    """Run every reference computation up front, then free the reference.

    The forward pass on the first input set also records the output of
    each diffusers module that has a KerasHub counterpart, so that every
    Keras layer can later be fed exactly the inputs its counterpart saw.
    The reference is local to this call so that callers can sequence it
    against the Keras model rather than holding both.
    """
    print(f"Loading the diffusers reference in {reference_dtype}...")
    reference = _load_reference(spec, reference_dtype)
    reference.eval()
    _align_reference_epsilon(reference)
    run = {
        "config": dict(reference.config),
        "params": sum(p.numel() for p in reference.parameters()),
        "trace": {},
    }
    forward = functools.partial(_reference_forward, reference, reference_dtype)
    print("Running the reference...")
    with torch.no_grad():
        hooks = _trace_reference(reference, run["trace"])
        try:
            run["outputs"] = [forward(input_sets[0])]
        finally:
            for hook in hooks:
                hook.remove()
        run["outputs"] += [forward(tensors) for tensors in input_sets[1:]]
        if sampling_steps:
            run["sampled"] = _sample(forward, input_sets[0], sampling_steps)
    # Free the reference before running Keras, so the two 11.9B parameter
    # copies are never resident at the same time.
    del forward, reference
    gc.collect()
    return run


def _per_sample(ids, batch):
    # diffusers takes one `(sequence, 3)` id table for the whole batch,
    # the Keras backbone one per sample.
    return np.repeat(ids.numpy()[None], batch, axis=0)


def _keras_output(backbone, tensors, guidance_embed):
    """Run the converted backbone on the same inputs as the reference."""
    batch = tensors["image"].shape[0]
    inputs = {
        "image": tensors["image"].numpy(),
        "text": tensors["text"].numpy(),
        "y": tensors["pooled"].numpy(),
        "timesteps": tensors["timestep"].numpy(),
        "image_ids": _per_sample(tensors["image_ids"], batch),
        "text_ids": _per_sample(tensors["text_ids"], batch),
    }
    if guidance_embed:
        inputs["guidance"] = tensors["guidance"].numpy()
    return np.asarray(
        backbone.predict(inputs, batch_size=batch, verbose=0), dtype="float32"
    )


def _keras_numpy(value):
    """A Keras tensor (or a numpy array) as numpy, upcast to float32."""
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(
        keras.ops.convert_to_numpy(keras.ops.cast(value, "float32"))
    )


class _Checks:
    """Records every check's verdict, so that one failure hides no other.

    Gated checks decide the outcome. Diagnostics only add context: they are
    flagged when out of tolerance, but never fail validation.
    """

    def __init__(self, dtype):
        self.dtype = dtype
        self.passed = 0
        self.failed = []

    def verdict(self, name, ok, detail, gate=True):
        if gate and ok:
            self.passed += 1
        elif gate:
            self.failed.append(name)
        else:
            detail += "  (diagnostic)"
        mark = "✅" if ok else "❌" if gate else "⚠️ "
        print(f"  {mark} {name:<24}{detail}")

    def record(self, name, error, tolerance, detail="", gate=True):
        """Judge a relative error against its tolerance."""
        # `error <= tolerance` rather than `not error > tolerance`: a NaN
        # anywhere must fail, not slip through the comparison.
        detail = f"max rel {error:.2e}" + (f"  {detail}" if detail else "")
        self.verdict(name, bool(error <= tolerance), detail, gate)

    def compare(self, name, expected, actual, tolerance, detail="", gate=True):
        try:
            error = _relative_error(expected, _keras_numpy(actual))
        except AssertionError as e:  # A shape mismatch.
            self.verdict(name, False, str(e), gate)
        else:
            self.record(name, error, tolerance, detail, gate)

    def finish(self):
        """Fail unless every gated check passed; return how many did."""
        low_precision = self.dtype.removeprefix("mixed_") != "float32"
        if self.failed:
            hint = ""
            if low_precision:
                hint = (
                    f" In {self.dtype} these comparisons are dominated by "
                    "rounding noise; re-run with `--validation_dtype "
                    "float32` before concluding the conversion is wrong."
                )
            raise SystemExit(
                f"\nNumerical validation FAILED: {len(self.failed)} of "
                f"{self.passed + len(self.failed)} checks failed "
                f"({', '.join(self.failed)}). Do not upload this preset." + hint
            )
        if low_precision:
            print(
                f"\n  ⚠️  A {self.dtype} pass only rules out gross faults (a "
                "missing, transposed or random tensor); subtler mapping "
                "errors, such as swapped q/k norm scales, pass it too. "
                "Re-run with `--validation_dtype float32` before uploading."
            )
        return self.passed


def _plain(value):
    """A config value with tuples as lists, as it reads back from JSON."""
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _check_architecture(checks, backbone, run, preset):
    """Check the preset's config and size against the reference."""
    print("\nArchitecture:")
    config = backbone.get_config()
    # What KerasHub's own Hugging Face converter builds from the reference
    # config, including the dynamic sequence axes.
    expected = convert_flux.convert_backbone_config(run["config"])
    mismatched = [
        f"{key} is {_plain(config.get(key))}, diffusers {_plain(value)}"
        for key, value in expected.items()
        if _plain(config.get(key)) != _plain(value)
    ]
    checks.verdict(
        "config",
        not mismatched,
        "; ".join(mismatched)
        or (
            f"matches diffusers: {config['depth']} double + "
            f"{config['depth_single_blocks']} single blocks, hidden size "
            f"{config['hidden_size']}, {config['num_heads']} heads, "
            f"guidance_embed {config['guidance_embed']}, dynamic sequence "
            "lengths"
        ),
    )
    params = backbone.count_params()
    checks.verdict(
        "parameter count",
        params == run["params"],
        f"{params:,} (diffusers {run['params']:,})",
    )
    metadata = REGISTERED_PRESETS.get(preset, {}).get("metadata", {})
    if "params" in metadata:
        checks.verdict(
            "preset metadata",
            metadata["params"] == params,
            f"flux_presets.py lists {metadata['params']:,} parameters",
            gate=False,
        )


def _check_layers(
    checks,
    backbone,
    trace,
    tensors,
    expected,
    tolerance,
    sinusoid_tolerance,
    output_tolerance,
):
    """Check every KerasHub layer against its diffusers counterpart.

    Each layer is fed the reference's own inputs (teacher forcing), so its
    error is its own: a fault shows up at the layer that has it, instead of
    in every layer downstream. Alongside, KerasHub's own forward pass is
    replayed layer by layer ("free-running"), a diagnostic of how the error
    accumulates with depth. Returns that pass's output.
    """
    batch, text_tokens = tensors["text"].shape[:2]

    def tolerance_note(value):
        # Marks the checks judged at other than their section's tolerance.
        return f"(tol {value:.0e})" if value != tolerance else ""

    sinusoid_note = tolerance_note(sinusoid_tolerance)
    output_note = tolerance_note(output_tolerance)

    print(
        "\nEmbeddings, each fed its diffusers counterpart's inputs "
        f"(tol {tolerance:.0e}):"
    )
    sinusoids = trace["sinusoids"]
    time_sinusoid = backbone.timestep_embedding(tensors["timestep"].numpy())
    checks.compare(
        "timestep sinusoid",
        sinusoids[0],
        time_sinusoid,
        sinusoid_tolerance,
        sinusoid_note,
    )
    checks.compare(
        "time_in",
        trace["time_in"][0],
        backbone.time_input_embedder(sinusoids[0]),
        tolerance,
    )
    vec = backbone.time_input_embedder(time_sinusoid)
    if backbone.guidance_embed:
        guidance_sinusoid = backbone.timestep_embedding(
            tensors["guidance"].numpy()
        )
        checks.compare(
            "guidance sinusoid",
            sinusoids[1],
            guidance_sinusoid,
            sinusoid_tolerance,
            sinusoid_note,
        )
        checks.compare(
            "guidance_in",
            trace["guidance_in"][0],
            backbone.guidance_input_embedder(sinusoids[1]),
            tolerance,
        )
        vec = vec + backbone.guidance_input_embedder(guidance_sinusoid)
    vector = backbone.vector_embedder(tensors["pooled"].numpy())
    checks.compare("vector_in", trace["vector_in"][0], vector, tolerance)
    vec = vec + vector
    checks.compare(
        "vec (modulation)",
        trace["vec"][0],
        vec,
        output_tolerance,
        f"from KerasHub's own sinusoids {output_note}".rstrip(),
    )
    image = backbone.image_input_embedder(tensors["image"].numpy())
    checks.compare("img_in", trace["img_in"][0], image, tolerance)
    text = backbone.text_input_embedder(tensors["text"].numpy())
    checks.compare("txt_in", trace["txt_in"][0], text, tolerance)
    ids = np.concatenate(
        [
            _per_sample(tensors["text_ids"], batch),
            _per_sample(tensors["image_ids"], batch),
        ],
        axis=1,
    )
    rope = backbone.positional_embedder(ids)
    # diffusers keeps one `(sequence, head_dim)` table each of cos and sin
    # for the whole batch, with every frequency repeated twice; KerasHub
    # one `(sequence, head_dim // 2, 2)` table of `(cos, sin)` pairs per
    # sample.
    cos, sin = trace["rope"][0]
    table = _keras_numpy(rope)
    checks.compare(
        "RoPE cos/sin tables",
        np.broadcast_to(np.stack([cos, sin])[:, None], (2, batch) + cos.shape),
        np.stack([np.repeat(table[..., i], 2, axis=-1) for i in (0, 1)]),
        tolerance,
    )
    forced_rope = np.repeat(
        np.stack([cos[:, ::2], sin[:, ::2]], axis=-1)[None], batch, axis=0
    )

    print(
        f"\nBlocks, each fed the reference's inputs (tol {tolerance:.0e}); "
        "free-running is KerasHub's own forward pass, for context:"
    )
    forced_vec = trace["vec"][0]
    forced_image, forced_text = trace["img_in"][0], trace["txt_in"][0]
    for index, block in enumerate(backbone.double_blocks):
        expected_image, expected_text = trace["double"][index]
        out_image, out_text = block(
            image=forced_image,
            text=forced_text,
            modulation_encoding=forced_vec,
            positional_encoding=forced_rope,
        )
        image, text = block(
            image=image,
            text=text,
            modulation_encoding=vec,
            positional_encoding=rope,
        )
        # Judged per stream: each has its own weights and its own scale.
        image_error = _relative_error(expected_image, _keras_numpy(out_image))
        text_error = _relative_error(expected_text, _keras_numpy(out_text))
        drift = np.max(
            [
                _relative_error(expected_image, _keras_numpy(image)),
                _relative_error(expected_text, _keras_numpy(text)),
            ]
        )
        checks.record(
            f"double block {index}",
            # `np.max`, unlike `max`, propagates a NaN from either stream.
            float(np.max([image_error, text_error])),
            tolerance,
            f"(image {image_error:.2e}, text {text_error:.2e}), "
            f"free-running {drift:.2e}",
        )
        forced_image, forced_text = expected_image, expected_text

    forced = np.concatenate([forced_text, forced_image], axis=1)
    sequence = keras.ops.concatenate([text, image], axis=1)
    for index, block in enumerate(backbone.single_blocks):
        out = block(
            forced,
            modulation_encoding=forced_vec,
            positional_encoding=forced_rope,
        )
        sequence = block(
            sequence, modulation_encoding=vec, positional_encoding=rope
        )
        drift = _relative_error(trace["single"][index], _keras_numpy(sequence))
        checks.compare(
            f"single block {index}",
            trace["single"][index],
            out,
            tolerance,
            f"free-running {drift:.2e}",
        )
        forced = trace["single"][index]
    checks.compare(
        "final layer",
        expected,
        backbone.final_layer(
            np.ascontiguousarray(forced[:, text_tokens:]), forced_vec
        ),
        tolerance,
    )
    output = backbone.final_layer(
        backbone.strip_text_tokens(sequence, text), vec
    )
    return _keras_numpy(output)


def _check_output(checks, name, expected, actual, tolerance):
    """An end-to-end comparison, reported in full."""
    expected = np.asarray(expected, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    if expected.shape == actual.shape:
        difference = np.abs(expected - actual)
        cosine = float(
            np.sum(expected * actual)
            / max(np.linalg.norm(expected) * np.linalg.norm(actual), 1e-30)
        )
        print(
            f"    output shape {expected.shape}, max |reference| "
            f"{np.abs(expected).max():.4g}"
        )
        print(
            f"    max abs error {difference.max():.3e}, mean abs error "
            f"{difference.mean():.3e}, cosine similarity {cosine:.8f}"
        )
    checks.compare(name, expected, actual, tolerance)


def validate_preset(
    preset_dir, preset, dtype, tolerance=None, sampling_steps=_SAMPLING_STEPS
):
    """Compare a saved preset against the real diffusers model.

    Uses the diffusers-format copy of the same checkpoint as an independent
    reference: if both the architecture and the weight mapping are right,
    the two must agree to floating point noise -- layer by layer, end to
    end, and over a sampling trajectory.

    This reliably catches artifact-level faults -- a mis-mapped, transposed
    or missing tensor, which land around 1e-1 -- and the per-layer checks
    name the layer that has one. It is a weaker detector of subtly wrong
    formulas: an incorrect activation function moves the end-to-end error
    only to ~1e-4. That class is covered instead by the unit tests in
    `keras_hub/src/models/flux/flux_layers_test.py`, which check the layers
    individually.

    Both models run in `dtype`, whatever dtype the preset was saved in, and
    the reference runs and is freed *before* the preset is read back, so
    only one 11.9B parameter model is ever resident (~48GB at float32, ~24GB
    at bfloat16). Validating the saved preset rather than the model in
    memory also covers the save / load round trip.

    Returns the number of checks passed; raises `SystemExit` if any failed.
    """
    spec = PRESETS[preset]
    guidance_embed = spec["guidance_embed"]
    reference_dtype, layer_tolerance, sinusoid_tolerance, output_tolerance = (
        _reference_dtype_and_tolerances(dtype, guidance_embed)
    )
    if tolerance is not None:
        layer_tolerance = sinusoid_tolerance = output_tolerance = tolerance
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
    full_precision = dtype.removeprefix("mixed_") == "float32"
    if full_precision:
        _use_full_float32_matmuls()
    input_sets = [
        _validation_inputs(guidance_embed),
        _validation_inputs(
            guidance_embed,
            _SHAPE_CHECK_TIMESTEPS,
            _SHAPE_CHECK_GRID,
            _SHAPE_CHECK_TEXT_TOKENS,
            seed=1,
        ),
    ]
    print(
        f"Validating against diffusers `FluxTransformer2DModel` "
        f"({spec['repo_id']}), both in {dtype}, on "
        f"{_describe_inputs(input_sets[0])}."
    )
    print(
        f"  keras {keras.__version__} ({keras.config.backend()} backend), "
        f"torch {torch.__version__}, diffusers "
        f"{importlib.metadata.version('diffusers')}"
    )
    run = _reference_run(spec, reference_dtype, input_sets, sampling_steps)
    print(f"Loading the converted preset from {preset_dir} in {dtype}...")
    keras.config.set_dtype_policy(dtype)
    backbone = FluxBackbone.from_preset(
        os.path.abspath(preset_dir), dtype=dtype
    )
    checks = _Checks(dtype)
    _check_architecture(checks, backbone, run, preset)
    # The replay calls layers eagerly, which on the torch backend would
    # otherwise record an autograd graph holding every activation.
    with torch.no_grad():
        eager = _check_layers(
            checks,
            backbone,
            run["trace"],
            input_sets[0],
            run["outputs"][0],
            layer_tolerance,
            sinusoid_tolerance,
            output_tolerance,
        )
        print(
            "\nEnd to end, `FluxBackbone.predict` on the saved preset "
            f"(tol {output_tolerance:.0e}):"
        )
        predicted = []
        for tensors, expected in zip(input_sets, run["outputs"]):
            print(f"  {_describe_inputs(tensors)}:")
            predicted.append(_keras_output(backbone, tensors, guidance_embed))
            rows, cols = tensors["grid"]
            _check_output(
                checks,
                f"output ({rows}x{cols} grid)",
                expected,
                predicted[-1],
                output_tolerance,
            )
        if sampling_steps:
            sampled = _sample(
                functools.partial(
                    _keras_output, backbone, guidance_embed=guidance_embed
                ),
                input_sets[0],
                sampling_steps,
            )
            # Gated in float32 only: in the low precision dtypes the
            # rounding noise compounds over the steps too.
            checks.compare(
                f"{sampling_steps}-step sampling",
                run["sampled"],
                sampled,
                output_tolerance,
                f"latents after {sampling_steps} Euler steps, t=1 to 0",
                gate=full_precision,
            )
        checks.compare(
            "eager replay vs predict",
            predicted[0],
            eager,
            layer_tolerance,
            gate=False,
        )
    del backbone
    gc.collect()
    return checks.finish()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
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
            "Override the max relative error tolerance of every check "
            "(default: 1e-4 at float32, except 1e-3 for the sinusoidal "
            "embeddings and, in guidance-distilled models, the checks "
            "downstream of the guidance embedding; 5e-2 at float16; 1e-1 "
            "at bfloat16)."
        ),
    )
    parser.add_argument(
        "--skip_sampling",
        action="store_true",
        help=(
            f"Skip the {_SAMPLING_STEPS}-step sampling comparison, which "
            f"runs each model {_SAMPLING_STEPS} more times."
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
    if not args.skip_validation:
        check_validation_dependencies()

    spec = PRESETS[args.preset]
    guidance_embed = spec["guidance_embed"]
    output_dir = os.path.normpath(args.output_dir or args.preset)
    sampling_steps = 0 if args.skip_sampling else _SAMPLING_STEPS

    if args.validate_only:
        passed = validate_preset(
            args.validate_only,
            args.preset,
            args.validation_dtype,
            args.tolerance,
            sampling_steps,
        )
        print(
            f"✅ Preset validated in {args.validation_dtype}: all {passed} "
            "checks passed"
        )
        validated_dir = os.path.normpath(args.validate_only)
        if validated_dir.endswith(".unvalidated"):
            # The staging directory of an earlier run whose validation
            # failed or did not finish.
            print(
                "Move it into place with:\n"
                f"    mv {validated_dir} "
                f"{validated_dir.removesuffix('.unvalidated')}"
            )
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
                f"`{staging_dir}` is left over from an earlier run whose "
                "validation failed or did not finish. Either validate it "
                f"without re-converting (`--validate_only {staging_dir}`, "
                f"then move it to `{output_dir}`), or delete it and re-run."
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
        passed = validate_preset(
            staging_dir,
            args.preset,
            args.validation_dtype,
            args.tolerance,
            sampling_steps,
        )
    except SystemExit:
        # A verdict (the preset failed a check), not an interruption.
        print(f"\nThe unvalidated preset was left in {staging_dir}.")
        raise
    except BaseException:
        # Validation itself broke (e.g. a network error fetching the
        # reference, or Ctrl-C) after the conversion had finished: resume
        # from the saved preset instead of converting again.
        print(
            f"\nValidation did not finish. The converted preset is in "
            f"{staging_dir}; validate it without re-converting:\n"
            f"    python {os.path.relpath(__file__)} --preset {args.preset} "
            f"--validate_only {staging_dir}"
        )
        raise
    os.replace(staging_dir, output_dir)
    print(
        f"✅ Preset validated in {args.validation_dtype}: all {passed} "
        "checks passed"
    )
    print(f"Saved the validated preset to {output_dir}")


if __name__ == "__main__":
    main()
