r"""Convert the official FLUX.2 [klein] checkpoint to a KerasHub preset.

Usage:
    python tools/checkpoint_conversion/convert_flux2_checkpoints.py \
        --preset flux2_klein_4b

Steps:

1. Download the Black Forest Labs single-file checkpoint (for klein 4B,
   `flux-2-klein-4b.safetensors`, ~7.8GB of bfloat16 weights) and the
   diffusers `transformer/config.json` of the same repo.
2. Build `Flux2Backbone` from that config and port every weight with
   `keras_hub.src.utils.transformers.convert_flux2`. The run fails unless
   every Keras weight is assigned exactly once, every checkpoint tensor is
   read, and the parameter counts agree.
3. Save the preset to `<output_dir>.unvalidated`.
4. Reload the *saved* preset and compare it with diffusers'
   `Flux2Transformer2DModel`, loaded from the repo's own diffusers-format
   weights (an independent conversion of the same checkpoint), on inputs
   built the way `Flux2KleinPipeline` builds them, at two input shapes.
5. Only then move it to `<output_dir>`.

Validation runs both models in `--validation_dtype` (float32 by default),
one after the other, so only one is resident at a time: ~16GB of host RAM
at float32, ~8GB at bfloat16. It needs `diffusers>=0.37` (FLUX.2 [klein]
support) and `accelerate`.

To convert and validate in separate processes:

    python tools/checkpoint_conversion/convert_flux2_checkpoints.py \
        --preset flux2_klein_4b --skip_validation
    python tools/checkpoint_conversion/convert_flux2_checkpoints.py \
        --preset flux2_klein_4b --validate_only flux2_klein_4b

Conversion runs on the host; set FLUX2_CONVERT_ALLOW_GPU=1 to allow a GPU.
"""

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if os.environ.get("FLUX2_CONVERT_ALLOW_GPU") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors import safe_open  # noqa: E402

from keras_hub.src.models.flux2.flux2_backbone import (  # noqa: E402
    Flux2Backbone,
)
from keras_hub.src.utils.transformers import convert_flux2  # noqa: E402

PRESETS = {
    "flux2_klein_4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "filename": "flux-2-klein-4b.safetensors",
        "size_gb": 8,
    },
}

# Inputs are built as `Flux2KleinPipeline` builds them: image tokens at
# `(0, row, col, 0)`, text tokens at `(0, 0, 0, index)`. Two shapes, with
# different aspect ratios and prompt lengths, so a preset pinned to one
# sequence length, or a transposed position axis, cannot pass.
_VALIDATION_CASES = (
    # (batch, latent rows, latent columns, text tokens)
    (2, 8, 12, 16),
    (1, 6, 4, 7),
)

# Max relative L2 error of the output. The checkpoint is bfloat16, so a
# float32 comparison resolves the weight mapping to float32 noise; in
# bfloat16 every op rounds and two correct implementations drift apart.
_TOLERANCES = {"float32": 1e-4, "float16": 5e-2, "bfloat16": 1e-1}
_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class TorchSafetensorLoader:
    """A `SafetensorLoader`-compatible reader for bfloat16 checkpoints.

    `keras_hub`'s `SafetensorLoader` opens files with `framework="np"`, and
    numpy has no bfloat16, so it cannot read the official checkpoint. This
    reads through torch and upcasts one tensor at a time. It also records
    which tensors were read and which Keras variables were written, for
    `check_conversion`.
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

    def get_tensor(self, hf_weight_key):
        tensor = self._file.get_tensor(hf_weight_key)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()

    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        if hf_weight_key not in self.keys:
            raise KeyError(f"Checkpoint is missing tensor `{hf_weight_key}`.")
        tensor = self.get_tensor(hf_weight_key)
        if hook_fn is not None:
            tensor = hook_fn(tensor, list(keras_variable.shape))
        keras_variable.assign(tensor)
        self.used.add(hf_weight_key)
        entry = self.assigned.setdefault(
            id(keras_variable), [keras_variable.path, 0]
        )
        entry[1] += 1

    def num_params(self):
        return sum(
            int(np.prod(self._file.get_slice(key).get_shape()))
            for key in self.keys
        )


def check_conversion(backbone, loader):
    """Fail unless the weight mapping was complete and one-to-one."""
    never = [w.path for w in backbone.weights if id(w) not in loader.assigned]
    twice = [path for path, count in loader.assigned.values() if count > 1]
    unused = sorted(loader.keys - loader.used)
    keras_params = backbone.count_params()
    checkpoint_params = loader.num_params()
    problems = []
    if never:
        problems.append(f"{len(never)} Keras weights never assigned: {never}")
    if twice:
        problems.append(f"{len(twice)} Keras weights assigned twice: {twice}")
    if unused:
        problems.append(f"{len(unused)} checkpoint tensors unread: {unused}")
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
        f"  All {len(backbone.weights)} Keras weights were assigned exactly "
        f"once from the checkpoint's {len(loader.keys)} tensors "
        f"({keras_params:,} parameters)."
    )


def load_transformer_config(repo_id):
    path = hf_hub_download(repo_id, "transformer/config.json")
    with open(path) as f:
        return json.load(f)


def download_checkpoint(spec):
    free_gb = shutil.disk_usage(_existing_ancestor(_hf_cache())).free / 1e9
    if free_gb < spec["size_gb"] + 1:
        raise SystemExit(
            f"Only {free_gb:.1f}GB free for the Hugging Face cache, but "
            f"{spec['filename']} needs ~{spec['size_gb']}GB. Free up space "
            "or point HF_HOME at a larger volume."
        )
    print(f"Downloading {spec['repo_id']}/{spec['filename']}...")
    return hf_hub_download(spec["repo_id"], spec["filename"])


def _hf_cache():
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE


def _existing_ancestor(path):
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    return path


def check_validation_dependencies():
    missing = [
        name
        for name in ("diffusers", "accelerate")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(
            f"Validation needs {', '.join(missing)}. Install with\n"
            f"    {sys.executable} -m pip install 'diffusers>=0.37' "
            "accelerate\nor convert now and validate later with "
            "--skip_validation / --validate_only."
        )


def validation_inputs(config, batch, rows, cols, text_len, seed):
    rng = np.random.default_rng(seed)
    channels = config["image_shape"][-1]
    text_dim = config["text_shape"][-1]
    hh, ww = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    zeros = np.zeros_like(hh)
    image_ids = np.stack([zeros, hh, ww, zeros], axis=-1).reshape(-1, 4)
    index = np.arange(text_len)
    text_ids = np.stack([0 * index, 0 * index, 0 * index, index], axis=-1)
    inputs = {
        "image": rng.standard_normal((batch, rows * cols, channels)),
        "image_ids": np.broadcast_to(image_ids, (batch, rows * cols, 4)),
        "text": rng.standard_normal((batch, text_len, text_dim)),
        "text_ids": np.broadcast_to(text_ids, (batch, text_len, 4)),
        "timesteps": np.linspace(1.0, 0.25, batch),
    }
    if config["guidance_embed"]:
        inputs["guidance"] = np.full((batch,), 4.0)
    return {k: np.ascontiguousarray(v, "float32") for k, v in inputs.items()}


def keras_outputs(preset_dir, dtype, input_sets):
    keras.config.set_dtype_policy(dtype)
    backbone = Flux2Backbone.from_preset(preset_dir, dtype=dtype)
    outputs = [
        np.asarray(
            backbone.predict(
                inputs, batch_size=len(inputs["image"]), verbose=0
            ),
            dtype="float32",
        )
        for inputs in input_sets
    ]
    del backbone
    gc.collect()
    return outputs


def reference_outputs(repo_id, dtype, input_sets):
    from diffusers import Flux2Transformer2DModel

    torch_dtype = _TORCH_DTYPES[dtype]
    reference = Flux2Transformer2DModel.from_pretrained(
        repo_id, subfolder="transformer", torch_dtype=torch_dtype
    ).eval()
    outputs = []
    with torch.no_grad():
        for inputs in input_sets:

            def to_torch(name, cast=True):
                value = torch.from_numpy(inputs[name])
                return value.to(torch_dtype) if cast else value

            output = reference(
                hidden_states=to_torch("image"),
                encoder_hidden_states=to_torch("text"),
                timestep=to_torch("timesteps"),
                # Position ids stay float32, as in the pipeline.
                img_ids=to_torch("image_ids", cast=False),
                txt_ids=to_torch("text_ids", cast=False),
                guidance=(
                    to_torch("guidance") if "guidance" in inputs else None
                ),
                return_dict=False,
            )[0]
            outputs.append(output.float().numpy())
    del reference
    gc.collect()
    return outputs


def validate_preset(preset_dir, spec, dtype, tolerance):
    with open(os.path.join(preset_dir, "config.json")) as f:
        config = json.load(f)["config"]
    input_sets = [
        validation_inputs(config, *case, seed=i)
        for i, case in enumerate(_VALIDATION_CASES)
    ]
    print(f"Running the KerasHub preset in {dtype}...")
    actual = keras_outputs(os.path.abspath(preset_dir), dtype, input_sets)
    print(f"Running diffusers `Flux2Transformer2DModel` in {dtype}...")
    expected = reference_outputs(spec["repo_id"], dtype, input_sets)

    tolerance = tolerance or _TOLERANCES[dtype]
    failed = False
    for case, want, got in zip(_VALIDATION_CASES, expected, actual):
        if want.shape != got.shape:
            raise SystemExit(f"Shape mismatch: {got.shape} vs {want.shape}")
        rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        max_abs = float(np.max(np.abs(got - want)))
        ok = rel < tolerance
        failed |= not ok
        print(
            f"  {'PASS' if ok else 'FAIL'} (batch, rows, cols, text)={case}: "
            f"relative L2 {rel:.2e} (tol {tolerance:.0e}), "
            f"max abs {max_abs:.2e}"
        )
    if failed:
        raise SystemExit("❌ The preset does not match diffusers.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset", default="flux2_klein_4b", choices=sorted(PRESETS)
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write the preset. Defaults to ./<preset>.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(_TOLERANCES),
        help="Dtype the preset is saved in. bfloat16 is lossless.",
    )
    parser.add_argument(
        "--validation_dtype",
        default="float32",
        choices=sorted(_TOLERANCES),
        help="Dtype both models run in for validation.",
    )
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument(
        "--validate_only",
        default=None,
        metavar="PRESET_DIR",
        help="Skip conversion and validate an already saved preset.",
    )
    args = parser.parse_args()

    if args.validate_only and args.skip_validation:
        parser.error("--validate_only and --skip_validation are exclusive.")
    if not args.skip_validation:
        check_validation_dependencies()

    spec = PRESETS[args.preset]
    output_dir = os.path.normpath(args.output_dir or args.preset)

    if args.validate_only:
        validate_preset(
            args.validate_only, spec, args.validation_dtype, args.tolerance
        )
        print(f"✅ {args.validate_only} matches diffusers.")
        return

    staging_dir = f"{output_dir}.unvalidated"
    for path in (output_dir, staging_dir):
        if os.path.exists(path) and os.listdir(path):
            raise SystemExit(
                f"`{path}` already exists. Validate it with "
                f"`--validate_only {path}` or remove it."
            )

    keras.config.set_dtype_policy(args.dtype)
    hf_config = load_transformer_config(spec["repo_id"])
    backbone = Flux2Backbone(**convert_flux2.convert_backbone_config(hf_config))
    checkpoint_path = download_checkpoint(spec)

    # `convert_flux2` is written against `SafetensorLoader`'s interface,
    # which resolves files relative to a preset directory. Link the cached
    # checkpoint into a temporary one rather than copying ~8GB.
    with tempfile.TemporaryDirectory() as tmp_dir:
        linked = os.path.join(tmp_dir, "model.safetensors")
        os.symlink(os.path.abspath(checkpoint_path), linked)
        print("Converting weights...")
        with TorchSafetensorLoader(linked) as loader:
            convert_flux2.convert_weights(backbone, loader, hf_config)
            check_conversion(backbone, loader)

    target_dir = output_dir if args.skip_validation else staging_dir
    print(f"Saving preset to {target_dir}")
    backbone.save_to_preset(target_dir)
    del backbone
    gc.collect()

    if args.skip_validation:
        print(
            f"⚠️  {output_dir} is UNVALIDATED. Do not upload it before "
            f"`--validate_only {output_dir}` passes."
        )
        return

    validate_preset(staging_dir, spec, args.validation_dtype, args.tolerance)
    os.replace(staging_dir, output_dir)
    print(f"✅ Validated preset saved to {output_dir}")


if __name__ == "__main__":
    main()
