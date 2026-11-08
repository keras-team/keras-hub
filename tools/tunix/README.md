# Gemma3 checkpoint export and Tunix fine-tuning

This example exports a KerasHub Gemma3 causal LM to Hugging Face safetensors,
loads it into Tunix, compares logits on identical tokens, and runs a short
supervised fine-tuning loop with Tunix's `PeftTrainer`. It reuses existing
APIs and adds no required dependencies to KerasHub.

Use a separate Python 3.12 environment. From the repository root:

```shell
python3.12 -m venv .venv-tunix
source .venv-tunix/bin/activate
python -m pip install -e . -r tools/tunix/requirements.txt
```

The requirements pin the optional integration dependencies used for this
example. JAX uses the CPU installation here; for accelerator runs, install
the matching JAX accelerator packages for the same version separately.
The example uses one device, without distributed training.

## CPU smoke check

```shell
KERAS_BACKEND=jax JAX_PLATFORMS=cpu PYTHONPATH=. \
    python tools/tunix/finetune_gemma3.py \
    --output_dir /tmp/gemma3-tunix-smoke
```

Without `--preset`, the script builds a small randomly initialized model with
six layers, including local and global attention. The local window is shorter
than the input sequence. Token IDs and a completion mask are synthetic, so this
check requires no checkpoint download or tokenizer. It verifies:

- Equal parameter counts after checkpoint conversion.
- Logits match within `atol=1e-4, rtol=1e-4`, in float32.
- Two SFT steps update model parameters and reduce loss on the same batch.
- Trained parameters saved with Orbax reproduce the trained logits on reload.

A batch without active next-token targets raises `ValueError` before model
evaluation or training, rather than reporting a zero or non-finite loss.

This is a test of the integration, not pretrained model quality. Assertions
abort the script if a check fails. `results.json` contains numerical errors,
losses, package versions, and the device used. `export/` holds the original
KerasHub export; `trained_parameters/` holds the trained Tunix parameters.
Use a dedicated output directory: rerunning replaces these artifacts.

## Gemma3 270M IT

Accept the Gemma model terms on Hugging Face and authenticate with an account
that can download `google/gemma-3-270m-it`. Then run:

```shell
KERAS_BACKEND=jax PYTHONPATH=. \
    python tools/tunix/finetune_gemma3.py \
    --preset hf://google/gemma-3-270m-it \
    --output_dir /tmp/gemma3-270m-tunix
```

This mode uses the checkpoint's tokenizer and a short prompt/completion pair.
Only completion tokens contribute to the loss. It performs full fine-tuning
with SGD for two steps in float32. Model copies, gradients, and checkpoint
validation require more memory than loading the 270M checkpoint alone.
Access failures are reported directly; the script does not fall back to random
weights. The pretrained mode still needs validation with authorized model
access; the included notebook output covers only the CPU smoke check.

The example currently covers text-only Gemma3, unpadded input, and one training
batch. Gemma4, LoRA, training quality evaluation, and importing Tunix-trained
weights back into KerasHub are separate follow-up work.
