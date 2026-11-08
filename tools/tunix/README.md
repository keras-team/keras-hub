# Gemma3 checkpoint export and Tunix fine-tuning

This example loads the pretrained `google/gemma-3-270m-it` checkpoint in
KerasHub, exports it to Hugging Face safetensors, loads it in Tunix, compares
logits, and runs two supervised fine-tuning steps with `PeftTrainer`.
It reuses existing APIs and adds no required dependencies to KerasHub.

## Setup

Use a separate Python 3.12 environment. From the repository root:

```shell
python3.12 -m venv .venv-tunix
source .venv-tunix/bin/activate
python -m pip install -e . -r tools/tunix/requirements.txt
```

Accept the Gemma model terms on Hugging Face and authenticate with an account
that can download `google/gemma-3-270m-it`. Keep authentication tokens out of
the notebook. The optional requirements pin the integration dependencies.
JAX uses CPU here; accelerator runs need matching JAX accelerator packages
for the same version. The example uses one device.

## Run

```shell
KERAS_BACKEND=jax JAX_PLATFORMS=cpu PYTHONPATH=. \
    python tools/tunix/finetune_gemma3.py \
    --output_dir /tmp/gemma3-270m-tunix
```

The checkpoint's tokenizer encodes a single instruction/completion pair:
"Say hello." / "Hello!". Only the completion and its turn-ending token
contribute to loss. The script checks:

- Matching parameter counts: 268,098,176.
- Float32 logits within `atol=1e-3, rtol=1e-4`, with identical argmax tokens.
- Two full-fine-tuning steps update parameters and reduce the same-batch loss.
- Trained parameters saved with Orbax reproduce logits exactly on reload.

The CPU run recorded maximum/mean logits errors of `3.28e-4` / `4.96e-5`.
A separate HF float32 model loaded from the same export gave maximum errors
of `8.24e-4` for KerasHub and `6.52e-4` for Tunix, also with identical argmax
predictions. These measurements cover this input and environment.

SGD uses `--learning_rate 1e-5` by default. On the single training example,
loss decreased from `4.57253` to `0.01496`. This checks gradient computation
and parameter updates; it does not measure generalization or establish a
fine-tuning recipe. `--steps` and `--logit_atol` are configurable as well.
A batch with no active next-token targets raises `ValueError` before model
evaluation or training. The notebook includes target-mask regression checks.

Assertions abort the script if a check fails. `results.json` records errors,
losses, versions, and the device. `export/` contains the original KerasHub
export and `trained_parameters/` contains trained Tunix parameters. Use a
dedicated output directory: rerunning replaces these artifacts. Allow several
GB of memory for model copies, gradients, and checkpoint validation.

The notebook includes an executed CPU run with the real checkpoint. The
example covers text-only Gemma3, unpadded input, and one training batch.
Gemma4, LoRA, distributed training, training quality evaluation, and importing
trained weights back into KerasHub are separate follow-up work.
