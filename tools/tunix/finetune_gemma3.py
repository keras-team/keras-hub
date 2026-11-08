"""Export a KerasHub Gemma3 model, check Tunix parity, and run short SFT.

Run with the optional dependencies in requirements.txt in this directory:

    KERAS_BACKEND=jax PYTHONPATH=. python tools/tunix/finetune_gemma3.py

The default is a small random model for a CPU smoke check. To use real weights:

    KERAS_BACKEND=jax PYTHONPATH=. python tools/tunix/finetune_gemma3.py \
        --preset hf://google/gemma-3-270m-it --output_dir /tmp/gemma3-tunix
"""

import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import keras
import numpy as np
import optax
from flax import nnx
from orbax import checkpoint as ocp
from tunix.models.gemma3 import model as gemma3
from tunix.models.gemma3 import params_safetensors
from tunix.sft import peft_trainer
from tunix.sft import utils

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM


def create_model(preset):
    if preset:
        if preset != "hf://google/gemma-3-270m-it":
            raise ValueError("This example supports Gemma3 270M IT only.")
        return Gemma3CausalLM.from_preset(preset, dtype="float32")
    # Six layers exercise both local and global attention. The window is
    # shorter than the input to check local masking as well as weight loading.
    backbone = Gemma3Backbone(
        vocabulary_size=64,
        image_size=None,
        num_layers=6,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=32,
        intermediate_dim=64,
        head_dim=16,
        use_post_ffw_norm=True,
        use_post_attention_norm=True,
        use_sliding_window_attention=True,
        sliding_window_size=4,
        dtype="float32",
    )
    return Gemma3CausalLM(backbone=backbone, preprocessor=None)


def model_inputs(batch):
    mask = batch.input_tokens != 0
    return {
        "input_tokens": batch.input_tokens,
        "input_mask": batch.input_mask,
        "positions": utils.build_positions_from_mask(mask),
        "attention_mask": utils.make_causal_attn_mask(mask),
    }


def logits(model, batch):
    inputs = model_inputs(batch)
    values, _ = model(
        inputs["input_tokens"],
        inputs["positions"],
        None,
        inputs["attention_mask"],
    )
    return values


def loss(model, batch):
    values = logits(model, batch)[:, :-1].astype(jnp.float32)
    targets = batch.input_tokens[:, 1:]
    weights = batch.input_mask[:, 1:]
    losses = optax.softmax_cross_entropy_with_integer_labels(values, targets)
    return float(jnp.sum(losses * weights) / jnp.sum(weights))


def run(preset, output_dir, steps):
    if keras.config.backend() != "jax":
        raise ValueError("Set KERAS_BACKEND=jax before running this example.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    keras.utils.set_random_seed(42)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = output_dir / "export"
    keras_model = create_model(preset)
    backbone = keras_model.backbone
    keras_model.export_to_transformers(str(export_dir))

    if preset:
        # Prompt tokens are inputs, while only completion tokens contribute
        # to the supervised loss. This is a plumbing demo, not a benchmark.
        tokenizer = keras_model.preprocessor.tokenizer
        prompt = tokenizer(
            "<start_of_turn>user\nSay hello.<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        completion = tokenizer("Hello!<end_of_turn>")
        prompt = np.asarray(prompt, dtype="int32").reshape(-1)
        completion = np.asarray(completion, dtype="int32").reshape(-1)
        tokens = np.concatenate(
            ([tokenizer.start_token_id], prompt, completion)
        )[None].astype("int32")
        mask = np.zeros_like(tokens, dtype="bool")
        mask[:, 1 + len(prompt) :] = True
    else:
        tokens = np.array([[2, 3, 4, 5, 6, 7, 8, 9]], dtype="int32")
        mask = np.array([[False] * 4 + [True] * 4])
    batch = peft_trainer.TrainingInput(input_tokens=tokens, input_mask=mask)
    keras_logits = np.asarray(
        keras.ops.convert_to_numpy(
            keras_model(
                {
                    "token_ids": tokens,
                    "padding_mask": np.ones_like(tokens),
                }
            )
        )
    )

    config = gemma3.ModelConfig(
        num_layers=backbone.num_layers,
        num_embed=backbone.vocabulary_size,
        embed_dim=backbone.hidden_dim,
        hidden_dim=backbone.intermediate_dim,
        num_heads=backbone.num_query_heads,
        head_dim=backbone.head_dim,
        num_kv_heads=backbone.num_key_value_heads,
        sliding_window_size=backbone.sliding_window_size,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        param_dtype=jnp.float32,
    )
    mesh = jax.make_mesh(
        (1, 1),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )
    with jax.set_mesh(mesh):
        tunix_model = params_safetensors.create_model_from_safe_tensors(
            str(export_dir), config, mesh, dtype=jnp.float32
        )
        before = np.asarray(logits(tunix_model, batch))
        difference = np.abs(keras_logits - before)
        np.testing.assert_allclose(keras_logits, before, atol=1e-4, rtol=1e-4)
        state = nnx.state(tunix_model, nnx.Param)
        parameter_count = sum(x.size for x in jax.tree.leaves(state))
        assert parameter_count == backbone.count_params()
        original_parameters = [np.array(x) for x in jax.tree.leaves(state)]
        initial_loss = loss(tunix_model, batch)
        trainer = peft_trainer.PeftTrainer(
            model=tunix_model,
            optimizer=optax.sgd(learning_rate=1e-3),
            training_config=peft_trainer.TrainingConfig(
                eval_every_n_steps=steps + 1,
                max_steps=steps,
            ),
        ).with_gen_model_input_fn(model_inputs)
        trainer.train(train_ds=[batch] * steps)
        final_loss = loss(tunix_model, batch)
        after = np.asarray(logits(tunix_model, batch))
        assert np.isfinite(initial_loss) and np.isfinite(final_loss)
        assert not np.array_equal(before, after), (
            "Training did not change logits."
        )
        assert final_loss < initial_loss, (initial_loss, final_loss)
        trained_state = nnx.state(tunix_model, nnx.Param)
        parameter_change = max(
            float(np.max(np.abs(old - np.asarray(new))))
            for old, new in zip(
                original_parameters, jax.tree.leaves(trained_state)
            )
        )
        assert parameter_change > 0, "Training did not update parameters."
        # Save the trained NNX parameters separately from the original export.
        checkpoint_path = str((output_dir / "trained_parameters").resolve())
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(checkpoint_path, trained_state, force=True)
            checkpointer.wait_until_finished()
            reloaded = params_safetensors.create_model_from_safe_tensors(
                str(export_dir), config, mesh, dtype=jnp.float32
            )
            restored = checkpointer.restore(
                checkpoint_path, target=nnx.state(reloaded, nnx.Param)
            )
            nnx.update(reloaded, restored)
            restored_logits = np.asarray(logits(reloaded, batch))
        np.testing.assert_array_equal(after, restored_logits)

    results = {
        "preset": preset or "random Gemma3 CPU smoke check",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("keras", "google-tunix", "jax", "flax", "safetensors")
        },
        "devices": [str(device) for device in jax.devices()],
        "parameters": parameter_count,
        "max_absolute_logit_error": float(difference.max()),
        "mean_absolute_logit_error": float(difference.mean()),
        "training_steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "max_logit_change_after_training": float(
            np.max(np.abs(after - before))
        ),
        "max_parameter_change": parameter_change,
        "trained_checkpoint_logits_match": True,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir:
        run(args.preset, args.output_dir, args.steps)
    else:
        with tempfile.TemporaryDirectory(prefix="gemma3-tunix-") as directory:
            run(args.preset, Path(directory), args.steps)


if __name__ == "__main__":
    main()
