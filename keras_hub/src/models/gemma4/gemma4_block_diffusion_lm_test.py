import os
from unittest.mock import patch

import keras
import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm import (
    Gemma4BlockDiffusionLM,
)
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm_layers import (
    Gemma4BlockDiffusionSelfConditioning,
)
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm_preprocessor import (
    Gemma4BlockDiffusionLMPreprocessor,
)
from keras_hub.src.samplers.entropy_bound_sampler import EntropyBoundSampler
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma4BlockDiffusionLMTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()
        vocab_size = self.tokenizer.vocabulary_size()

        self.preprocessor = Gemma4BlockDiffusionLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
            canvas_length=4,
        )

        backbone_kwargs = {
            "vocabulary_size": vocab_size,
            "image_size": 16,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "use_sliding_window_attention": True,
            "sliding_window_size": 16,
            "attention_logit_soft_cap": None,
            "final_logit_soft_cap": None,
            "vision_encoder": None,
            "has_diffusion_self_conditioning": True,
        }
        self.backbone = Gemma4Backbone(**backbone_kwargs)
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.sampler = EntropyBoundSampler()

        self.train_data = (
            {
                "prompts": ["the quick brown fox", "the quick brown fox"],
                "responses": ["the earth is round", "the earth is round"],
            },
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_call_shape(self):
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        logits = model(self.input_data)
        # (batch=2, seq_len=8, vocab_size)
        self.assertEqual(logits.shape, (2, 8, self.tokenizer.vocabulary_size()))

    def test_task_basics(self):
        self.run_task_test(
            cls=Gemma4BlockDiffusionLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, self.tokenizer.vocabulary_size()),
        )

    def test_generate_single_string(self):
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        output = model.generate("the quick brown fox")
        self.assertIsInstance(output, str)

    def test_generate_batched_strings(self):
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        outputs = model.generate(["the quick brown fox", "the quick brown fox"])
        self.assertEqual(len(outputs), 2)
        for out in outputs:
            self.assertIsInstance(out, str)

    def test_generate_without_preprocessor(self):
        model = Gemma4BlockDiffusionLM(
            backbone=self.backbone,
            preprocessor=None,
            canvas_length=self.preprocessor.canvas_length,
        )
        model.compile(sampler=self.sampler)
        processed = self.preprocessor.generate_preprocess("the quick brown fox")
        # Add batch dimension.
        inputs = {
            "token_ids": ops.expand_dims(processed["token_ids"], axis=0),
            "padding_mask": ops.expand_dims(processed["padding_mask"], axis=0),
        }
        output = model.generate(inputs, stop_token_ids=None)
        canvas = np.array(output)
        # Shape: (1, canvas_length) or (canvas_length,) after scalar squeeze.
        self.assertEqual(canvas.shape[-1], self.preprocessor.canvas_length)

    @parameterized.parameters(2, 4, 6, 8)
    def test_generate_respects_max_length(self, max_length):
        model = Gemma4BlockDiffusionLM(
            backbone=self.backbone,
            preprocessor=None,
            canvas_length=self.preprocessor.canvas_length,
        )
        model.compile(sampler=self.sampler, run_eagerly=True)
        processed = self.preprocessor.generate_preprocess("the quick brown fox")
        inputs = {
            "token_ids": ops.expand_dims(processed["token_ids"], axis=0),
            "padding_mask": ops.expand_dims(processed["padding_mask"], axis=0),
        }

        output = model.generate(
            inputs, max_length=max_length, stop_token_ids=None
        )

        self.assertEqual(np.array(output).shape, (1, max_length))

    def test_generate_rejects_non_positive_max_length(self):
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            model.generate("the quick brown fox", max_length=0)

    def test_generate_step_stops_and_pads_each_sequence(self):
        model = Gemma4BlockDiffusionLM(
            backbone=self.backbone,
            preprocessor=None,
            canvas_length=4,
            stop_token_ids=(1, 6),
            pad_token_id=0,
        )
        inputs = {
            "token_ids": ops.ones((2, 4), dtype="int32"),
            "padding_mask": ops.ones((2, 4), dtype="bool"),
        }
        canvases = [
            ops.array([[4, 6, 7, 8], [4, 5, 7, 8]], dtype="int32"),
            ops.array([[9, 10, 11, 12], [9, 1, 11, 12]], dtype="int32"),
        ]

        with (
            patch.object(model, "_encode_prompt", return_value=(None, 4)),
            patch.object(model, "_encode_canvas_as_context", return_value=None),
            patch.object(EntropyBoundSampler, "__call__", side_effect=canvases),
        ):
            output = model.generate_step(
                inputs,
                max_length=8,
                stop_token_ids=model.stop_token_ids,
            )

        self.assertAllEqual(
            output["token_ids"],
            [[4, 6, 0, 0, 0, 0, 0, 0], [4, 5, 7, 8, 9, 1, 0, 0]],
        )
        self.assertAllEqual(
            output["padding_mask"],
            [
                [True, True, False, False, False, False, False, False],
                [True, True, True, True, True, True, False, False],
            ],
        )

    def test_self_conditioning_matmul_uses_embedding_dtype(self):
        layer = Gemma4BlockDiffusionSelfConditioning(
            hidden_dim=4,
            intermediate_dim=8,
            dtype="float16",
        )
        canvas_embeds = ops.ones((1, 2, 4), dtype="float16")
        prev_logits = ops.array(
            [
                [
                    [0.10001, 0.20002, 0.30003, 0.40004, 0.50005, 0.60006],
                    [0.70007, 0.80008, 0.90009, 1.0001, 1.1001, 1.2001],
                ]
            ],
            dtype="float32",
        )
        embedding_weights = ops.ones((6, 4), dtype="float16")

        class _MockEmbedding:
            embeddings = embedding_weights

        object.__setattr__(layer, "_token_embedding_layer", _MockEmbedding())

        operand_dtypes = []
        softmax_inputs = []
        original_matmul = ops.matmul
        original_softmax = ops.softmax

        def record_matmul(x, y):
            operand_dtypes.append(
                (
                    keras.backend.standardize_dtype(x.dtype),
                    keras.backend.standardize_dtype(y.dtype),
                )
            )
            return original_matmul(x, y)

        def record_softmax(x, axis=-1):
            softmax_inputs.append(ops.convert_to_numpy(x))
            return original_softmax(x, axis=axis)

        with (
            patch(
                "keras_hub.src.models.gemma4."
                "gemma4_block_diffusion_lm_layers.ops.matmul",
                side_effect=record_matmul,
            ),
            patch(
                "keras_hub.src.models.gemma4."
                "gemma4_block_diffusion_lm_layers.ops.softmax",
                side_effect=record_softmax,
            ),
        ):
            layer(canvas_embeds, prev_logits)

        self.assertEqual(operand_dtypes, [("float16", "float16")])
        expected_logits = ops.cast(ops.cast(prev_logits, "float16"), "float32")
        self.assertAllEqual(softmax_inputs[0], expected_logits)

    def test_generate_compilation_is_cached(self):
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        model.generate("the quick brown fox")
        first_fn = model.generate_function
        model.generate("the quick brown fox")
        second_fn = model.generate_function
        self.assertEqual(first_fn, second_fn)

    def test_compile_resets_generate_function(self):
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        model.generate("the quick brown fox")
        model.compile(sampler=self.sampler)
        self.assertIsNone(model.generate_function)

    def test_default_sampler_resolves_by_name(self):
        model = Gemma4BlockDiffusionLM(
            **self.init_kwargs,
            canvas_length=4,
        )
        model.compile()

        canvas = ops.zeros((1, 4), dtype="int32")
        logits = ops.zeros(
            (1, 4, self.tokenizer.vocabulary_size()), dtype="float32"
        )
        sampled_canvas = model.sampler(
            next=lambda canvas, prev_logits, step: logits,
            canvas=canvas,
            max_steps=1,
            model=model,
        )

        self.assertEqual(sampled_canvas.shape, canvas.shape)

    def test_constructor_sampler(self):
        sampler = EntropyBoundSampler(
            entropy_bound=0.2,
            confidence_threshold=0.01,
            stability_threshold=2,
        )

        model = Gemma4BlockDiffusionLM(
            **self.init_kwargs,
            sampler=sampler,
        )

        self.assertIs(model.sampler, sampler)

    @parameterized.named_parameters(
        ("default_generation_config", {}),
        (
            "custom_generation_config",
            {
                "canvas_length": 8,
                "max_denoising_steps": 2,
                "t_min": 0.2,
                "t_max": 0.7,
                "sampler": EntropyBoundSampler(entropy_bound=0.2),
            },
        ),
    )
    def test_serialization(self, extra_kwargs):
        model = Gemma4BlockDiffusionLM(**self.init_kwargs, **extra_kwargs)
        self.run_serialization_test(model)

    def test_saved_model(self):
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model_output = model(self.input_data)

        path = os.path.join(self.get_temp_dir(), "model.weights.h5")
        model.save_weights(path)

        restored_model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        # Build the restored model before loading weights.
        _ = restored_model(self.input_data)
        restored_model.load_weights(path)

        # Verify weight count matches.
        self.assertEqual(len(model.weights), len(restored_model.weights))
        for w1, w2 in zip(model.get_weights(), restored_model.get_weights()):
            self.assertAllClose(w1, w2, atol=1e-5, rtol=1e-5)

        # Verify outputs match after weight restore.
        restored_output = restored_model(self.input_data)
        self.assertAllClose(model_output, restored_output, atol=1e-5, rtol=1e-5)

    def test_encoder_layer_scalar_weights_exist(self):
        """has_encoder_layer_scalar=True registers encoder_layer_scalar on
        each block."""

        backbone_kwargs = {
            "vocabulary_size": self.tokenizer.vocabulary_size(),
            "image_size": 16,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "use_sliding_window_attention": True,
            "sliding_window_size": 16,
            "vision_encoder": None,
            "has_encoder_layer_scalar": True,
        }
        backbone = Gemma4Backbone(**backbone_kwargs)
        for layer in backbone.transformer_layers:
            self.assertTrue(
                hasattr(layer, "encoder_layer_scalar"),
                f"{layer.name} missing encoder_layer_scalar",
            )
            self.assertTrue(
                hasattr(layer, "layer_scalar"),
                f"{layer.name} missing layer_scalar",
            )

    def test_encoder_and_decoder_scalars_are_independent(self):
        """encoder_layer_scalar and layer_scalar independently scale
        layer outputs."""
        backbone_kwargs = {
            "vocabulary_size": self.tokenizer.vocabulary_size(),
            "image_size": 16,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "use_sliding_window_attention": True,
            "sliding_window_size": 16,
            "vision_encoder": None,
            "has_encoder_layer_scalar": True,
        }
        backbone = Gemma4Backbone(**backbone_kwargs)

        # Test the scalar effect directly on a single transformer layer.
        # The KV cache is computed *before* the scalar is applied, so RMSNorm
        # in subsequent layers would cancel a scalar visible only in the cache.
        # Testing the layer output directly avoids that cancellation.
        layer = backbone.transformer_layers[0]

        # Fixed input: (batch=1, seq=4, hidden_dim=8).
        x = ops.ones((1, 4, backbone.hidden_dim), dtype="float32")

        # Set encoder scalar ≠ decoder scalar.
        layer.encoder_layer_scalar.assign(2.0)
        layer.layer_scalar.assign(0.5)

        # Encoder pass — must use encoder_layer_scalar (2.0).
        out_enc, _ = layer(x, use_encoder_scalar=True)
        # Decoder pass — must use layer_scalar (0.5).
        out_dec, _ = layer(x, use_encoder_scalar=False)

        # 2.0 ≠ 0.5, so the outputs must differ.
        self.assertNotAllClose(
            np.array(ops.stop_gradient(out_enc)),
            np.array(ops.stop_gradient(out_dec)),
            msg="encoder_layer_scalar had no effect on layer output",
        )

        # Symmetry: when both scalars are equal the outputs must match.
        layer.encoder_layer_scalar.assign(0.5)
        out_enc_equal, _ = layer(x, use_encoder_scalar=True)
        self.assertAllClose(
            np.array(ops.stop_gradient(out_enc_equal)),
            np.array(ops.stop_gradient(out_dec)),
            atol=1e-5,
            msg=(
                "Outputs should match when encoder and decoder scalars "
                "are equal"
            ),
        )

    def test_encoder_scalar_not_applied_in_decode_step(self):
        """_decode_canvas_step always uses layer_scalar (decoder scalar)."""
        backbone_kwargs = {
            "vocabulary_size": self.tokenizer.vocabulary_size(),
            "image_size": 16,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "use_sliding_window_attention": True,
            "sliding_window_size": 16,
            "vision_encoder": None,
            "has_encoder_layer_scalar": True,
            "has_diffusion_self_conditioning": True,
        }
        backbone = Gemma4Backbone(**backbone_kwargs)
        model = Gemma4BlockDiffusionLM(
            backbone=backbone,
            preprocessor=self.preprocessor,
            canvas_length=self.preprocessor.canvas_length,
        )
        model.compile(sampler=self.sampler)

        processed = self.preprocessor.generate_preprocess("the quick brown fox")
        inputs = {
            "token_ids": ops.expand_dims(processed["token_ids"], axis=0),
            "padding_mask": ops.expand_dims(processed["padding_mask"], axis=0),
        }
        encoder_kv_cache, prompt_length = model._encode_prompt(inputs)

        encoder_kv_cache = model._prepare_encoder_cache_for_decoding(
            encoder_kv_cache
        )
        canvas_length = self.preprocessor.canvas_length
        canvas = ops.zeros(
            (1, canvas_length),
            dtype="int32",
        )
        canvas_embeds = model._prepare_canvas_embeds(canvas, None)

        # Run decode step with layer_scalar=1.0, encoder_layer_scalar=99.0
        for layer in backbone.transformer_layers:
            layer.layer_scalar.assign(1.0)
            layer.encoder_layer_scalar.assign(99.0)
        out_decoder_scalar = np.array(
            ops.stop_gradient(
                model._decode_canvas_step(
                    canvas_embeds, encoder_kv_cache, prompt_length
                )
            )
        )

        # Now set encoder_layer_scalar=1.0 too — decode output should match.
        for layer in backbone.transformer_layers:
            layer.encoder_layer_scalar.assign(1.0)
        out_same_scalar = np.array(
            ops.stop_gradient(
                model._decode_canvas_step(
                    canvas_embeds, encoder_kv_cache, prompt_length
                )
            )
        )

        self.assertAllClose(
            out_decoder_scalar,
            out_same_scalar,
            atol=1e-5,
            msg="_decode_canvas_step was affected by encoder_layer_scalar",
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma4BlockDiffusionLM.presets:
            self.run_preset_test(
                cls=Gemma4BlockDiffusionLM,
                preset=preset,
                input_data=self.input_data,
            )
