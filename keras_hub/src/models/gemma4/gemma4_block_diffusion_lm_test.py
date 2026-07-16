import os

import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm import (
    Gemma4BlockDiffusionLM,
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
        self.sampler = EntropyBoundSampler(vocabulary_size=vocab_size)

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
        # run_task_test relies on task._inputs_struct, which Keras sets at
        # construction only for functional models. Gemma4BlockDiffusionLM
        # defines its own call(), making it a subclassed model where
        # _inputs_struct is absent until after the first forward pass.
        # Replicate the key fit/predict/serialization checks manually.
        task = Gemma4BlockDiffusionLM(**self.init_kwargs)
        self.run_serialization_test(task)

        # Pass raw prompts/responses — Task.preprocess_samples routes through
        # the preprocessor automatically, so pre-processed data would cause a
        # double-preprocessing KeyError.
        raw_x = self.train_data[0]

        # Predict through task + preprocessor pipeline; verify output shape.
        output = task.predict(raw_x)
        self.assertEqual(output.shape, (2, 8, self.tokenizer.vocabulary_size()))

        # Fit through task + preprocessor pipeline for one epoch.
        task.fit(raw_x, epochs=1)

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
        output = model.generate(inputs)
        canvas = np.array(output)
        # Shape: (1, canvas_length) or (canvas_length,) after scalar squeeze.
        self.assertEqual(canvas.shape[-1], self.preprocessor.canvas_length)

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

    @parameterized.named_parameters(
        ("default_canvas", {}),
        ("custom_canvas_length", {"canvas_length": 8}),
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
