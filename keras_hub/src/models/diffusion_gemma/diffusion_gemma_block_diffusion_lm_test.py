from unittest.mock import patch

import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_block_diffusion_lm import (  # noqa: E501
    DiffusionGemmaBlockDiffusionLM,
)
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_block_diffusion_lm_preprocessor import (  # noqa: E501
    DiffusionGemmaBlockDiffusionLMPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.samplers.entropy_bound_sampler import EntropyBoundSampler
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class DiffusionGemmaBlockDiffusionLMTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()
        vocab_size = self.tokenizer.vocabulary_size()

        self.preprocessor = DiffusionGemmaBlockDiffusionLMPreprocessor(
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
        }
        self.backbone = DiffusionGemmaBackbone(**backbone_kwargs)
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

        # === Vision-enabled model (image_converter + vision_encoder) ===
        self.image_converter = Gemma4ImageConverter(
            image_size=(16, 16),
            patch_size=4,
        )
        self.vision_preprocessor = DiffusionGemmaBlockDiffusionLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=24,
            canvas_length=4,
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
        )
        vision_encoder = Gemma4VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        vision_backbone_kwargs = dict(backbone_kwargs)
        vision_backbone_kwargs["vision_encoder"] = vision_encoder
        self.vision_backbone = DiffusionGemmaBackbone(**vision_backbone_kwargs)
        self.vision_init_kwargs = {
            "backbone": self.vision_backbone,
            "preprocessor": self.vision_preprocessor,
        }
        self.vision_train_data = (
            {
                "prompts": [
                    "the <|image|> fox",
                    "the <|image|> fox",
                ],
                "responses": ["the earth is round", "the earth is round"],
                "pixel_values": np.ones([2, 1, 16, 3 * 4 * 4], dtype="float32"),
                "pixel_position_ids": np.ones([2, 1, 16, 2], dtype="int32"),
            },
        )
        self.vision_input_data = self.vision_preprocessor(
            *self.vision_train_data
        )[0]

    def test_call_shape(self):
        model = DiffusionGemmaBlockDiffusionLM(**self.init_kwargs)
        logits = model(self.input_data)
        # (batch=2, seq_len=8, vocab_size)
        self.assertEqual(logits.shape, (2, 8, self.tokenizer.vocabulary_size()))

    @parameterized.named_parameters(
        ("text_only", "text_only"), ("text_and_vision", "text_and_vision")
    )
    def test_task_basics(self, modality_type):
        if modality_type == "text_and_vision":
            init_kwargs = self.vision_init_kwargs
            train_data = self.vision_train_data
            seq_len = self.vision_preprocessor.sequence_length
        else:
            init_kwargs = self.init_kwargs
            train_data = self.train_data
            seq_len = self.preprocessor.sequence_length

        self.run_task_test(
            cls=DiffusionGemmaBlockDiffusionLM,
            init_kwargs=init_kwargs,
            train_data=train_data,
            expected_output_shape=(
                2,
                seq_len,
                self.tokenizer.vocabulary_size(),
            ),
        )

    def test_generate_single_string(self):
        model = DiffusionGemmaBlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        output = model.generate("the quick brown fox")
        self.assertIsInstance(output, str)

    def test_generate_batched_strings(self):
        model = DiffusionGemmaBlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        outputs = model.generate(["the quick brown fox", "the quick brown fox"])
        self.assertEqual(len(outputs), 2)
        for out in outputs:
            self.assertIsInstance(out, str)

    def test_generate_with_image(self):
        # Exercises the vision-interleaving branch of _encode_prompt, which
        # is only reachable through generate()/generate_step(), not through
        # a plain functional model(...) call.
        model = DiffusionGemmaBlockDiffusionLM(**self.vision_init_kwargs)
        model.compile(sampler=self.sampler)
        output = model.generate(
            {
                "prompts": "the <|image|> fox",
                "images": np.ones((16, 16, 3), dtype="float32"),
            }
        )
        self.assertIsInstance(output, str)

    def test_generate_without_preprocessor(self):
        model = DiffusionGemmaBlockDiffusionLM(
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
        model = DiffusionGemmaBlockDiffusionLM(
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
        model = DiffusionGemmaBlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            model.generate("the quick brown fox", max_length=0)

    def test_generate_step_stops_and_pads_each_sequence(self):
        model = DiffusionGemmaBlockDiffusionLM(
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

    def test_generate_compilation_is_cached(self):
        model = DiffusionGemmaBlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        model.generate("the quick brown fox")
        first_fn = model.generate_function
        model.generate("the quick brown fox")
        second_fn = model.generate_function
        self.assertEqual(first_fn, second_fn)

    def test_compile_resets_generate_function(self):
        model = DiffusionGemmaBlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        model.generate("the quick brown fox")
        model.compile(sampler=self.sampler)
        self.assertIsNone(model.generate_function)

    def test_default_sampler_resolves_by_name(self):
        model = DiffusionGemmaBlockDiffusionLM(
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

        model = DiffusionGemmaBlockDiffusionLM(
            **self.init_kwargs,
            sampler=sampler,
        )

        self.assertIs(model.sampler, sampler)

    def test_compile_without_sampler_preserves_constructor_sampler(self):
        # A plain compile() call (e.g. before fine-tuning) must not silently
        # reset a sampler configured via the constructor or from_config().
        sampler = EntropyBoundSampler(entropy_bound=0.2)
        model = DiffusionGemmaBlockDiffusionLM(
            **self.init_kwargs,
            sampler=sampler,
        )
        model.compile(optimizer="adam")
        self.assertIs(model.sampler, sampler)

    def test_constructor_rejects_non_diffusion_sampler(self):
        # Constructing (not just compiling) with a standard autoregressive
        # sampler must fail clearly, since the sampler is resolved directly
        # in __init__, not only in compile().
        with self.assertRaisesRegex(ValueError, "DiffusionSampler"):
            DiffusionGemmaBlockDiffusionLM(
                **self.init_kwargs,
                sampler="greedy",
            )

    def test_serialization_custom_generation_config(self):
        model = DiffusionGemmaBlockDiffusionLM(
            **self.init_kwargs,
            canvas_length=8,
            max_denoising_steps=2,
            t_min=0.2,
            t_max=0.7,
            sampler=EntropyBoundSampler(entropy_bound=0.2),
        )
        self.run_serialization_test(model)

    @parameterized.named_parameters(
        ("text_only", "text_only"), ("text_and_vision", "text_and_vision")
    )
    def test_saved_model(self, modality_type):
        if modality_type == "text_and_vision":
            init_kwargs = self.vision_init_kwargs
            input_data = self.vision_input_data
        else:
            init_kwargs = self.init_kwargs
            input_data = self.input_data

        self.run_model_saving_test(
            cls=DiffusionGemmaBlockDiffusionLM,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )

    def test_encoder_scalar_not_applied_in_decode_step(self):
        """_decode_canvas_step always uses layer_scalar (decoder scalar)."""
        model = DiffusionGemmaBlockDiffusionLM(
            **self.init_kwargs,
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
        for layer in self.backbone.transformer_layers:
            layer.layer_scalar.assign(1.0)
            layer.encoder_layer_scalar.assign(99.0)
        out_decoder_scalar = ops.convert_to_numpy(
            ops.stop_gradient(
                model._decode_canvas_step(
                    canvas_embeds, encoder_kv_cache, prompt_length
                )
            )
        )

        # Now set encoder_layer_scalar=1.0 too — decode output should match.
        for layer in self.backbone.transformer_layers:
            layer.encoder_layer_scalar.assign(1.0)
        out_same_scalar = ops.convert_to_numpy(
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
        for preset in DiffusionGemmaBlockDiffusionLM.presets:
            self.run_preset_test(
                cls=DiffusionGemmaBlockDiffusionLM,
                preset=preset,
                input_data=self.input_data,
            )
