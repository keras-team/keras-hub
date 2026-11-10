from unittest.mock import patch

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_causal_lm import (
    Qwen2_5VLCausalLM,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_causal_lm_preprocessor import (
    Qwen2_5VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_image_converter import (
    Qwen2_5VLImageConverter,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_tokenizer import (
    Qwen2_5VLTokenizer,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_vision_encoder import (
    Qwen2_5VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2_5VLCausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i"]
        self.merges += ["p l", "n e", "Ġa t", "p o", "r t", "Ġt h"]
        self.merges += ["ai r", "pl a", "po rt", "Ġai r", "Ġa i"]
        self.merges += ["pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        self.vocab += ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        self.vocab += ["<|video_pad|>", "!"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.preprocessor = Qwen2_5VLCausalLMPreprocessor(
            Qwen2_5VLTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=7,
        )
        self.backbone = Qwen2_5VLBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=4,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Qwen2_5VLCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 7, 37),
        )

    def test_generate(self):
        causal_lm = Qwen2_5VLCausalLM(**self.init_kwargs)
        # String input.
        prompt = " airplane at airport"
        output = causal_lm.generate(" airplane at airport")
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        self.assertAllEqual(
            outputs["token_ids"][:, :5],
            prompt_ids["token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5],
            prompt_ids["padding_mask"][:, :5],
        )

    def test_generate_with_image(self):
        """Multimodal generate: image embeddings interleave into prefill."""
        image_converter = Qwen2_5VLImageConverter(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=1,
            min_pixels=16,
            max_pixels=100,
        )
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            Qwen2_5VLTokenizer(vocabulary=self.vocab, merges=self.merges),
            image_converter=image_converter,
            sequence_length=16,
        )
        backbone = Qwen2_5VLBackbone(
            vocabulary_size=preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            # head_dim = 8/4 = 2 → mrope sections must sum to 1.
            mrope_section=[1, 0, 0],
            vision_encoder=Qwen2_5VLVisionEncoder(
                patch_size=2,
                temporal_patch_size=2,
                in_channels=3,
                embed_dim=16,
                out_dim=8,
                depth=2,
                num_heads=2,
                intermediate_dim=256,
                spatial_merge_size=1,
            ),
        )
        causal_lm = Qwen2_5VLCausalLM(
            backbone=backbone, preprocessor=preprocessor
        )
        image = np.ones((4, 4, 3), dtype="float32")
        prompt = "<|vision_start|><|image_pad|><|vision_end|> air"
        output = causal_lm.generate(
            {"prompts": prompt, "images": image},
            max_length=16,
        )
        self.assertTrue(isinstance(output, str))
        self.assertIn(" air", output)

    def test_fit_with_image(self):
        """fit() trains through the multimodal preprocessor path."""
        image_converter = Qwen2_5VLImageConverter(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=1,
            min_pixels=16,
            max_pixels=100,
        )
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            Qwen2_5VLTokenizer(vocabulary=self.vocab, merges=self.merges),
            image_converter=image_converter,
            sequence_length=32,
        )
        backbone = Qwen2_5VLBackbone(
            vocabulary_size=preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            mrope_section=[1, 0, 0],
            vision_encoder=Qwen2_5VLVisionEncoder(
                patch_size=2,
                temporal_patch_size=2,
                in_channels=3,
                embed_dim=16,
                out_dim=8,
                depth=2,
                num_heads=2,
                intermediate_dim=256,
                spatial_merge_size=1,
            ),
        )
        causal_lm = Qwen2_5VLCausalLM(
            backbone=backbone, preprocessor=preprocessor
        )
        images = np.ones((2, 4, 4, 3), dtype="float32")
        x = {
            "prompts": [
                "<|vision_start|><|image_pad|><|vision_end|> air",
            ]
            * 2,
            "responses": [" at airport"] * 2,
            "images": images,
        }
        embedding = causal_lm.backbone.token_embedding.embeddings
        old_weight = ops.convert_to_numpy(embedding)
        causal_lm.fit(x=x, batch_size=2, epochs=1, verbose=0)
        self.assertNotAllClose(old_weight, ops.convert_to_numpy(embedding))

    def test_generate_strip_prompt(self):
        causal_lm = Qwen2_5VLCausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_early_stopping(self):
        causal_lm = Qwen2_5VLCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = [" airplane at airport", " airplane"]
            output = causal_lm.generate(prompt)
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = Qwen2_5VLCausalLM(**self.init_kwargs)
        causal_lm.generate(" airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate(" airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    def test_score(self):
        causal_lm = Qwen2_5VLCausalLM(**self.init_kwargs)
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        logits = causal_lm.score(prompt_ids)
        self.assertEqual(tuple(ops.convert_to_numpy(logits).shape[:2]), (1, 7))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen2_5VLCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen2_5VLCausalLM.presets:
            self.run_preset_test(
                cls=Qwen2_5VLCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
