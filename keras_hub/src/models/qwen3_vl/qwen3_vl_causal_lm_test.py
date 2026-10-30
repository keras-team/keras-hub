from unittest.mock import patch

import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm import Qwen3_VLCausalLM
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm_preprocessor import (
    Qwen3_VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    Qwen3_VLImageConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_tokenizer import Qwen3_VLTokenizer
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3_VLVisionEncoder,
)
from keras_hub.src.samplers.beam_sampler import BeamSampler
from keras_hub.src.samplers.greedy_sampler import GreedySampler
from keras_hub.src.tests.test_case import TestCase

SEQUENCE_LENGTH = 12
# Two text tokens, a 2x2 block of vision tokens, then two text tokens.
PROMPT_LENGTH = 8
VISION_POSITIONS = [2, 3, 4, 5]
# Temporal, height and width positions of the prompt above. The image
# advances the position by 2 instead of 4, so HF's rope_deltas is -2.
PROMPT_POSITIONS = [
    [0, 1, 2, 2, 2, 2, 4, 5],
    [0, 1, 2, 2, 3, 3, 4, 5],
    [0, 1, 2, 3, 2, 3, 4, 5],
]


class Qwen3_VLCausalLMTest(TestCase):
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
        self.vocab += ["<|video_pad|>", "<think>", "!"]
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.preprocessor = Qwen3_VLCausalLMPreprocessor(
            Qwen3_VLTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=SEQUENCE_LENGTH,
        )
        self.text_kwargs = {
            "vocabulary_size": self.preprocessor.tokenizer.vocabulary_size(),
            "num_layers": 3,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "mrope_section": [2, 1, 1],
        }
        self.backbone = Qwen3_VLBackbone(**self.text_kwargs)
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def _multimodal_model(self, preprocessor=None, tie_word_embeddings=False):
        vision_encoder = Qwen3_VLVisionEncoder(
            depth=3,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            out_hidden_size=16,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=64,
            deepstack_visual_indexes=(0, 1),
        )
        backbone = Qwen3_VLBackbone(
            **self.text_kwargs,
            tie_word_embeddings=tie_word_embeddings,
            vision_encoder=vision_encoder,
        )
        model = Qwen3_VLCausalLM(backbone, preprocessor)
        # Default weights make attention nearly uniform and hide errors.
        rng = np.random.RandomState(0)
        model.set_weights([0.3 * rng.randn(*w.shape) for w in model.weights])
        return model

    def _multimodal_prompt(self):
        rng = np.random.RandomState(1)
        token_ids = np.zeros((1, SEQUENCE_LENGTH), dtype="int32")
        token_ids[:, :PROMPT_LENGTH] = rng.randint(1, 20, size=PROMPT_LENGTH)
        padding_mask = np.zeros((1, SEQUENCE_LENGTH), dtype="bool")
        padding_mask[:, :PROMPT_LENGTH] = True
        position_ids = np.zeros((1, SEQUENCE_LENGTH, 3), "int32")
        position_ids[:, :PROMPT_LENGTH] = np.array(PROMPT_POSITIONS).T
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "position_ids": position_ids,
            "pixel_values": rng.randn(1, 16, 2, 4, 4, 3).astype("float32"),
            "image_grid_thw": np.array([[[1, 4, 4]]], dtype="int32"),
            "vision_indices": np.array([VISION_POSITIONS], dtype="int32"),
        }

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Qwen3_VLCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(
                2,
                SEQUENCE_LENGTH,
                self.text_kwargs["vocabulary_size"],
            ),
        )

    def test_generate(self):
        causal_lm = Qwen3_VLCausalLM(**self.init_kwargs)
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

    def test_generate_text_only_on_multimodal_backbone(self):
        causal_lm = self._multimodal_model(self.preprocessor)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)

    def test_generate_with_beam_sampler(self):
        causal_lm = Qwen3_VLCausalLM(**self.init_kwargs)
        causal_lm.compile(sampler=BeamSampler(num_beams=2))
        prompts = [" airplane at airport", " airplane"]
        outputs = causal_lm.generate(prompts)
        self.assertEqual(len(outputs), 2)
        for prompt, output in zip(prompts, outputs):
            self.assertTrue(prompt in output)
        # Batch rows must not depend on each other.
        self.assertEqual(causal_lm.generate(prompts[1]), outputs[1])

    def test_generate_single_prompt_with_media(self):
        preprocessor = Qwen3_VLCausalLMPreprocessor(
            self.preprocessor.tokenizer,
            image_converter=Qwen3_VLImageConverter(
                patch_size=4,
                temporal_patch_size=2,
                spatial_merge_size=2,
                min_pixels=64,
                max_pixels=1024,
            ),
            sequence_length=24,
        )
        causal_lm = self._multimodal_model(preprocessor)
        causal_lm.compile(sampler="greedy")
        prompt = "<|vision_start|><|image_pad|><|vision_end|> airplane"
        image = np.random.RandomState(0).randint(
            0, 255, (16, 16, 3), dtype="uint8"
        )
        single = causal_lm.generate({"prompts": prompt, "images": [image]})
        reordered = causal_lm.generate({"images": [image], "prompts": prompt})
        batched = causal_lm.generate({"prompts": [prompt], "images": [[image]]})
        self.assertIsInstance(single, str)
        self.assertIsInstance(reordered, str)
        self.assertEqual(single, reordered)
        self.assertEqual([single], batched)

    def test_generate_strip_prompt(self):
        causal_lm = Qwen3_VLCausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_early_stopping(self):
        causal_lm = Qwen3_VLCausalLM(**self.init_kwargs)
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
        causal_lm = Qwen3_VLCausalLM(**self.init_kwargs)
        causal_lm.generate(" airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate(" airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    def test_cache_matches_full_forward_multimodal(self):
        causal_lm = self._multimodal_model()
        prompt = self._multimodal_prompt()
        features, *deepstack = causal_lm.backbone.vision_encoder(
            {
                "pixel_values": prompt["pixel_values"],
                "image_grid_thw": prompt["image_grid_thw"],
            }
        )
        # Prefill the prompt, then decode one token at HF's text position.
        new_token = np.array([[7]], dtype="int32")
        token_ids = prompt["token_ids"].copy()
        token_ids[0, PROMPT_LENGTH] = 7
        position_ids = ops.transpose(prompt["position_ids"], (0, 2, 1))
        embeddings, updates = causal_lm._embed_prompt(
            token_ids, features, deepstack, prompt["vision_indices"]
        )
        _, cache = causal_lm._build_cache(
            token_ids,
            prompt["padding_mask"],
            position_ids,
            embeddings=embeddings,
            deepstack_updates=updates,
        )
        delta = PROMPT_POSITIONS[0][-1] + 1 - PROMPT_LENGTH
        step_positions = np.full((1, 3, 1), PROMPT_LENGTH + delta, "int32")
        logits, _, _ = causal_lm.call_with_cache(
            new_token, cache, PROMPT_LENGTH, position_ids=step_positions
        )

        # The same extended sequence through the full backbone path.
        full = dict(prompt)
        full["padding_mask"] = prompt["padding_mask"].copy()
        full["padding_mask"][0, PROMPT_LENGTH] = True
        full["token_ids"] = token_ids
        full["position_ids"] = prompt["position_ids"].copy()
        full["position_ids"][0, PROMPT_LENGTH] = PROMPT_LENGTH + delta
        full_logits = causal_lm.predict_on_batch(
            {
                k: v.astype("int32") if k == "padding_mask" else v
                for k, v in full.items()
            }
        )
        self.assertAllClose(
            logits[:, 0], full_logits[:, PROMPT_LENGTH], atol=1e-4, rtol=1e-4
        )

    def test_generate_multimodal_matches_full_forward(self):
        causal_lm = self._multimodal_model()
        causal_lm.compile(sampler="greedy")
        prompt = self._multimodal_prompt()
        outputs = causal_lm.generate(prompt, stop_token_ids=None)
        generated = ops.convert_to_numpy(outputs["token_ids"])

        # Greedy decoding without a cache, one token at a time.
        token_ids = prompt["token_ids"].copy()
        position_ids = prompt["position_ids"].copy()
        padding_mask = prompt["padding_mask"].astype("int32")
        delta = PROMPT_POSITIONS[0][-1] + 1 - PROMPT_LENGTH
        for i in range(PROMPT_LENGTH, SEQUENCE_LENGTH):
            padding_mask[0, i] = 1
            position_ids[0, i] = i + delta
            inputs = dict(prompt)
            inputs.update(
                token_ids=token_ids,
                padding_mask=padding_mask,
                position_ids=position_ids,
            )
            logits = causal_lm.predict_on_batch(inputs)
            token_ids[0, i] = int(np.argmax(logits[0, i - 1]))
        self.assertAllEqual(generated, token_ids)

    def test_unequal_prompt_lengths_keep_vision_features(self):
        class RecordingSampler(GreedySampler):
            def __init__(self):
                super().__init__()
                self.probabilities = []

            def get_next_token(self, probabilities):
                self.probabilities.append(ops.convert_to_numpy(probabilities))
                return super().get_next_token(probabilities)

        # JAX traces the sampler loop even when `run_eagerly=True`, so a
        # sampler cannot record concrete probabilities there.
        record = keras.config.backend() != "jax"
        causal_lm = self._multimodal_model()
        image_prompt = self._multimodal_prompt()
        sampler = RecordingSampler() if record else "greedy"
        causal_lm.compile(sampler=sampler, run_eagerly=True)
        alone = causal_lm.generate(image_prompt, stop_token_ids=None)
        if record:
            alone_probabilities = sampler.probabilities[0][0]

        # A short text prompt makes the sampler re-feed the image slots of the
        # longer row, which must keep their vision features.
        short_length = 4
        padding_mask = np.zeros((2, SEQUENCE_LENGTH), dtype="bool")
        padding_mask[0] = image_prompt["padding_mask"][0]
        padding_mask[1, :short_length] = True
        token_ids = np.zeros((2, SEQUENCE_LENGTH), dtype="int32")
        token_ids[0] = image_prompt["token_ids"][0]
        token_ids[1, :short_length] = [3, 5, 7, 9]
        position_ids = np.zeros((2, SEQUENCE_LENGTH, 3), dtype="int32")
        position_ids[0] = image_prompt["position_ids"][0]
        position_ids[1, :short_length] = np.arange(short_length)[:, None]
        batch = dict(
            image_prompt,
            token_ids=token_ids,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )
        if record:
            sampler.probabilities.clear()
        batched = causal_lm.generate(batch, stop_token_ids=None)
        self.assertAllEqual(
            ops.convert_to_numpy(batched["token_ids"])[0],
            ops.convert_to_numpy(alone["token_ids"])[0],
        )
        if record:
            # The first token the long row generates is predicted at the step
            # where the sampler index reaches its prompt length.
            step = PROMPT_LENGTH - short_length
            self.assertAllClose(
                sampler.probabilities[step][0],
                alone_probabilities,
                atol=1e-5,
                rtol=1e-4,
            )

    def test_get_config_with_vision_encoder(self):
        causal_lm = self._multimodal_model()
        config = causal_lm.backbone.get_config()
        clone = Qwen3_VLBackbone.from_config(config)
        self.assertEqual(
            clone.vision_encoder.deepstack_visual_indexes,
            causal_lm.backbone.vision_encoder.deepstack_visual_indexes,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3_VLCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        """Test LiteRT export for the text path of Qwen3_VLCausalLM."""
        model = Qwen3_VLCausalLM(**self.init_kwargs)
        input_data = dict(self.input_data)
        # LiteRT needs an integer padding mask.
        input_data["padding_mask"] = ops.cast(
            input_data["padding_mask"], "int32"
        )
        self.run_litert_export_test(
            model=model,
            input_data=input_data,
            expected_output_shape=(
                2,
                SEQUENCE_LENGTH,
                self.preprocessor.tokenizer.vocabulary_size(),
            ),
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-5}},
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3_VLCausalLM.presets:
            self.run_preset_test(
                cls=Qwen3_VLCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
