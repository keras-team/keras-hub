from unittest.mock import patch

import keras
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

    def _multimodal_init_kwargs(
        self, sequence_length=16, hidden_dim=8, mrope_section=(1, 0, 0)
    ):
        """Small image-capable preprocessor + backbone for media tests."""
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
            sequence_length=sequence_length,
        )
        backbone = Qwen2_5VLBackbone(
            vocabulary_size=preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=hidden_dim,
            intermediate_dim=16,
            # head_dim = hidden_dim/4; mrope sections must sum to half.
            mrope_section=list(mrope_section),
            vision_encoder=Qwen2_5VLVisionEncoder(
                patch_size=2,
                temporal_patch_size=2,
                in_channels=3,
                embed_dim=16,
                out_dim=hidden_dim,
                depth=2,
                num_heads=2,
                intermediate_dim=256,
                spatial_merge_size=1,
                fullatt_block_indexes=[1],
            ),
        )
        return {"preprocessor": preprocessor, "backbone": backbone}

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

    def _prefill_hidden_states(self, causal_lm, inputs):
        """Prefill hidden states through the same path `generate()`
        uses, including the interleaving first `call_with_cache`."""
        normalized, _ = causal_lm._normalize_generate_inputs(dict(inputs))
        preprocessed = causal_lm.preprocessor.generate_preprocess(normalized[0])
        img_embeddings = causal_lm.backbone.vision_encoder(
            preprocessed["pixel_values"],
            preprocessed["image_grid_thw"],
        )
        hidden_states, _ = causal_lm._build_cache(
            preprocessed["token_ids"],
            preprocessed["padding_mask"],
            img_embeddings=img_embeddings,
            vision_indices=preprocessed["vision_indices"],
            position_ids=preprocessed["position_ids"],
        )
        return ops.convert_to_numpy(hidden_states)

    def test_generate_with_image(self):
        """Multimodal generate: image embeddings interleave into prefill."""
        keras.utils.set_random_seed(0)
        causal_lm = Qwen2_5VLCausalLM(
            **self._multimodal_init_kwargs(sequence_length=16)
        )
        causal_lm.compile(sampler="greedy")
        prompt = "<|vision_start|><|image_pad|><|vision_end|> air"
        hidden_states = []
        for value in (0.0, 255.0):
            image = np.full((4, 4, 3), value, dtype="float32")
            inputs = {"prompts": prompt, "images": image}
            output = causal_lm.generate(
                inputs,
                max_length=20,
                strip_prompt=True,
                stop_token_ids=None,
            )
            self.assertTrue(isinstance(output, str))
            hidden_states.append(self._prefill_hidden_states(causal_lm, inputs))
        # Greedy decodings can coincide for different images; prefill
        # hidden states cannot unless the vision embeddings are dropped.
        self.assertGreater(
            np.max(np.abs(hidden_states[0] - hidden_states[1])), 1e-3
        )

    def test_generate_with_video(self):
        """Video inputs flow through the shared image converter, with
        `second_per_grid_ts` scaling the temporal M-RoPE positions."""
        keras.utils.set_random_seed(0)
        causal_lm = Qwen2_5VLCausalLM(
            **self._multimodal_init_kwargs(sequence_length=32)
        )
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.preprocessor.video_converter)
        prompt = "<|vision_start|><|video_pad|><|vision_end|> air"
        hidden_states = []
        for value in (0.0, 255.0):
            video = np.full((8, 2, 2, 3), value, dtype="float32")
            inputs = {
                "prompts": prompt,
                "videos": [video],
                "second_per_grid_ts": [0.5],
            }
            output = causal_lm.generate(
                inputs,
                max_length=36,
                strip_prompt=True,
                stop_token_ids=None,
            )
            self.assertTrue(isinstance(output, str))
            hidden_states.append(self._prefill_hidden_states(causal_lm, inputs))
        # Same rationale as `test_generate_with_image`.
        self.assertGreater(
            np.max(np.abs(hidden_states[0] - hidden_states[1])), 1e-3
        )

    def test_fit_with_image(self):
        """fit() trains through the multimodal preprocessor path; all
        weights — including the vision tower — stay finite and update."""
        causal_lm = Qwen2_5VLCausalLM(
            **self._multimodal_init_kwargs(sequence_length=32)
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
        vision_kernel = causal_lm.backbone.vision_encoder.merger.mlp_fc1.kernel
        old_embedding = ops.convert_to_numpy(embedding)
        old_vision = ops.convert_to_numpy(vision_kernel)
        causal_lm.fit(x=x, batch_size=2, epochs=2, verbose=0)
        self.assertNotAllClose(old_embedding, ops.convert_to_numpy(embedding))
        self.assertNotAllClose(old_vision, ops.convert_to_numpy(vision_kernel))
        # NaN/inf weights would pass `assertNotAllClose` silently — check
        # finiteness explicitly across the whole model.
        for w in causal_lm.trainable_weights:
            self.assertTrue(
                np.isfinite(ops.convert_to_numpy(w)).all(),
                msg=f"non-finite weight after fit: {w.path}",
            )

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

    def test_score_with_image(self):
        """Multimodal dict inputs score through the interleaved path."""
        causal_lm = Qwen2_5VLCausalLM(
            **self._multimodal_init_kwargs(sequence_length=16)
        )
        image = np.ones((4, 4, 3), dtype="float32")
        prompt = "<|vision_start|><|image_pad|><|vision_end|> air"
        inputs = causal_lm.preprocessor.generate_preprocess(
            {"prompts": [prompt], "images": [image]}
        )
        logits = ops.convert_to_numpy(causal_lm.score(inputs))
        self.assertEqual(logits.shape[0], 1)
        self.assertTrue(np.isfinite(logits).all())

        # Vision embeddings must actually perturb the logits: scoring
        # the same token ids without vision inputs differs.
        text_only = {
            "token_ids": inputs["token_ids"],
            "padding_mask": inputs["padding_mask"],
        }
        logits_text = ops.convert_to_numpy(causal_lm.score(text_only))
        self.assertGreater(np.abs(logits - logits_text).max(), 1e-4)

    def test_score_with_video_second_per_grid_ts(self):
        """`second_per_grid_ts` scales temporal positions enough to move
        video scores."""
        keras.utils.set_random_seed(0)
        causal_lm = Qwen2_5VLCausalLM(
            **self._multimodal_init_kwargs(
                sequence_length=32, hidden_dim=32, mrope_section=(2, 1, 1)
            )
        )
        video = np.random.default_rng(0).random((8, 2, 2, 3)).astype("float32")
        prompt = "<|vision_start|><|video_pad|><|vision_end|> air"

        def score_with_sgt(sgt):
            inputs = causal_lm.preprocessor.generate_preprocess(
                {
                    "prompts": [prompt],
                    "videos": [video],
                    "second_per_grid_ts": [sgt],
                }
            )
            return inputs, ops.convert_to_numpy(causal_lm.score(inputs))

        inputs_slow, logits_slow = score_with_sgt(0.5)
        inputs_fast, logits_fast = score_with_sgt(8.0)
        self.assertTrue(np.isfinite(logits_slow).all())
        # The temporal M-RoPE positions themselves must differ...
        self.assertFalse(
            np.array_equal(
                ops.convert_to_numpy(inputs_slow["position_ids"]),
                ops.convert_to_numpy(inputs_fast["position_ids"]),
            )
        )
        # ...and that difference must propagate into the scores.
        self.assertGreater(np.abs(logits_slow - logits_fast).max(), 1e-4)

    def test_generate_step_positions_include_sgt(self):
        """sgt reaches the backbone: prefill positions differ across sgt
        values in the generate path.

        Spies through a direct `generate_step` call rather than
        `generate()` — the latter is traced under tf.function, where
        eager numpy conversion of captured tensors is impossible.
        """
        causal_lm = Qwen2_5VLCausalLM(
            **self._multimodal_init_kwargs(sequence_length=32)
        )
        video = np.ones((8, 2, 2, 3), dtype="float32")
        prompt = "<|vision_start|><|video_pad|><|vision_end|> air"

        def prefill_positions(sgt):
            inputs = causal_lm.preprocessor.generate_preprocess(
                {
                    "prompts": [prompt],
                    "videos": [video],
                    "second_per_grid_ts": [sgt],
                }
            )
            seen = []
            call_with_cache = causal_lm.call_with_cache

            def spy(*args, **kwargs):
                if not seen and kwargs.get("position_ids") is not None:
                    seen.append(ops.convert_to_numpy(kwargs["position_ids"]))
                return call_with_cache(*args, **kwargs)

            with patch.object(causal_lm, "call_with_cache", wraps=spy):
                causal_lm.generate_step(inputs)
            self.assertTrue(seen, msg="prefill position_ids never captured")
            return seen[0]

        positions_slow = prefill_positions(0.5)
        positions_fast = prefill_positions(2.0)
        self.assertFalse(np.array_equal(positions_slow, positions_fast))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen2_5VLCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_saved_model_multimodal(self):
        """Save/reload a model whose preprocessor emits image inputs."""
        init_kwargs = self._multimodal_init_kwargs(sequence_length=32)
        train_data = (
            {
                "prompts": [
                    "<|vision_start|><|image_pad|><|vision_end|> air",
                ]
                * 2,
                "responses": [" at airport"] * 2,
                "images": np.ones((2, 4, 4, 3), dtype="float32"),
            },
        )
        input_data = init_kwargs["preprocessor"](*train_data)[0]
        self.run_model_saving_test(
            cls=Qwen2_5VLCausalLM,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )
