from unittest.mock import patch

import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm import Qwen3VLCausalLM
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm_preprocessor import (
    Qwen3VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    Qwen3VLImageConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_tokenizer import Qwen3VLTokenizer
from keras_hub.src.models.qwen3_vl.qwen3_vl_video_converter import (
    Qwen3VLVideoConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3VLVisionEncoder,
)
from keras_hub.src.samplers.beam_sampler import BeamSampler
from keras_hub.src.samplers.greedy_sampler import GreedySampler
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu

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


def _atol(cpu):
    # GPU kernels differ from the CPU ones by up to ~1e-3 in float32.
    return 2e-3 if running_on_gpu() else cpu


class Qwen3VLCausalLMTest(TestCase):
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
        self.preprocessor = Qwen3VLCausalLMPreprocessor(
            Qwen3VLTokenizer(vocabulary=self.vocab, merges=self.merges),
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
        self.backbone = Qwen3VLBackbone(**self.text_kwargs)
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def _multimodal_model(self, preprocessor=None, tie_word_embeddings=False):
        vision_encoder = Qwen3VLVisionEncoder(
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
        backbone = Qwen3VLBackbone(
            **self.text_kwargs,
            tie_word_embeddings=tie_word_embeddings,
            vision_encoder=vision_encoder,
        )
        model = Qwen3VLCausalLM(backbone, preprocessor)
        # Default weights make attention nearly uniform and hide errors.
        rng = np.random.RandomState(0)
        model.set_weights([0.3 * rng.randn(*w.shape) for w in model.weights])
        return model

    def _media_preprocessor(self, sequence_length=48):
        converter_kwargs = dict(
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            min_pixels=64,
            max_pixels=1024,
        )
        return Qwen3VLCausalLMPreprocessor(
            self.preprocessor.tokenizer,
            image_converter=Qwen3VLImageConverter(**converter_kwargs),
            video_converter=Qwen3VLVideoConverter(
                fps=2.0, min_frames=2, max_frames=4, **converter_kwargs
            ),
            sequence_length=sequence_length,
        )

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
            cls=Qwen3VLCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(
                2,
                SEQUENCE_LENGTH,
                self.text_kwargs["vocabulary_size"],
            ),
        )

    def test_generate(self):
        causal_lm = Qwen3VLCausalLM(**self.init_kwargs)
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

    def test_pad_token_in_prompt_padding_does_not_stop(self):
        # `<|endoftext|>` is both the pad token and `end_token2`.
        causal_lm = Qwen3VLCausalLM(**self.init_kwargs)
        causal_lm.compile(sampler="greedy")
        end_token2_id = self.preprocessor.tokenizer.end_token2_id
        self.assertEqual(
            end_token2_id, self.preprocessor.tokenizer.pad_token_id
        )
        x = self.preprocessor.generate_preprocess([" airplane at airport"])
        causal_lm.preprocessor = None
        free = causal_lm.generate(x, stop_token_ids=None)
        stopped = causal_lm.generate(x, stop_token_ids=[end_token2_id])
        length = int(np.sum(ops.convert_to_numpy(x["padding_mask"])))
        tokens = ops.convert_to_numpy(free["token_ids"])[0, length:]
        ends = np.flatnonzero(tokens == end_token2_id)
        # Generation runs up to the first generated stop token.
        generated = ends[0] + 1 if len(ends) else len(tokens)
        mask = ops.convert_to_numpy(stopped["padding_mask"])[0, length:]
        self.assertEqual(int(mask.sum()), generated)
        self.assertAllEqual(
            ops.convert_to_numpy(stopped["token_ids"])[0, length:][mask],
            tokens[:generated],
        )

    def test_generate_text_only_on_multimodal_backbone(self):
        causal_lm = self._multimodal_model(self.preprocessor)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)

    def test_generate_with_beam_sampler(self):
        causal_lm = Qwen3VLCausalLM(**self.init_kwargs)
        rng = np.random.RandomState(0)
        causal_lm.set_weights(
            [0.3 * rng.randn(*w.shape) for w in causal_lm.weights]
        )
        causal_lm.compile(sampler=BeamSampler(num_beams=2))
        # Prompts of equal length. `BeamSampler` does not give the same row
        # for prompts of unequal length.
        prompts = [" airplane at airport", " airport at airplane"]
        outputs = causal_lm.generate(prompts)
        self.assertEqual(len(outputs), 2)
        for prompt, output in zip(prompts, outputs):
            self.assertTrue(prompt in output)
            # Batch rows must not depend on each other.
            self.assertEqual(causal_lm.generate(prompt), output)

    def test_generate_single_prompt_with_media(self):
        preprocessor = Qwen3VLCausalLMPreprocessor(
            self.preprocessor.tokenizer,
            image_converter=Qwen3VLImageConverter(
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
        # Bytes prompts and `None` entries are single prompts too, and the
        # caller's dict is left as it was.
        inputs = {"prompts": prompt.encode(), "images": [image], "videos": None}
        self.assertEqual(causal_lm.generate(inputs), single)
        self.assertEqual(inputs["prompts"], prompt.encode())
        self.assertIsNone(inputs["videos"])

    def test_fit_with_image(self):
        preprocessor = Qwen3VLCausalLMPreprocessor(
            self.preprocessor.tokenizer,
            image_converter=Qwen3VLImageConverter(
                patch_size=4,
                temporal_patch_size=2,
                spatial_merge_size=2,
                min_pixels=64,
                max_pixels=1024,
            ),
            sequence_length=24,
        )
        causal_lm = self._multimodal_model(preprocessor)
        image = np.random.RandomState(0).randint(
            0, 255, (16, 16, 3), dtype="uint8"
        )
        x = {
            "prompts": ["<|vision_start|><|image_pad|><|vision_end|> airplane"]
            * 2,
            "images": [[image], [image]],
        }
        embedding = causal_lm.backbone.token_embedding.embeddings
        patch_kernel = causal_lm.backbone.vision_encoder.patch_embed.proj.kernel
        old_embedding = ops.convert_to_numpy(embedding)
        old_patch_kernel = ops.convert_to_numpy(patch_kernel)
        causal_lm.fit(x=x, batch_size=2, epochs=1, verbose=0)
        self.assertNotAllClose(old_embedding, ops.convert_to_numpy(embedding))
        # The loss reaches the vision encoder through the injected features.
        self.assertNotAllClose(
            old_patch_kernel, ops.convert_to_numpy(patch_kernel)
        )

    def test_generate_strip_prompt(self):
        causal_lm = Qwen3VLCausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_early_stopping(self):
        causal_lm = Qwen3VLCausalLM(**self.init_kwargs)
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
        causal_lm = Qwen3VLCausalLM(**self.init_kwargs)
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
            logits[:, 0],
            full_logits[:, PROMPT_LENGTH],
            atol=_atol(1e-4),
            rtol=1e-4,
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
                atol=_atol(1e-5),
                rtol=1e-4,
            )

    def test_score_logits(self):
        causal_lm = Qwen3VLCausalLM(**self.init_kwargs)
        x = self.preprocessor.generate_preprocess(
            [" airplane at airport", " airplane"]
        )
        scores = causal_lm.score(x["token_ids"], x["padding_mask"])
        self.assertEqual(
            ops.shape(scores),
            (2, SEQUENCE_LENGTH, self.preprocessor.tokenizer.vocabulary_size()),
        )
        # Calling the model skips the preprocessor, unlike `predict`. The
        # plain ids get 1D positions, which only differ on padding.
        expected = causal_lm(x)
        mask = ops.convert_to_numpy(x["padding_mask"]).astype(bool)
        self.assertAllClose(
            ops.convert_to_numpy(scores)[mask],
            ops.convert_to_numpy(expected)[mask],
            atol=1e-5,
        )
        self.assertAllClose(causal_lm.score(x), expected, atol=1e-5)
        # A `None` mask in the dict falls back to the `padding_mask` argument.
        x["padding_mask"] = None
        self.assertAllClose(
            causal_lm.score(x, padding_mask=mask), expected, atol=1e-5
        )

    def test_score_multimodal_matches_forward(self):
        causal_lm = self._multimodal_model()
        prompt = self._multimodal_prompt()
        prompt["padding_mask"] = prompt["padding_mask"].astype("int32")
        # Every decoder layer and the embeddings go through the intercept.
        seen = []

        def layer_intercept_fn(x, i):
            seen.append(i)
            return x

        scores = causal_lm.score(prompt, layer_intercept_fn=layer_intercept_fn)
        self.assertEqual(seen, list(range(-1, causal_lm.backbone.num_layers)))
        self.assertAllClose(
            scores, causal_lm.predict_on_batch(prompt), atol=_atol(1e-5)
        )

    def test_score_loss(self):
        causal_lm = self._multimodal_model()
        prompt = self._multimodal_prompt()
        target_ids = np.roll(prompt["token_ids"], -1, axis=1)
        loss = causal_lm.score(
            prompt, scoring_mode="loss", target_ids=target_ids
        )
        self.assertEqual(ops.shape(loss), (1, SEQUENCE_LENGTH))
        logits = causal_lm.score(prompt)
        expected = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )(target_ids, logits)
        self.assertAllClose(loss, expected)
        with self.assertRaisesRegex(ValueError, "target_ids"):
            causal_lm.score(prompt, scoring_mode="loss")

    def test_text_only_inputs_with_vision_encoder(self):
        # The preprocessor has no image or video converter.
        causal_lm = self._multimodal_model(preprocessor=self.preprocessor)
        vocabulary_size = self.text_kwargs["vocabulary_size"]
        x = {
            "token_ids": np.ones((2, 8), dtype="int32"),
            "padding_mask": np.ones((2, 8), dtype="int32"),
        }
        self.assertEqual(causal_lm(x).shape, (2, 8, vocabulary_size))
        output = causal_lm.predict(self.train_data[0], verbose=0)
        self.assertEqual(output.shape, (2, SEQUENCE_LENGTH, vocabulary_size))
        causal_lm.compile(optimizer="sgd")
        causal_lm.fit(self.train_data[0], epochs=1, verbose=0)

    def test_generate_with_video(self):
        causal_lm = self._multimodal_model(self._media_preprocessor())
        causal_lm.compile(sampler="greedy")
        prompt = "<|vision_start|><|video_pad|><|vision_end|> airplane"
        video = np.random.RandomState(0).randint(
            0, 255, (4, 16, 16, 3), dtype="uint8"
        )
        output = causal_lm.generate(
            {"prompts": prompt, "videos": [video], "video_fps": [2.0]}
        )
        self.assertIn(" airplane", output)
        # 4 frames make 2 temporal patches of 2x2 merged tokens each.
        x = causal_lm.preprocessor.generate_preprocess(
            {"prompts": [prompt], "videos": [[video]], "video_fps": [[2.0]]}
        )
        self.assertAllEqual(x["image_grid_thw"][0], [[2, 4, 4]])

    def test_generate_media_with_beam_sampler(self):
        causal_lm = self._multimodal_model(self._media_preprocessor())
        causal_lm.compile(sampler=BeamSampler(num_beams=2))
        prompt = "<|vision_start|><|image_pad|><|vision_end|> airplane"
        rng = np.random.RandomState(0)
        images = [rng.randint(0, 255, (16, 16, 3), dtype="uint8")] * 2
        images[1] = 255 - images[1]
        # Prompts of equal length. `BeamSampler` scores the re-fed prompt
        # slots of the longer prompts in a batch, so only those match the
        # single prompt runs.
        outputs = causal_lm.generate(
            {"prompts": [prompt, prompt], "images": [[images[0]], [images[1]]]}
        )
        self.assertLen(outputs, 2)
        for image, output in zip(images, outputs):
            self.assertIn(" airplane", output)
            # Each beam keeps its own sample's vision features.
            single = causal_lm.generate({"prompts": prompt, "images": [image]})
            self.assertEqual(single, output)

    def test_score_invalid_scoring_mode(self):
        causal_lm = Qwen3VLCausalLM(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "Unsupported scoring_mode"):
            causal_lm.score(self.input_data, scoring_mode="probs")

    def test_get_config_with_vision_encoder(self):
        causal_lm = self._multimodal_model()
        config = causal_lm.backbone.get_config()
        clone = Qwen3VLBackbone.from_config(config)
        self.assertEqual(
            clone.vision_encoder.deepstack_visual_indexes,
            causal_lm.backbone.vision_encoder.deepstack_visual_indexes,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3VLCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_saved_model_multimodal(self):
        backbone = self._multimodal_model(tie_word_embeddings=True).backbone
        self.assertTrue(backbone.tie_word_embeddings)
        self.run_model_saving_test(
            cls=Qwen3VLCausalLM,
            init_kwargs={
                "backbone": backbone,
                "preprocessor": self._media_preprocessor(),
            },
            input_data=self._multimodal_prompt(),
        )

    def test_litert_export(self):
        """Test LiteRT export for the text path of Qwen3VLCausalLM."""
        model = Qwen3VLCausalLM(**self.init_kwargs)
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

    def test_litert_export_multimodal(self):
        """Test LiteRT export with images for Qwen3VLCausalLM."""
        model = self._multimodal_model()
        input_data = self._multimodal_prompt()
        # LiteRT needs an integer padding mask.
        input_data["padding_mask"] = input_data["padding_mask"].astype("int32")
        self.run_litert_export_test(
            model=model,
            input_data=input_data,
            expected_output_shape=(
                1,
                SEQUENCE_LENGTH,
                self.preprocessor.tokenizer.vocabulary_size(),
            ),
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-5}},
        )

    @pytest.mark.large
    def test_from_preset(self):
        causal_lm = CausalLM.from_preset(
            "hf://yujiepan/qwen3-vl-tiny-random", load_weights=False
        )
        self.assertIsInstance(causal_lm, Qwen3VLCausalLM)
        preprocessor = causal_lm.preprocessor
        preprocessor.sequence_length = 96
        vocabulary_size = causal_lm.backbone.vocabulary_size
        # The preset's image converter resizes the image to 256x256, which
        # gives 64 vision tokens.
        image = np.random.RandomState(0).randint(
            0, 255, (32, 32, 3), dtype="uint8"
        )
        for x in [
            {"prompts": ["What is Keras?"]},
            {
                "prompts": [
                    "<|vision_start|><|image_pad|><|vision_end|>Describe it."
                ],
                "images": [[image]],
            },
        ]:
            inputs = preprocessor(x)[0]
            logits = causal_lm(inputs)
            self.assertEqual(logits.shape, (1, 96, vocabulary_size))
