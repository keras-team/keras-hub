import gc
import weakref
from unittest.mock import patch

import keras
import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_assistant_causal_lm import (  # noqa: E501
    MuseGlimmerAssistantCausalLM,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_causal_lm import (
    MuseGlimmerCausalLM,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_causal_lm_preprocessor import (  # noqa: E501
    MuseGlimmerCausalLMPreprocessor,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    MuseGlimmerImageConverter,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_tokenizer import (
    MuseGlimmerTokenizer,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionEncoder,
)
from keras_hub.src.samplers.top_k_sampler import TopKSampler
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerCausalLMTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|end_of_text|>", "<|begin_of_text|>"]
        self.vocab += ["<|finetune_right_pad|>", "<|eot|>"]
        self.vocab += ["<|image_start|>", "<|image_end|>", "<|vid_start|>"]
        self.vocab += ["<|vid_end|>", "<|vid_frame_separator|>"]
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.preprocessor = MuseGlimmerCausalLMPreprocessor(
            MuseGlimmerTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=7,
        )
        self.backbone = MuseGlimmerBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=4,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            sliding_window_size=4,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.causal_lm = MuseGlimmerCausalLM(**self.init_kwargs)
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

        # === Vision + Text model ===
        self.image_converter = MuseGlimmerImageConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            max_image_tokens=64,
            scale=1 / 255.0,
        )
        # Map `image_token_id` to a vocabulary token, so prompts can
        # contain image placeholders.
        image_vocab = {**self.vocab, "<|image|>": len(self.vocab)}
        self.multimodal_preprocessor = MuseGlimmerCausalLMPreprocessor(
            MuseGlimmerTokenizer(
                vocabulary=image_vocab,
                merges=self.merges,
                image_token_id=image_vocab["<|image|>"],
                unsplittable_tokens=["<|image|>"],
            ),
            image_converter=self.image_converter,
            sequence_length=10,
        )
        vision_encoder = MuseGlimmerVisionEncoder(
            num_layers=2,
            hidden_size=8,
            num_heads=2,
            intermediate_size=16,
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            pos_emb_height=4,
            pos_emb_width=4,
            layer_types=["window_attention", "full_attention"],
        )
        backbone = MuseGlimmerBackbone(
            vocabulary_size=len(image_vocab),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=32,
            intermediate_dim=32,
            head_dim=8,
            sliding_window_size=4,
            layer_types=["sliding_attention", "full_attention"],
            vision_encoder=vision_encoder,
            projector_hidden_dim=16,
        )
        self.multimodal_init_kwargs = {
            "preprocessor": self.multimodal_preprocessor,
            "backbone": backbone,
        }
        image = np.ones((8, 8, 3), dtype="float32") * 127.0
        self.multimodal_train_data = (
            {
                "prompts": ["<|image|> airplane at", " airplane<|image|>"],
                "images": np.stack([image, image]),
            },
        )
        self.multimodal_input_data = self.multimodal_preprocessor(
            *self.multimodal_train_data
        )[0]
        self.multimodal_causal_lm = MuseGlimmerCausalLM(
            backbone=backbone, preprocessor=None
        )

    def _make_assistant(self, block_size):
        # A tiny DFlash assistant. It reuses the target hidden_dim, because
        # noise_embeds come from the target token_embedding.
        # target_layer_ids=[1, 2] is valid for the target num_layers=4.
        assistant_backbone = MuseGlimmerBackbone(
            vocabulary_size=1,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            sliding_window_size=None,
            layer_types=["full_attention", "full_attention"],
            use_bidirectional_attention=True,
            context_projection_layer_ids=[1, 2],
            use_external_embeddings=True,
            enable_qk_scale_and_gate=False,
            use_sandwich_norm=False,
        )
        return MuseGlimmerAssistantCausalLM(
            backbone=assistant_backbone,
            block_size=block_size,
            mask_token_id=0,
        )

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_causal_lm_basics(self, model_type):
        if model_type == "text_and_vision":
            init_kwargs = self.multimodal_init_kwargs
            train_data = self.multimodal_train_data
            sequence_length = 10
        else:
            init_kwargs = self.init_kwargs
            train_data = self.train_data
            sequence_length = 7
        vocabulary_size = init_kwargs[
            "preprocessor"
        ].tokenizer.vocabulary_size()
        self.run_task_test(
            cls=MuseGlimmerCausalLM,
            init_kwargs=init_kwargs,
            train_data=train_data,
            expected_output_shape=(2, sequence_length, vocabulary_size),
        )

    def test_generate(self):
        prompt = " airplane at airport"
        output = self.causal_lm.generate(" airplane at airport")
        self.assertTrue(prompt in output)
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        self.causal_lm.preprocessor = None
        outputs = self.causal_lm.generate(prompt_ids, stop_token_ids=None)
        self.assertAllEqual(
            outputs["token_ids"][:, :5], prompt_ids["token_ids"][:, :5]
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5], prompt_ids["padding_mask"][:, :5]
        )

    def test_generate_with_image_under_compiled_backend(self):
        tokenizer = self.preprocessor.tokenizer

        image = np.random.randint(0, 255, (8, 8, 3)).astype("float32")
        result = self.image_converter(image)
        t, h, w = (int(v) for v in ops.convert_to_numpy(result["grid_thw"]))
        num_image_tokens = (
            t
            * (h // self.image_converter.merge_size)
            * (w // self.image_converter.merge_size)
        )

        # image_token_id (200092) exceeds this tiny vocab, so mark image
        # positions with a valid id (0); vision_indices is what matters.
        sequence_length = 16
        prefix_ids = ops.convert_to_numpy(tokenizer("airplane"))
        image_positions = np.arange(num_image_tokens) + len(prefix_ids)
        token_ids = np.concatenate(
            [prefix_ids, np.zeros((num_image_tokens,), dtype="int32")]
        )[:sequence_length]
        padding_mask = np.ones((sequence_length,), dtype="bool")
        if len(token_ids) < sequence_length:
            pad = sequence_length - len(token_ids)
            token_ids = np.concatenate(
                [token_ids, np.zeros((pad,), dtype="int32")]
            )
            padding_mask[sequence_length - pad :] = False
        vision_indices = image_positions[
            image_positions < sequence_length
        ].astype("int32")

        inputs = {
            "token_ids": ops.convert_to_tensor(token_ids[None, :]),
            "padding_mask": ops.convert_to_tensor(padding_mask[None, :]),
            "pixel_values": ops.expand_dims(result["patches"], axis=0),
            "image_grid_thw": ops.reshape(result["grid_thw"], (1, 1, 3)),
            "vision_indices": ops.convert_to_tensor(vision_indices[None, :]),
        }

        output = self.multimodal_causal_lm.generate(
            inputs, max_length=sequence_length, stop_token_ids=None
        )
        self.assertEqual(ops.shape(output["token_ids"]), (1, sequence_length))

    def _multimodal_generate_inputs(
        self, prompts, sequence_length, image_seeds=None
    ):
        # A 16x16 image gives 16 patches and 4 merged vision tokens. Each
        # seed gives the same image in every call.
        if image_seeds is None:
            image_seeds = range(len(prompts))
        images = [
            np.random.default_rng(seed)
            .uniform(0, 255, (16, 16, 3))
            .astype("float32")
            for seed in image_seeds
        ]
        return self.multimodal_preprocessor.generate_preprocess(
            {"prompts": prompts, "images": images},
            sequence_length=sequence_length,
        )

    def _first_image_position(self, row_token_ids):
        tokenizer = self.multimodal_preprocessor.tokenizer
        is_image = np.asarray(row_token_ids) == tokenizer.image_token_id
        return int(np.flatnonzero(is_image)[0])

    def test_generate_ragged_batch_with_images_matches_single_rows(self):
        self.multimodal_causal_lm.compile(sampler="greedy")
        prompts = ["<|image|> air", " airplane at airport at at<|image|> at"]
        sequence_length = 16
        batch_inputs = self._multimodal_generate_inputs(
            prompts, sequence_length
        )
        row_lengths = np.sum(np.asarray(batch_inputs["padding_mask"]), axis=-1)
        # The image of the longer row must start at or after the position
        # where generation starts. This is the case the old code broke.
        self.assertGreaterEqual(
            self._first_image_position(batch_inputs["token_ids"][1]),
            int(row_lengths.min()) - 1,
        )
        batch_output = self.multimodal_causal_lm.generate(
            batch_inputs, max_length=sequence_length, stop_token_ids=None
        )
        for row, prompt in enumerate(prompts):
            row_inputs = self._multimodal_generate_inputs(
                [prompt], sequence_length, image_seeds=[row]
            )
            row_output = self.multimodal_causal_lm.generate(
                row_inputs, max_length=sequence_length, stop_token_ids=None
            )
            self.assertAllEqual(
                batch_output["token_ids"][row], row_output["token_ids"][0]
            )

    def test_generate_prompt_ending_in_image_tokens_uses_image_features(self):
        self.multimodal_causal_lm.compile(sampler="greedy")
        sequence_length = 12
        inputs = self._multimodal_generate_inputs(
            [" airplane<|image|>"], sequence_length
        )
        # The layout ends with the image end token. Cut the prompt after the
        # last image token, so that the first generated token follows an
        # image token.
        image_token_id = self.multimodal_preprocessor.tokenizer.image_token_id
        token_ids = np.asarray(inputs["token_ids"])
        prompt_length = (
            int(np.flatnonzero(token_ids[0] == image_token_id)[-1]) + 1
        )
        padding_mask = np.zeros_like(np.asarray(inputs["padding_mask"]))
        padding_mask[:, :prompt_length] = 1
        inputs = {**inputs, "padding_mask": padding_mask}
        output = self.multimodal_causal_lm.generate(
            inputs, max_length=sequence_length, stop_token_ids=None
        )
        # The full forward pass sees the image features at every prompt
        # position. Each generated token must follow its last logit.
        generated = ops.convert_to_numpy(output["token_ids"])
        logits = self.multimodal_causal_lm(
            {
                "token_ids": ops.convert_to_tensor(generated),
                "padding_mask": ops.ones((1, sequence_length), dtype="int32"),
                "pixel_values": ops.convert_to_tensor(inputs["pixel_values"])[
                    None
                ],
                "image_grid_thw": ops.convert_to_tensor(
                    inputs["image_grid_thw"]
                )[None],
                "vision_indices": ops.convert_to_tensor(
                    inputs["vision_indices"]
                )[None],
            }
        )
        logits = ops.convert_to_numpy(logits)[0]
        for position in range(prompt_length, sequence_length):
            self.assertEqual(
                int(generated[0][position]),
                int(np.argmax(logits[position - 1])),
            )

    def _make_multimodal_assistant(self, block_size):
        # The assistant uses the hidden size of the multimodal target.
        assistant_backbone = MuseGlimmerBackbone(
            vocabulary_size=1,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=32,
            intermediate_dim=32,
            head_dim=8,
            sliding_window_size=None,
            layer_types=["full_attention", "full_attention"],
            use_bidirectional_attention=True,
            context_projection_layer_ids=[0, 1],
            use_external_embeddings=True,
            enable_qk_scale_and_gate=False,
            use_sandwich_norm=False,
        )
        return MuseGlimmerAssistantCausalLM(
            backbone=assistant_backbone,
            block_size=block_size,
            mask_token_id=0,
        )

    def test_assisted_greedy_with_image_matches_plain_greedy(self):
        self.multimodal_causal_lm.compile(sampler="greedy")
        block_size, max_length = 5, 11
        inputs = self._multimodal_generate_inputs(
            [" air<|image|> at"], max_length
        )
        # The prompt must leave room to generate tokens.
        prompt_length = int(np.sum(np.asarray(inputs["padding_mask"])))
        self.assertLess(prompt_length, max_length)
        # The verify window starts at `max_length - block_size` at the
        # latest. It must reach the image for the old code to fail.
        last_image_position = int(
            np.flatnonzero(
                np.asarray(inputs["token_ids"])
                == self.multimodal_preprocessor.tokenizer.image_token_id
            )[-1]
        )
        self.assertLessEqual(max_length - block_size, last_image_position)
        assistant = self._make_multimodal_assistant(block_size)
        plain = self.multimodal_causal_lm.generate(
            inputs, max_length=max_length, stop_token_ids=None
        )
        assisted = self.multimodal_causal_lm.generate(
            inputs,
            max_length=max_length,
            stop_token_ids=None,
            assistant_model=assistant,
        )
        self.assertAllEqual(assisted["token_ids"], plain["token_ids"])

    def test_early_stopping(self):
        call_with_cache = self.causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(self.causal_lm, "call_with_cache", wraps=wrapper):
            prompt = [" airplane at airport", " airplane"]
            output = self.causal_lm.generate(prompt)
            self.assertEqual(prompt, output)

    def test_generate_with_assistant(self):
        assistant = self._make_assistant(block_size=3)
        # Greedy on both sides: speculative decoding's accept/reject step
        # is only guaranteed to reproduce the target model's own output
        # exactly under matching (here, greedy) acceptance semantics.
        self.causal_lm.compile(sampler="greedy")
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None
        reference_output = self.causal_lm.generate(
            prompt_ids, stop_token_ids=None
        )

        output = self.causal_lm.generate(
            prompt_ids,
            stop_token_ids=None,
            assistant_model=assistant,
        )
        # The core speculative-decoding guarantee: regardless of the
        # drafter's own quality, verified/accepted output must exactly
        # match plain autoregressive decoding of the target model.
        self.assertAllEqual(output["token_ids"], reference_output["token_ids"])
        self.assertAllEqual(
            output["padding_mask"], reference_output["padding_mask"]
        )
        # A greedy target keeps greedy acceptance.
        speculative_sampler = self.causal_lm._assisted_generate_cache[2]
        self.assertIsNone(speculative_sampler.base_sampler)

    def test_generate_with_assistant_ragged_batch_keeps_prompt(self):
        assistant = self._make_assistant(block_size=3)
        self.causal_lm.compile(sampler="greedy")
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport", " airplane"]
        )
        self.causal_lm.preprocessor = None
        reference_output = self.causal_lm.generate(
            prompt_ids, stop_token_ids=None
        )
        output = self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        self.assertAllEqual(output["token_ids"], reference_output["token_ids"])
        # The prompt tokens of each row stay unchanged.
        prompt_mask = ops.convert_to_numpy(prompt_ids["padding_mask"]) > 0
        self.assertAllEqual(
            np.where(prompt_mask, ops.convert_to_numpy(output["token_ids"]), 0),
            np.where(
                prompt_mask, ops.convert_to_numpy(prompt_ids["token_ids"]), 0
            ),
        )

    def test_generate_with_assistant_rejects_short_max_length(self):
        assistant = self._make_assistant(block_size=6)
        self.causal_lm.compile(sampler="greedy")
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"], sequence_length=4
        )
        self.causal_lm.preprocessor = None
        sampler = self.causal_lm.sampler
        generate_function = self.causal_lm.generate_function
        num_weights = len(self.causal_lm.weights)
        with self.assertRaisesRegex(ValueError, "at least 6"):
            self.causal_lm.generate(
                prompt_ids, stop_token_ids=None, assistant_model=assistant
            )
        # The failed call must restore the plain sampler.
        self.assertIs(self.causal_lm.sampler, sampler)
        self.assertIs(self.causal_lm.generate_function, generate_function)
        # The failed call must not leave the assistant attached to the model.
        self.assertEqual(len(self.causal_lm.weights), num_weights)
        self.assertFalse(hasattr(self.causal_lm, "_assistant_model"))

    def test_generate_with_assistant_on_quantized_target(self):
        assistant = self._make_assistant(block_size=3)
        self.causal_lm.compile(sampler="greedy")
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"], sequence_length=16
        )
        self.causal_lm.preprocessor = None
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        self.causal_lm.quantize("int8")
        # The compiled assisted function holds the old target variables.
        self.assertIsNone(self.causal_lm._assisted_generate_cache)
        reference_output = self.causal_lm.generate(
            prompt_ids, stop_token_ids=None
        )
        output = self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        self.assertAllEqual(output["token_ids"], reference_output["token_ids"])

    def test_generate_with_assistant_rejects_beam_and_contrastive(self):
        assistant = self._make_assistant(block_size=3)
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None
        for sampler in ("beam", "contrastive"):
            self.causal_lm.compile(sampler=sampler)
            original_sampler = self.causal_lm.sampler
            with self.assertRaisesRegex(ValueError, "top-p samplers"):
                self.causal_lm.generate(
                    prompt_ids, stop_token_ids=None, assistant_model=assistant
                )
            self.assertIs(self.causal_lm.sampler, original_sampler)

    def test_get_quantization_layer_structure(self):
        structure = self.causal_lm.get_quantization_layer_structure("gptq")
        self.assertLen(structure["pre_block_layers"], 1)
        self.assertEqual(
            structure["sequential_blocks"],
            self.causal_lm.backbone.transformer_layers,
        )
        # The first block reads the normed embedding.
        backbone = self.causal_lm.backbone
        token_ids = self.input_data["token_ids"]
        expected = backbone.embed_norm(backbone.token_embedding(token_ids))
        self.assertAllClose(
            structure["pre_block_layers"][0](token_ids), expected
        )
        self.assertIsNone(
            self.causal_lm.get_quantization_layer_structure("int8")
        )

    def test_generate_with_invalid_assistant_raises(self):
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None

        def generate(assistant):
            self.causal_lm.generate(
                prompt_ids, stop_token_ids=None, assistant_model=assistant
            )

        with self.assertRaisesRegex(ValueError, "block_size"):
            generate(self._make_assistant(block_size=1))
        no_mask_token = self._make_assistant(block_size=3)
        no_mask_token.mask_token_id = None
        with self.assertRaisesRegex(ValueError, "mask_token_id"):
            generate(no_mask_token)
        vocabulary_size = self.backbone.vocabulary_size
        out_of_range = self._make_assistant(block_size=3)
        out_of_range.mask_token_id = vocabulary_size
        with self.assertRaisesRegex(ValueError, "mask_token_id"):
            generate(out_of_range)
        bad_layer_ids = self._make_assistant(block_size=3)
        bad_layer_ids.backbone.context_projection_layer_ids = [1, 7]
        with self.assertRaisesRegex(ValueError, "context_projection_layer_ids"):
            generate(bad_layer_ids)
        wrong_hidden_dim = self._make_assistant(block_size=3)
        wrong_hidden_dim.backbone.hidden_dim = 16
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            generate(wrong_hidden_dim)

    def test_target_layer_ids_match_layer_outputs(self):
        # Id `j` is the output of decoder layer `j`. The last layer gives
        # the final-norm output, like HF `output_hidden_states=[ids]`.
        backbone = self.causal_lm.backbone
        token_ids = self.input_data["token_ids"]
        padding_mask = self.input_data["padding_mask"]
        last_layer = backbone.num_layers - 1
        for layer_id in range(backbone.num_layers):
            if layer_id == last_layer:
                output = backbone.output
            else:
                output = backbone.get_layer(f"transformer_layer_{layer_id}")
                output = output.output
            expected = keras.Model(backbone.input, output)(self.input_data)
            _, _, target_hidden_states = self.causal_lm._build_cache(
                token_ids, padding_mask, target_layer_ids=[layer_id]
            )
            self.assertAllClose(
                target_hidden_states, expected, atol=1e-5, rtol=1e-5
            )

    def test_generate_with_beam_and_contrastive_samplers(self):
        for sampler in ("beam", "contrastive"):
            self.causal_lm.compile(sampler=sampler)
            output = self.causal_lm.generate(" airplane at airport")
            self.assertTrue(" airplane at airport" in output)

    def test_generate_with_beam_and_contrastive_samplers_with_image(self):
        inputs = self._multimodal_generate_inputs([" airplane<|image|>"], 12)
        for sampler in ("beam", "contrastive"):
            self.multimodal_causal_lm.compile(sampler=sampler)
            output = self.multimodal_causal_lm.generate(
                inputs, max_length=12, stop_token_ids=None
            )
            self.assertEqual(ops.shape(output["token_ids"]), (1, 12))

    def test_text_only_predict_evaluate_fit_on_vision_model(self):
        model = MuseGlimmerCausalLM(**self.multimodal_init_kwargs)
        prompts = [" airplane at airport"] * 2
        outputs = model.predict(prompts, batch_size=2, verbose=0)
        self.assertEqual(outputs.shape[0], 2)
        model.evaluate(x=prompts, batch_size=2, verbose=0)
        model.fit(x=prompts, batch_size=2, epochs=1, verbose=0)

    def test_generate_with_assistant_non_greedy(self):
        assistant = self._make_assistant(block_size=3)
        self.causal_lm.compile(sampler="top_p")
        target_sampler = self.causal_lm.sampler
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None
        output = self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        # The target sampler drives stochastic acceptance.
        speculative_sampler = self.causal_lm._assisted_generate_cache[2]
        self.assertIs(speculative_sampler.base_sampler, target_sampler)
        self.assertIs(self.causal_lm.sampler, target_sampler)
        num_prompt_tokens = int(ops.sum(prompt_ids["padding_mask"]))
        self.assertAllEqual(
            output["token_ids"][:, :num_prompt_tokens],
            prompt_ids["token_ids"][:, :num_prompt_tokens],
        )
        self.assertEqual(
            tuple(ops.shape(output["token_ids"])),
            tuple(ops.shape(prompt_ids["token_ids"])),
        )

    def test_generate_with_assistant_top_k_one_matches_plain(self):
        assistant = self._make_assistant(block_size=3)
        self.causal_lm.compile(sampler=TopKSampler(k=1))
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport", " airplane"]
        )
        self.causal_lm.preprocessor = None
        reference_output = self.causal_lm.generate(
            prompt_ids, stop_token_ids=None
        )
        # Each assisted call must give the plain output.
        for _ in range(3):
            output = self.causal_lm.generate(
                prompt_ids, stop_token_ids=None, assistant_model=assistant
            )
            self.assertAllEqual(
                output["token_ids"], reference_output["token_ids"]
            )

    def test_generate_with_assistant_reuses_compiled_function(self):
        assistant = self._make_assistant(block_size=3)
        self.causal_lm.compile(sampler="greedy")
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None
        self.causal_lm.generate(prompt_ids, stop_token_ids=None)
        plain_function = self.causal_lm.generate_function

        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        assisted_function = self.causal_lm._assisted_generate_cache[3]
        # Count drafter calls. A compiled backend reuses its traced graph
        # and does not call the drafter from Python again.
        with patch.object(
            assistant, "call_with_cache", wraps=assistant.call_with_cache
        ) as draft_mock:
            self.causal_lm.generate(
                prompt_ids, stop_token_ids=None, assistant_model=assistant
            )
        self.assertIs(
            self.causal_lm._assisted_generate_cache[3], assisted_function
        )
        if keras.config.backend() in ("jax", "tensorflow"):
            self.assertEqual(draft_mock.call_count, 0)
        self.assertIs(self.causal_lm.generate_function, plain_function)

        # A different assistant compiles a new function.
        other_assistant = self._make_assistant(block_size=3)
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=other_assistant
        )
        self.assertIsNot(
            self.causal_lm._assisted_generate_cache[3], assisted_function
        )

    def test_assistant_is_garbage_collected(self):
        assistant = self._make_assistant(block_size=3)
        self.causal_lm.compile(sampler="greedy")
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        assistant_ref = weakref.ref(assistant)
        del assistant
        gc.collect()
        self.assertIsNone(assistant_ref())
        # A plain call after the assisted call keeps working.
        self.causal_lm.generate(prompt_ids, stop_token_ids=None)

    def test_run_eagerly_gives_new_assisted_function(self):
        assistant = self._make_assistant(block_size=3)
        self.causal_lm.compile(sampler="greedy")
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        assisted_function = self.causal_lm._assisted_generate_cache[3]

        self.causal_lm.compile(sampler="greedy", run_eagerly=True)
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        self.assertIsNot(
            self.causal_lm._assisted_generate_cache[3], assisted_function
        )

    def test_jit_compile_gives_new_assisted_function(self):
        if keras.config.backend() == "torch":
            self.skipTest("Torch does not jit compile the generate function.")
        assistant = self._make_assistant(block_size=3)
        self.causal_lm.compile(sampler="greedy", jit_compile=True)
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        assisted_function = self.causal_lm._assisted_generate_cache[3]

        self.causal_lm.compile(sampler="greedy", jit_compile=False)
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        self.assertIsNot(
            self.causal_lm._assisted_generate_cache[3], assisted_function
        )

    def test_quantized_assistant_gives_new_assisted_function(self):
        assistant = self._make_assistant(block_size=3)
        self.causal_lm.compile(sampler="greedy")
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        assisted_function = self.causal_lm._assisted_generate_cache[3]

        assistant.quantize("int8")
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        self.assertIsNot(
            self.causal_lm._assisted_generate_cache[3], assisted_function
        )

    def test_generate_with_assistant_does_not_track_assistant(self):
        assistant = self._make_assistant(block_size=3)
        num_weights = len(self.causal_lm.weights)
        num_trainable_weights = len(self.causal_lm.trainable_weights)
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.causal_lm.preprocessor = None
        self.causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        self.assertEqual(len(self.causal_lm.weights), num_weights)
        self.assertEqual(
            len(self.causal_lm.trainable_weights), num_trainable_weights
        )

    def test_draft_block_matches_full_context(self):
        # The drafter must see the target context for every position
        # before the anchor. Compare the seeded cache against one call
        # that writes exactly that context.
        assistant = self._make_assistant(block_size=3)
        vocab_size = self.preprocessor.tokenizer.vocabulary_size()
        seq_len, anchor = 8, 5
        rng = np.random.default_rng(0)
        token_ids = ops.convert_to_tensor(
            rng.integers(0, vocab_size, size=(1, seq_len)).astype("int32")
        )
        padding_mask = ops.ones((1, seq_len), dtype="int32")
        _, _, context = self.causal_lm._build_cache(
            token_ids, padding_mask, target_layer_ids=[1, 2]
        )
        assistant_backbone = assistant.backbone
        empty_cache = ops.zeros(
            (
                1,
                assistant_backbone.num_layers,
                2,
                seq_len,
                assistant_backbone.num_key_value_heads,
                assistant_backbone.head_dim,
            )
        )
        last_context = context[:, anchor - 1 : anchor, :]

        # `generate()` seeds every slot, also the slots after the anchor.
        seeded_cache = self.causal_lm._write_assistant_context(
            assistant, context, empty_cache, 0
        )
        logits, _ = self.causal_lm._draft_block(
            assistant, token_ids, anchor, last_context, seeded_cache
        )

        noise_ids = ops.concatenate(
            [
                token_ids[:, anchor : anchor + 1],
                ops.zeros((1, 2), dtype="int32"),
            ],
            axis=1,
        )
        reference_hidden, _ = assistant.call_with_cache(
            noise_embeds=self.backbone.token_embedding(noise_ids),
            context_hidden_states=context[:, :anchor, :],
            cache=empty_cache,
            cache_update_index=0,
        )
        reference_logits = self.causal_lm._apply_logit_softcap(
            self.backbone.token_embedding(
                reference_hidden[:, 1:, :], reverse=True
            )
        )
        self.assertAllClose(logits, reference_logits, atol=1e-5, rtol=1e-5)

        # Without the seed, the drafter attends to zero key/value.
        unseeded_logits, _ = self.causal_lm._draft_block(
            assistant, token_ids, anchor, last_context, empty_cache
        )
        self.assertNotAllClose(
            ops.convert_to_numpy(unseeded_logits),
            ops.convert_to_numpy(reference_logits),
        )

    def test_generate_with_assistant_multi_cycle(self):
        # block_size=2 (1 candidate per cycle) over a long sequence forces
        # several drafting cycles, exercising the assistant's persistent
        # context cache actually growing across cycles (not just a single
        # write) — see MuseGlimmerTextAttention's docstring.
        assistant = self._make_assistant(block_size=2)
        self.causal_lm.compile(sampler="greedy")
        self.causal_lm.preprocessor = None

        vocab_size = self.preprocessor.tokenizer.vocabulary_size()
        seq_len, prompt_len = 16, 4
        token_ids = ops.convert_to_tensor(
            np.random.randint(0, vocab_size, size=(1, seq_len)).astype("int32")
        )
        padding_mask = ops.convert_to_tensor(
            np.array(
                [[1] * prompt_len + [0] * (seq_len - prompt_len)],
                dtype="int32",
            )
        )
        prompt_ids = {"token_ids": token_ids, "padding_mask": padding_mask}

        reference_output = self.causal_lm.generate(
            prompt_ids, stop_token_ids=None
        )
        target_call_with_cache = self.causal_lm.call_with_cache
        num_verify_calls = [0]

        def count_verify_calls(*args, **kwargs):
            # Only `_build_cache` and `verify_next` pass `target_layer_ids`.
            if kwargs.get("target_layer_ids") is not None:
                num_verify_calls[0] += 1
            return target_call_with_cache(*args, **kwargs)

        with (
            patch.object(
                assistant,
                "call_with_cache",
                wraps=assistant.call_with_cache,
            ) as draft_mock,
            patch.object(
                self.causal_lm,
                "call_with_cache",
                wraps=count_verify_calls,
            ),
        ):
            output = self.causal_lm.generate(
                prompt_ids, stop_token_ids=None, assistant_model=assistant
            )
        self.assertAllEqual(output["token_ids"], reference_output["token_ids"])
        self.assertAllEqual(
            output["padding_mask"], reference_output["padding_mask"]
        )
        # One seed call, then two drafter calls per verify cycle: one
        # context write and one block draft. Traced backends trace the
        # cycle once; eager backends run it once per cycle.
        num_cycles = num_verify_calls[0] - 1
        self.assertEqual(draft_mock.call_count, 1 + 2 * num_cycles)
        # The seed call writes the context for every position.
        seed_context = draft_mock.call_args_list[0].kwargs[
            "context_hidden_states"
        ]
        # Read the static shape. A traced tensor is out of scope here.
        self.assertEqual(seed_context.shape[1], seq_len)

    def _randomize_weights(self, model, seed):
        # Random weights make attention and RoPE matter for the output.
        rng = np.random.default_rng(seed)
        for variable in model.weights:
            shape = tuple(variable.shape)
            if len(shape) == 1:
                value = 1.0 + 0.3 * rng.standard_normal(shape)
            else:
                value = rng.standard_normal(shape) / np.sqrt(shape[0])
            variable.assign(value.astype("float32"))

    def _make_random_causal_lm(self):
        backbone = MuseGlimmerBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=4,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            head_dim=4,
            sliding_window_size=3,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        self._randomize_weights(backbone, seed=0)
        return MuseGlimmerCausalLM(backbone=backbone, preprocessor=None)

    def _make_rope_assistant(self, block_size, seed):
        # Every assistant layer applies RoPE, so each block position gives
        # different logits.
        backbone = MuseGlimmerBackbone(
            vocabulary_size=1,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            sliding_window_size=3,
            layer_types=["sliding_attention", "full_attention"],
            use_bidirectional_attention=True,
            context_projection_layer_ids=[1, 2],
            use_external_embeddings=True,
            enable_qk_scale_and_gate=False,
            qk_norm_with_scale=True,
            use_sandwich_norm=False,
            use_centered_norm=False,
        )
        self._randomize_weights(backbone, seed)
        return MuseGlimmerAssistantCausalLM(
            backbone=backbone, block_size=block_size, mask_token_id=0
        )

    def _empty_cache(self, backbone, length):
        return ops.zeros(
            (
                1,
                backbone.num_layers,
                2,
                length,
                backbone.num_key_value_heads,
                backbone.head_dim,
            )
        )

    def test_cached_decoding_matches_full_forward(self):
        model = self._make_random_causal_lm()
        seq_len = 9
        rng = np.random.default_rng(1)
        token_ids = rng.integers(
            0, model.backbone.vocabulary_size, size=(1, seq_len)
        ).astype("int32")
        full_logits = ops.convert_to_numpy(
            model(
                {
                    "token_ids": token_ids,
                    "padding_mask": np.ones((1, seq_len), "int32"),
                }
            )
        )
        cache = self._empty_cache(model.backbone, seq_len)
        for i in range(seq_len):
            logits, _, cache = model.call_with_cache(
                ops.convert_to_tensor(token_ids[:, i : i + 1]),
                cache,
                i,
                padding_mask=None,
            )
            self.assertAllClose(
                logits[:, 0],
                full_logits[:, i],
                atol=1e-4,
                rtol=1e-4,
                msg=f"Cached step {i} differs from the full forward pass.",
            )

    def test_cached_decoding_after_padded_prompt_matches_full_forward(self):
        model = self._make_random_causal_lm()
        backbone = model.backbone
        seq_len, prompt_len = 9, 5
        rng = np.random.default_rng(2)
        token_ids = rng.integers(
            0, backbone.vocabulary_size, size=(1, seq_len)
        ).astype("int32")
        inputs = {
            "token_ids": token_ids,
            "padding_mask": np.ones((1, seq_len), "int32"),
        }
        full_logits = ops.convert_to_numpy(model(inputs))
        full_hidden = ops.convert_to_numpy(
            keras.Model(backbone.input, backbone.output)(inputs)
        )
        # The prompt is right-padded. The padded slots hold other tokens
        # than the later steps write.
        padded_token_ids = token_ids.copy()
        padded_token_ids[:, prompt_len:] = 0
        padding_mask = np.array(
            [[1] * prompt_len + [0] * (seq_len - prompt_len)], "int32"
        )
        hidden, cache = model._build_cache(
            ops.convert_to_tensor(padded_token_ids),
            ops.convert_to_tensor(padding_mask),
        )
        self.assertAllClose(
            hidden[:, :prompt_len],
            full_hidden[:, :prompt_len],
            atol=1e-4,
            rtol=1e-4,
        )
        for i in range(prompt_len, seq_len):
            logits, _, cache = model.call_with_cache(
                ops.convert_to_tensor(token_ids[:, i : i + 1]),
                cache,
                i,
                padding_mask=None,
            )
            self.assertAllClose(
                logits[:, 0],
                full_logits[:, i],
                atol=1e-4,
                rtol=1e-4,
                msg=f"Cached step {i} differs from the full forward pass.",
            )

    @parameterized.named_parameters(("block_size_3", 3), ("block_size_5", 5))
    def test_drafter_follows_the_verified_prefix(self, block_size):
        # `SpeculativeSampler` runs its loop traced on JAX, so only eager
        # backends give concrete values for each cycle.
        if keras.config.backend() == "jax":
            self.skipTest("The sampler loop is traced on JAX.")
        lm = self.causal_lm
        self._randomize_weights(lm, seed=3)
        assistant = self._make_rope_assistant(block_size, seed=4)
        lm.compile(sampler="greedy", run_eagerly=True)
        lm.preprocessor = None
        vocab_size = lm.backbone.vocabulary_size
        seq_len, prompt_len = 22, 4
        rng = np.random.default_rng(5)
        prompt_ids = {
            "token_ids": ops.convert_to_tensor(
                rng.integers(0, vocab_size, size=(1, seq_len)).astype("int32")
            ),
            "padding_mask": ops.convert_to_tensor(
                np.array(
                    [[1] * prompt_len + [0] * (seq_len - prompt_len)],
                    dtype="int32",
                )
            ),
        }
        reference = ops.convert_to_numpy(
            lm.generate(prompt_ids, stop_token_ids=None)["token_ids"]
        )[0]

        drafts, windows = [], []
        draft_block = lm._draft_block
        call_with_cache = lm.call_with_cache

        def record_draft_block(
            assistant_model, prompt, anchor, last_context, assistant_cache
        ):
            logits, new_cache = draft_block(
                assistant_model, prompt, anchor, last_context, assistant_cache
            )
            drafts.append(
                {
                    "prompt": ops.convert_to_numpy(prompt).copy(),
                    "anchor": int(ops.convert_to_numpy(anchor)),
                    "logits": ops.convert_to_numpy(logits),
                }
            )
            return logits, new_cache

        def record_verify(*args, **kwargs):
            # Only the verify window has `block_size` tokens. The seed call
            # of `generate_step` covers the whole buffer.
            if (
                kwargs.get("target_layer_ids") is not None
                and ops.shape(args[0])[1] == block_size
            ):
                windows.append(
                    {
                        "tokens": ops.convert_to_numpy(args[0]).copy(),
                        "start": int(ops.convert_to_numpy(args[2])),
                    }
                )
            return call_with_cache(*args, **kwargs)

        with (
            patch.object(lm, "_draft_block", record_draft_block),
            patch.object(lm, "call_with_cache", record_verify),
        ):
            output = lm.generate(
                prompt_ids, stop_token_ids=None, assistant_model=assistant
            )
        self.assertAllEqual(output["token_ids"][0], reference)
        self.assertGreaterEqual(len(drafts), 3)
        self.assertEqual(len(drafts), len(windows))

        num_candidates = block_size - 1
        num_checked = 0
        for cycle, (draft, window) in enumerate(zip(drafts, windows)):
            anchor, prompt = draft["anchor"], draft["prompt"]
            # The anchor token and the prefix before it are verified.
            self.assertAllEqual(
                prompt[0, : anchor + 1], reference[: anchor + 1]
            )

            # The drafter must match one call on the full verified prefix.
            _, _, context = lm._build_cache(
                ops.convert_to_tensor(prompt),
                ops.ones((1, seq_len), dtype="int32"),
                target_layer_ids=[1, 2],
            )
            noise_ids = np.zeros((1, block_size), "int32")
            noise_ids[0, 0] = prompt[0, anchor]
            hidden, _ = assistant.call_with_cache(
                noise_embeds=lm.backbone.token_embedding(
                    ops.convert_to_tensor(noise_ids)
                ),
                context_hidden_states=context[:, :anchor, :],
                cache=self._empty_cache(assistant.backbone, seq_len),
                cache_update_index=0,
                padding_mask=ops.ones((1, block_size), dtype="int32"),
            )
            reference_logits = lm._apply_logit_softcap(
                lm.backbone.token_embedding(hidden[:, 1:, :], reverse=True)
            )
            self.assertAllClose(
                draft["logits"],
                reference_logits,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Cycle {cycle} drafts from the wrong context.",
            )

            # A window that starts at the anchor holds the draft tokens.
            # Candidate `i` is the argmax of block position `i`.
            if window["start"] != anchor:
                continue
            drafted = window["tokens"][0, 1:]
            self.assertAllEqual(
                drafted,
                np.argmax(draft["logits"][0], axis=-1),
                msg=f"Cycle {cycle} uses the wrong block position.",
            )
            # The next anchor is the last verified position. The verifier
            # accepts the draft tokens that match the plain greedy output.
            expected = reference[anchor + 1 : anchor + 1 + num_candidates]
            num_accepted = 0
            for draft_token, expected_token in zip(drafted, expected):
                if draft_token != expected_token:
                    break
                num_accepted += 1
            if cycle + 1 < len(drafts):
                self.assertEqual(
                    drafts[cycle + 1]["anchor"],
                    anchor + num_accepted + 1,
                    msg=f"Cycle {cycle + 1} starts at the wrong anchor.",
                )
            num_checked += 1
        self.assertGreaterEqual(num_checked, 2)

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_saved_model(self, model_type):
        if model_type == "text_and_vision":
            init_kwargs = self.multimodal_init_kwargs
            input_data = self.multimodal_input_data
        else:
            init_kwargs = self.init_kwargs
            input_data = self.input_data

        self.run_model_saving_test(
            cls=MuseGlimmerCausalLM,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )
