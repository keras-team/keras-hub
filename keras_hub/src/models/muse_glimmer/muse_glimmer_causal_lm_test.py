from unittest.mock import patch

import keras
import numpy as np
import pytest
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
        # Assistant wiring must not leak into the model's state afterward.
        self.assertIsNone(getattr(self.causal_lm, "_assistant_model", None))
        # A greedy target keeps greedy acceptance.
        speculative_sampler = self.causal_lm._assisted_generate_cache[1]
        self.assertIsNone(speculative_sampler.base_sampler)

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
        speculative_sampler = self.causal_lm._assisted_generate_cache[1]
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
        assisted_function = self.causal_lm._assisted_generate_cache[2]
        # Count drafter calls. A compiled backend reuses its traced graph
        # and does not call the drafter from Python again.
        with patch.object(
            assistant, "call_with_cache", wraps=assistant.call_with_cache
        ) as draft_mock:
            self.causal_lm.generate(
                prompt_ids, stop_token_ids=None, assistant_model=assistant
            )
        self.assertIs(
            self.causal_lm._assisted_generate_cache[2], assisted_function
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
            self.causal_lm._assisted_generate_cache[2], assisted_function
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

    @pytest.mark.large
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
