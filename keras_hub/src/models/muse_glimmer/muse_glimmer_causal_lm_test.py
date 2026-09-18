from unittest.mock import patch

import numpy as np
import pytest
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


class MuseGlimmerCausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|end_of_text|>", "<|begin_of_text|>"]
        self.vocab += ["<|finetune_right_pad|>"]
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
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=MuseGlimmerCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(
                2,
                7,
                self.preprocessor.tokenizer.vocabulary_size(),
            ),
        )

    def test_generate(self):
        causal_lm = MuseGlimmerCausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(" airplane at airport")
        self.assertTrue(prompt in output)
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        self.assertAllEqual(
            outputs["token_ids"][:, :5], prompt_ids["token_ids"][:, :5]
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5], prompt_ids["padding_mask"][:, :5]
        )

    def test_generate_with_image_under_compiled_backend(self):
        image_converter = MuseGlimmerImageConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            max_image_tokens=64,
            scale=1 / 255.0,
        )
        tokenizer = self.preprocessor.tokenizer
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
            vocabulary_size=tokenizer.vocabulary_size(),
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
        causal_lm = MuseGlimmerCausalLM(backbone=backbone, preprocessor=None)

        image = np.random.randint(0, 255, (8, 8, 3)).astype("float32")
        result = image_converter(image)
        t, h, w = (int(v) for v in ops.convert_to_numpy(result["grid_thw"]))
        num_image_tokens = (
            t
            * (h // image_converter.merge_size)
            * (w // image_converter.merge_size)
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

        output = causal_lm.generate(
            inputs, max_length=sequence_length, stop_token_ids=None
        )
        self.assertEqual(ops.shape(output["token_ids"]), (1, sequence_length))

    def test_early_stopping(self):
        causal_lm = MuseGlimmerCausalLM(**self.init_kwargs)
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

    def test_generate_with_assistant(self):
        # DFlash speculative decoding: a tiny assistant backbone reusing
        # the target's hidden_dim (noise_embeds come from the target's own
        # token_embedding) and target_layer_ids=[1, 2] (valid against the
        # target's num_layers=4).
        target_layer_ids = [1, 2]
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
            context_projection_layer_ids=target_layer_ids,
            use_external_embeddings=True,
            enable_qk_scale_and_gate=False,
            use_sandwich_norm=False,
        )
        assistant = MuseGlimmerAssistantCausalLM(
            backbone=assistant_backbone,
            block_size=3,
            mask_token_id=0,
        )
        causal_lm = MuseGlimmerCausalLM(**self.init_kwargs)
        # Greedy on both sides: speculative decoding's accept/reject step
        # is only guaranteed to reproduce the target model's own output
        # exactly under matching (here, greedy) acceptance semantics.
        causal_lm.compile(sampler="greedy")
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        causal_lm.preprocessor = None
        reference_output = causal_lm.generate(prompt_ids, stop_token_ids=None)

        output = causal_lm.generate(
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
        self.assertIsNone(getattr(causal_lm, "_assistant_model", None))

    def test_generate_with_assistant_multi_cycle(self):
        # block_size=2 (1 candidate per cycle) over a long sequence forces
        # several drafting cycles, exercising the assistant's persistent
        # context cache actually growing across cycles (not just a single
        # write) — see MuseGlimmerTextAttention's docstring.
        target_layer_ids = [1, 2]
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
            context_projection_layer_ids=target_layer_ids,
            use_external_embeddings=True,
            enable_qk_scale_and_gate=False,
            use_sandwich_norm=False,
        )
        assistant = MuseGlimmerAssistantCausalLM(
            backbone=assistant_backbone,
            block_size=2,
            mask_token_id=0,
        )
        causal_lm = MuseGlimmerCausalLM(**self.init_kwargs)
        causal_lm.compile(sampler="greedy")
        causal_lm.preprocessor = None

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

        reference_output = causal_lm.generate(prompt_ids, stop_token_ids=None)
        output = causal_lm.generate(
            prompt_ids, stop_token_ids=None, assistant_model=assistant
        )
        self.assertAllEqual(output["token_ids"], reference_output["token_ids"])
        self.assertAllEqual(
            output["padding_mask"], reference_output["padding_mask"]
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MuseGlimmerCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
