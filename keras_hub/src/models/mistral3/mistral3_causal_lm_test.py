from unittest.mock import patch

import numpy as np
import pytest
from keras import ops
from keras import tree

from keras_hub.src.models.mistral3.mistral3_backbone import Mistral3Backbone
from keras_hub.src.models.mistral3.mistral3_causal_lm import Mistral3CausalLM
from keras_hub.src.models.mistral3.mistral3_causal_lm_preprocessor import (
    Mistral3CausalLMPreprocessor,
)
from keras_hub.src.models.mistral3.mistral3_image_converter import (
    Mistral3ImageConverter,
)
from keras_hub.src.models.mistral3.mistral3_tokenizer import (
    MISTRAL3_TEKKEN_SPLIT_PATTERN as _TEKKEN_SPLIT_PATTERN,
)
from keras_hub.src.models.mistral3.mistral3_tokenizer import Mistral3Tokenizer
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    Mistral3MultiModalProjector,
)
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    Mistral3VisionEncoder,
)
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    compute_image_placeholder_indices,
)
from keras_hub.src.tests.test_case import TestCase


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _tekken_vision_init_kwargs():
    """Build a tiny Tekken (byte-level BPE) vocabulary with image tokens."""
    byte_encoder = _bytes_to_unicode()
    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<pad>",
        "[INST]",
        "[IMG]",
        "[IMG_BREAK]",
        "[IMG_END]",
    ]
    vocabulary = {token: i for i, token in enumerate(special_tokens)}
    offset = len(special_tokens)
    for i in range(256):
        vocabulary[byte_encoder[i]] = offset + i
    merges = []
    next_id = offset + 256
    for a, b in [("t", "h"), ("th", "e"), ("i", "n")]:
        vocabulary[a + b] = next_id
        merges.append(f"{a} {b}")
        next_id += 1
    return {
        "vocabulary": vocabulary,
        "merges": merges,
        "split_pattern": _TEKKEN_SPLIT_PATTERN,
    }


class Mistral3CausalLMTest(TestCase):
    def setUp(self):
        self.tokenizer = Mistral3Tokenizer(**_tekken_vision_init_kwargs())
        self.image_converter = Mistral3ImageConverter(
            longest_edge=8, patch_size=4, spatial_merge_size=1
        )
        self.preprocessor = Mistral3CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=16,
            spatial_merge_size=1,
        )
        vision_encoder = Mistral3VisionEncoder(
            image_size=8,
            patch_size=4,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            intermediate_dim=8,
        )
        multimodal_projector = Mistral3MultiModalProjector(
            vision_hidden_dim=8,
            text_hidden_dim=16,
            spatial_merge_size=1,
            patch_size=4,
            image_size=8,
        )
        self.backbone = Mistral3Backbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=8,
            num_key_value_heads=4,
            hidden_dim=16,
            intermediate_dim=8,
            sliding_window=2,
            vision_encoder=vision_encoder,
            multimodal_projector=multimodal_projector,
            image_token_index=self.tokenizer.image_placeholder_token_id,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (
            {
                "prompts": ["the [IMG] tin", "in [IMG] the"],
                "images": [
                    [np.zeros((8, 8, 3), dtype="float32")],
                    [np.ones((8, 8, 3), dtype="float32")],
                ],
            },
        )
        self.input_data = tree.map_structure(
            ops.convert_to_tensor, self.preprocessor(*self.train_data)[0]
        )

    def test_multimodal_generate(self):
        vision_encoder = Mistral3VisionEncoder(
            image_size=8,
            patch_size=4,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            intermediate_dim=8,
        )
        multimodal_projector = Mistral3MultiModalProjector(
            vision_hidden_dim=8,
            text_hidden_dim=8,
            spatial_merge_size=1,
            patch_size=4,
            image_size=8,
        )
        # `image_token_index` must stay inside `vocabulary_size`, since the
        # token embedding lookup happens before the image-text merger
        # overwrites those positions.
        image_token_index = 9
        backbone = Mistral3Backbone(
            vocabulary_size=10,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            vision_encoder=vision_encoder,
            multimodal_projector=multimodal_projector,
            image_token_index=image_token_index,
        )
        causal_lm = Mistral3CausalLM(backbone=backbone, preprocessor=None)

        # Two 8x8 images, each a 2x2 patch grid; `spatial_merge_size=1`
        # makes every patch its own merge window (4 rows per image).
        # Followed by one real token, then padding for incremental decoding.
        token_ids = ops.array(
            [
                [image_token_index] * 4 + [3, 0, 0],
                [image_token_index] * 4 + [4, 0, 0],
            ],
            dtype="int32",
        )
        padding_mask = ops.array(
            [
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
            ],
        )
        placeholder_indices = compute_image_placeholder_indices(
            token_ids, image_token_index=image_token_index
        )
        input_data = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": ops.convert_to_tensor(
                np.random.rand(2, 3, 8, 8).astype("float32")
            ),
            "image_sizes": ops.array([[8, 8], [8, 8]], dtype="int32"),
            "placeholder_indices": ops.convert_to_tensor(placeholder_indices),
        }
        output = causal_lm.generate(input_data, stop_token_ids=None)
        self.assertEqual(ops.shape(output["token_ids"]), (2, 7))
        self.assertEqual(ops.shape(output["padding_mask"]), (2, 7))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Mistral3CausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_generate(self):
        causal_lm = Mistral3CausalLM(**self.init_kwargs)
        prompt = "the tin"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        prompts = ["the tin", "in the"]
        outputs = causal_lm.generate(prompts)
        for prompt, output in zip(prompts, outputs):
            self.assertTrue(prompt in output)
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        self.assertAllEqual(
            outputs["token_ids"][:, :2], prompt_ids["token_ids"][:, :2]
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :2], prompt_ids["padding_mask"][:, :2]
        )

    def test_early_stopping(self):
        causal_lm = Mistral3CausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the tin", "in the"]
            output = causal_lm.generate(prompt)
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = Mistral3CausalLM(**self.init_kwargs)
        causal_lm.generate("the tin")
        first_fn = causal_lm.generate_function
        causal_lm.generate("the tin")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        token_ids = ops.array([[1, 1824, 349, 524, 11234, 28804]])
        input_data = {
            "token_ids": token_ids,
            "padding_mask": ops.ones_like(token_ids),
            "pixel_values": ops.zeros((0, 3, 14, 14), dtype="float32"),
            "image_sizes": ops.zeros((0, 2), dtype="int32"),
            "placeholder_indices": ops.zeros((1, 0), dtype="int32"),
        }
        for preset in Mistral3CausalLM.presets:
            self.run_preset_test(
                cls=Mistral3CausalLM,
                preset=preset,
                input_data=input_data,
            )
