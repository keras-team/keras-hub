from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm import Qwen2VLCausalLM
from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm_preprocessor import (
    Qwen2VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLCausalLMTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "\u0120air", "plane", "\u0120at", "port"]
        self.vocab += ["<|endoftext|>"]
        self.vocab += ["<|eot_id|>"]
        self.vocab += ["<|vision_start|>"]
        self.vocab += ["<|vision_end|>"]
        self.vocab += ["<|image_pad|>"]
        self.vocab = dict(
            [(token, i) for i, token in enumerate(self.vocab)]
        )
        self.merges = [
            "\u0120 a", "\u0120 t", "\u0120 i", "\u0120 b",
            "a i", "p l", "n e",
        ]
        self.merges += [
            "\u0120a t", "p o", "r t", "\u0120t h", "ai r",
            "pl a", "po rt",
        ]
        self.merges += ["\u0120ai r", "\u0120a i", "pla ne"]
        self.preprocessor = Qwen2VLCausalLMPreprocessor(
            Qwen2VLTokenizer(
                vocabulary=self.vocab, merges=self.merges
            ),
            sequence_length=7,
        )
        self.backbone = Qwen2VLBackbone(
            vocabulary_size=(
                self.preprocessor.tokenizer.vocabulary_size()
            ),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=32,
            intermediate_dim=64,
            mrope_section=[1, 1, 2],
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            {
                "prompts": [" airplane at airport"] * 2,
                "responses": [" airplane"] * 2,
            },
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Qwen2VLCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 7, 11),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen2VLCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
