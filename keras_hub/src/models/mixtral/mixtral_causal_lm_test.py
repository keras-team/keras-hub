import os

import keras
import pytest
from keras import ops

from keras_hub.src.models.mixtral.mixtral_backbone import MixtralBackbone
from keras_hub.src.models.mixtral.mixtral_causal_lm import MixtralCausalLM
from keras_hub.src.models.mixtral.mixtral_causal_lm_preprocessor import (
    MixtralCausalLMPreprocessor,
)
from keras_hub.src.models.mixtral.mixtral_tokenizer import MixtralTokenizer
from keras_hub.src.tests.test_case import TestCase


class MixtralCausalLMTest(TestCase):
    def setUp(self):
        self.preprocessor = MixtralCausalLMPreprocessor(
            MixtralTokenizer(
                # Generated using create_mixtral_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "mixtral_test_vocab.spm"
                )
            ),
            sequence_length=8,
        )
        self.backbone = MixtralBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            num_experts=2,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (["the quick brown fox", "the earth is round"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=MixtralCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, 10),
        )

    def test_generate(self):
        self.run_generate_test(
            cls=MixtralCausalLM,
            init_kwargs=self.init_kwargs,
            prompt="the quick brown fox",
        )

    def test_early_stopping(self):
        self.run_early_stopping_test(
            cls=MixtralCausalLM,
            init_kwargs=self.init_kwargs,
            prompt=["the quick brown fox", "the earth"],
        )

    def test_generate_compilation(self):
        self.run_generate_compilation_test(
            cls=MixtralCausalLM,
            init_kwargs=self.init_kwargs,
            prompt="the quick brown fox",
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MixtralCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.xfail(
        condition=keras.backend.backend() == "torch",
        reason="litert-torch cannot lower aten._assert_async from MoE routing.",
    )
    def test_litert_export(self):
        self.run_litert_export_test(
            cls=MixtralCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MixtralCausalLM.presets:
            self.run_preset_test(
                cls=MixtralCausalLM,
                preset=preset,
                input_data=self.input_data,
            )

    def test_score_logits(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = ["the quick brown fox", "the quick brown fox"]
        causal_lm = MixtralCausalLM(**self.init_kwargs)
        expected_score_shape = (2, 8, 10)

        # Preprocess prompts to get tokenized representations and padding masks.
        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]

        # Get the scores and assert their shape.
        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="logits",
        )

        self.assertEqual(ops.shape(scores), expected_score_shape)

    def test_score_loss(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = ["the quick brown fox", "the quick brown fox"]
        causal_lm = MixtralCausalLM(**self.init_kwargs)
        expected_score_shape = (2, 8)

        # Preprocess prompts to get tokenized representations and padding masks.
        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]
        target_ids = ops.roll(token_ids, shift=-1, axis=1)

        # Get the scores and assert their shape.
        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="loss",
            target_ids=target_ids,
        )

        self.assertEqual(ops.shape(scores), expected_score_shape)

    def test_score_layer_intercept_fn_exfiltration(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = ["the quick brown fox", "the quick brown fox"]
        causal_lm = MixtralCausalLM(**self.init_kwargs)
        expected_embedded_shape = (2, 8, 8)
        expected_score_shape = (2, 8, 10)

        # Preprocess prompts to get tokenized representations and padding masks.
        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]

        # Setup a custom intercept function that extracts the embeddings to a
        # a variable from the embeddings layer and otherwise asserts on shapes.
        embedded_prompts = None

        def layer_intercept_fn_for_testing(x, i):
            if i == -1:
                nonlocal embedded_prompts
                embedded_prompts = x
            else:
                nonlocal expected_embedded_shape
                self.assertEqual(ops.shape(x), expected_embedded_shape)
            return x

        # Get the scores.
        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="logits",
            layer_intercept_fn=layer_intercept_fn_for_testing,
        )

        # Assert shapes for info exfiltrated into the parent context.
        self.assertEqual(ops.shape(embedded_prompts), expected_embedded_shape)
        self.assertEqual(ops.shape(scores), expected_score_shape)
