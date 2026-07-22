import os
from unittest.mock import patch

import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.hrm_text.hrm_text_backbone import HrmTextBackbone
from keras_hub.src.models.hrm_text.hrm_text_causal_lm import HrmTextCausalLM
from keras_hub.src.models.hrm_text.hrm_text_causal_lm_preprocessor import (
    HrmTextCausalLMPreprocessor,
)
from keras_hub.src.models.hrm_text.hrm_text_tokenizer import HrmTextTokenizer
from keras_hub.src.tests.test_case import TestCase


class HrmTextCausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += [
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
        ]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        vocabulary = []
        for merge in self.merges:
            left, right = merge.split(" ")
            vocabulary.extend([left, right, left + right])
        vocabulary += ["!", "<|im_start|>", "<|box_end|>", "<|endoftext|>"]
        vocabulary = dict(
            (token, index)
            for index, token in enumerate(sorted(set(vocabulary)))
        )
        for token in (
            "<|im_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
        ):
            vocabulary[token] = len(vocabulary)
        self.preprocessor = HrmTextCausalLMPreprocessor(
            HrmTextTokenizer(vocabulary=vocabulary, merges=self.merges),
            sequence_length=7,
        )
        self.backbone = HrmTextBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            hidden_dim=16,
            intermediate_dim=32,
            num_layers_per_stack=2,
            num_attention_heads=4,
            head_dim=4,
            h_cycles=2,
            l_cycles=2,
            max_sequence_length=8,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=HrmTextCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 7, self.backbone.vocabulary_size),
        )

    def test_prefix_lm_response_weights(self):
        inputs, _, sample_weight = self.preprocessor(
            {"prefix": [" airplane"], "response": [" at airport"]}
        )
        self.assertEqual(inputs["token_type_ids"][0, 0], 1)
        self.assertTrue(
            np.any(ops.convert_to_numpy(inputs["token_type_ids"][0] == 0))
        )
        self.assertEqual(sample_weight[0, 0], 0)
        self.assertTrue(np.any(ops.convert_to_numpy(sample_weight[0] == 1)))

    def test_generate(self):
        causal_lm = HrmTextCausalLM(**self.init_kwargs)
        prompt = " airplane"
        output = causal_lm.generate(prompt)
        self.assertTrue(isinstance(output, str))
        self.assertTrue(prompt in output)
        prompt_ids = self.preprocessor.generate_preprocess(
            [prompt], sequence_length=7
        )
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        self.assertAllEqual(
            outputs["token_ids"][:, :2], prompt_ids["token_ids"][:, :2]
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :2], prompt_ids["padding_mask"][:, :2]
        )

    def test_generate_strip_prompt(self):
        causal_lm = HrmTextCausalLM(**self.init_kwargs)
        prompt = " airplane"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_generate_compilation(self):
        causal_lm = HrmTextCausalLM(**self.init_kwargs)
        causal_lm.generate(" airplane")
        first_function = causal_lm.generate_function
        causal_lm.generate(" airplane")
        self.assertEqual(first_function, causal_lm.generate_function)
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    def test_early_stopping_with_unequal_prompts(self):
        causal_lm = HrmTextCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            logits = ops.slice_update(
                logits,
                (0, 0, index),
                ops.expand_dims(update, axis=-1),
            )
            return logits, hidden_states, cache

        prompts = [" airplane at airport", " airplane"]
        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            outputs = causal_lm.generate(prompts)
        self.assertEqual(outputs, prompts)

    def test_tied_and_untied_embeddings(self):
        for tie_word_embeddings in (True, False):
            backbone = HrmTextBackbone(
                vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
                hidden_dim=16,
                intermediate_dim=32,
                num_layers_per_stack=1,
                num_attention_heads=4,
                head_dim=4,
                h_cycles=1,
                l_cycles=1,
                tie_word_embeddings=tie_word_embeddings,
            )
            model = HrmTextCausalLM(backbone, self.preprocessor)
            outputs = model(self.input_data)
            self.assertEqual(
                outputs.shape[-1], self.preprocessor.tokenizer.vocabulary_size()
            )
            self.assertEqual(
                backbone.token_embedding.tie_weights, tie_word_embeddings
            )

    def test_l_bp_cycles_control_training_gradients(self):
        def make_model(l_bp_cycles):
            backbone = HrmTextBackbone(
                vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
                hidden_dim=16,
                intermediate_dim=32,
                num_layers_per_stack=1,
                num_attention_heads=4,
                head_dim=4,
                h_cycles=2,
                l_cycles=2,
                l_bp_cycles=l_bp_cycles,
            )
            model = HrmTextCausalLM(backbone, self.preprocessor)
            model.compile(
                optimizer=keras.optimizers.SGD(learning_rate=0.01),
                loss=keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                ),
            )
            return model

        train_data = {
            "prefix": [" airplane", " airplane"],
            "response": [" at airport", " at airport"],
        }
        frozen_model = make_model([0, 0])
        frozen_before = [
            ops.convert_to_numpy(weight).copy()
            for weight in frozen_model.backbone.L_module.weights
        ]
        frozen_model.fit(train_data, batch_size=2, epochs=1, verbose=0)
        frozen_after = [
            ops.convert_to_numpy(weight)
            for weight in frozen_model.backbone.L_module.weights
        ]
        for before, after in zip(frozen_before, frozen_after):
            self.assertAllClose(before, after)

        enabled_model = make_model([2, 2])
        enabled_before = [
            ops.convert_to_numpy(weight).copy()
            for weight in enabled_model.backbone.L_module.weights
        ]
        enabled_model.fit(train_data, batch_size=2, epochs=1, verbose=0)
        enabled_after = [
            ops.convert_to_numpy(weight)
            for weight in enabled_model.backbone.L_module.weights
        ]
        self.assertTrue(
            any(
                not np.allclose(before, after)
                for before, after in zip(enabled_before, enabled_after)
            )
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=HrmTextCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_local_preset_round_trip(self):
        model = HrmTextCausalLM(**self.init_kwargs)
        expected = model(self.input_data)
        expected_special_token_ids = {
            "start": self.preprocessor.tokenizer.start_token_id,
            "prefix_end": self.preprocessor.tokenizer.prefix_end_token_id,
            "end": self.preprocessor.tokenizer.end_token_id,
            "pad": self.preprocessor.tokenizer.pad_token_id,
            "direct": self.preprocessor.tokenizer.direct_condition_token_id,
            "cot": self.preprocessor.tokenizer.cot_condition_token_id,
            "noisy": self.preprocessor.tokenizer.noisy_condition_token_id,
            "synth": self.preprocessor.tokenizer.synth_condition_token_id,
        }
        preset_dir = os.path.join(self.get_temp_dir(), "hrm_text_preset")
        model.save_to_preset(preset_dir)
        restored = HrmTextCausalLM.from_preset(preset_dir)
        self.assertAllClose(expected, restored(self.input_data))
        restored_tokenizer = restored.preprocessor.tokenizer
        actual_special_token_ids = {
            "start": restored_tokenizer.start_token_id,
            "prefix_end": restored_tokenizer.prefix_end_token_id,
            "end": restored_tokenizer.end_token_id,
            "pad": restored_tokenizer.pad_token_id,
            "direct": restored_tokenizer.direct_condition_token_id,
            "cot": restored_tokenizer.cot_condition_token_id,
            "noisy": restored_tokenizer.noisy_condition_token_id,
            "synth": restored_tokenizer.synth_condition_token_id,
        }
        self.assertEqual(actual_special_token_ids, expected_special_token_ids)
