import numpy as np
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
