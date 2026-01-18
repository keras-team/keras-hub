import os.path

import numpy as np
import pytest

from keras_hub.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_hub.src.models.pali_gemma.pali_gemma_causal_lm import (
    PaliGemmaCausalLM,
)
from keras_hub.src.models.pali_gemma.pali_gemma_causal_lm_preprocessor import (
    PaliGemmaCausalLMPreprocessor,
)
from keras_hub.src.models.pali_gemma.pali_gemma_image_converter import (
    PaliGemmaImageConverter,
)
from keras_hub.src.models.pali_gemma.pali_gemma_tokenizer import (
    PaliGemmaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class PaliGemmaCausalLMTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.text_sequence_length = 16
        self.image_size = 16
        self.dummy_text = [
            "the quick brown fox" for _ in range(self.batch_size)
        ]
        self.dummy_images = np.random.uniform(size=(self.batch_size, 20, 20, 3))

        proto = "gemma_test_vocab.spm"
        tokenizer = PaliGemmaTokenizer(
            os.path.join(self.get_test_data_dir(), proto)
        )
        image_converter = PaliGemmaImageConverter(
            image_size=(16, 16),
        )
        self.vocabulary_size = tokenizer.vocabulary_size()
        self.preprocessor = PaliGemmaCausalLMPreprocessor(
            tokenizer,
            image_converter,
            self.text_sequence_length,
            add_start_token=False,
            add_end_token=False,
        )

        self.backbone = PaliGemmaBackbone(
            vocabulary_size=self.vocabulary_size,
            image_size=self.image_size,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vit_patch_size=4,
            vit_num_layers=2,
            vit_num_heads=2,
            vit_hidden_dim=8,
            vit_intermediate_dim=16,
        )
        self.train_data = (
            {
                "images": self.dummy_images,
                "prompts": self.dummy_text,
                "responses": self.dummy_text,
            },
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=PaliGemmaCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 16, 11),
        )

    @pytest.mark.large
    def test_saved_model(self):
        input_data = {
            "token_ids": np.random.rand(
                self.batch_size, self.text_sequence_length
            ),
            "images": np.ones(
                (self.batch_size, self.image_size, self.image_size, 3)
            ),
            "padding_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "response_mask": np.zeros(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
        }
        self.run_model_saving_test(
            cls=PaliGemmaCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
        )

    def test_litert_export(self):
        input_data = {
            "token_ids": np.random.randint(
                0,
                self.vocabulary_size,
                size=(self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "images": np.ones(
                (self.batch_size, self.image_size, self.image_size, 3)
            ),
            "padding_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "response_mask": np.zeros(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
        }
        self.run_litert_export_test(
            cls=PaliGemmaCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 2e-6, "mean": 1e-6}},
        )

    def test_pali_gemma_causal_model(self):
        preprocessed, _, _ = self.preprocessor(
            {
                "images": self.dummy_images,
                "prompts": self.dummy_text,
                "responses": self.dummy_text,
            }
        )
        pali_gemma = PaliGemmaCausalLM(self.preprocessor, self.backbone)
        output = pali_gemma(inputs=preprocessed)
        self.assertAllEqual(
            output.shape,
            (self.batch_size, self.text_sequence_length, self.vocabulary_size),
        )

    def test_pali_gemma_causal_lm_fit(self):
        pali_gemma = PaliGemmaCausalLM(self.preprocessor, self.backbone)
        pali_gemma.fit(
            x={
                "images": self.dummy_images,
                "prompts": self.dummy_text,
                "responses": self.dummy_text,
            },
            batch_size=2,
        )

    def test_pali_gemma_causal_lm_generate(self):
        pali_gemma = PaliGemmaCausalLM(self.preprocessor, self.backbone)
        output = pali_gemma.generate(
            inputs={
                "images": self.dummy_images,
                "prompts": self.dummy_text,
            },
        )
        self.assertEqual(len(output), self.batch_size)
