import os

import numpy as np
import pytest

from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_flan_t5_lm import BLIP2FlanT5
from keras_hub.src.models.blip2.blip2_flan_t5_tokenizer import (
    BLIP2FlanT5Tokenizer,
)
from keras_hub.src.models.blip2.blip2_image_converter import BLIP2ImageConverter
from keras_hub.src.models.blip2.blip2_qformer import BLIP2QFormer
from keras_hub.src.models.blip2.blip2_seq_2_seq_lm import BLIP2Seq2SeqLM
from keras_hub.src.models.blip2.blip2_seq_2_seq_lm_preprocessor import (
    BLIP2Seq2SeqLMPreprocessor,
)
from keras_hub.src.models.blip2.blip2_vision_encoder import BLIP2VisionEncoder
from keras_hub.src.tests.test_case import TestCase


class BLIP2Seq2SeqLMTest(TestCase):
    def setUp(self):
        tokenizer = BLIP2FlanT5Tokenizer(
            proto=os.path.join(self.get_test_data_dir(), "t5_test_vocab.spm")
        )
        vocabulary_size = tokenizer.vocabulary_size()

        self.preprocessor = BLIP2Seq2SeqLMPreprocessor(
            tokenizer=tokenizer,
            image_converter=BLIP2ImageConverter(image_size=(32, 32)),
            encoder_sequence_length=8,
            decoder_sequence_length=8,
        )
        vision_encoder = BLIP2VisionEncoder(
            image_size=32,
            patch_size=8,
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            use_patch_bias=True,
            use_class_token=True,
            use_mha_bias=True,
            use_mlp_bias=True,
            dropout_rate=0.0,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
        )
        qformer = BLIP2QFormer(
            num_query_tokens=4,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            vision_dim=16,
            cross_attention_frequency=2,
            dropout=0.0,
            layer_norm_epsilon=1e-5,
        )
        language_model = BLIP2FlanT5(
            vocabulary_size=vocabulary_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            key_value_dim=8,
            num_query_tokens=4,
            qformer_hidden_dim=8,
            dropout=0.0,
        )
        backbone = BLIP2Backbone(
            vision_encoder=vision_encoder,
            qformer=qformer,
            language_model=language_model,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": backbone,
        }
        self.train_data = (
            {
                "images": np.ones((2, 32, 32, 3), dtype="float32"),
                "encoder_text": ["a photo of a cat", "a photo of a dog"],
                "decoder_text": ["a cat sitting", "a running dog"],
            },
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_seq_2_seq_lm_basics(self):
        self.run_task_test(
            cls=BLIP2Seq2SeqLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(
                2,
                8,
                self.preprocessor.tokenizer.vocabulary_size(),
            ),
        )

    def test_generate_compilation(self):
        seq_2_seq_lm = BLIP2Seq2SeqLM(**self.init_kwargs)
        prompt = {
            "images": np.ones((2, 32, 32, 3), dtype="float32"),
            "encoder_text": ["a photo of", "a photo of"],
        }
        seq_2_seq_lm.generate(prompt, max_length=8)
        seq_2_seq_lm.compile(sampler="top_k")
        seq_2_seq_lm.generate(prompt, max_length=8)

    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BLIP2Seq2SeqLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            atol=5e-3,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BLIP2Seq2SeqLM.presets:
            self.run_preset_test(
                cls=BLIP2Seq2SeqLM,
                preset=preset,
                input_data=self.input_data,
            )
