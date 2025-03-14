import os

import keras
import numpy as np
import pytest

from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_for_conditional_generation import (  # noqa: E501
    MoonshineForConditionalGeneration,
)
from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineForConditionalGenerationTest(TestCase):
    def setUp(self):
        self.tokenizer = MoonshineTokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "moonshine_test_vocab.spm"
            )
        )
        self.vocab_size = 1036
        hidden_dim = 32
        self.audio_converter = MoonshineAudioConverter(
            filter_dim=hidden_dim,
            sampling_rate=16000,
            do_normalize=False,
            return_attention_mask=True,
            padding_value=0.0,
            initializer_range=0.02,
        )
        self.backbone = MoonshineBackbone(
            vocabulary_size=self.vocab_size,
            hidden_dim=hidden_dim,
            encoder_num_layers=2,
            decoder_num_layers=2,
            encoder_num_heads=4,
            decoder_num_heads=4,
            intermediate_dim=hidden_dim * 4,
            feedforward_expansion_factor=4,
            encoder_use_swiglu_activation=False,
            decoder_use_swiglu_activation=True,
            max_position_embeddings=2048,
            pad_head_dim_to_multiple_of=None,
            partial_rotary_factor=0.62,
            dropout=0.0,
            initializer_range=0.02,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            rope_scaling=None,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "audio_converter": self.audio_converter,
            "tokenizer": self.tokenizer,
        }
        self.audio_input = keras.ops.convert_to_tensor(
            np.random.randn(1, 16000).astype("float32")
        )
        self.token_ids = keras.ops.array(
            [[1, 3, 8, 4, 6, 2, 0, 0]], dtype="int32"
        )
        self.input_data = {
            "audio": self.audio_input,
            "token_ids": self.token_ids,
        }

    def test_model_basics(self):
        model = MoonshineForConditionalGeneration(**self.init_kwargs)
        outputs = model(self.input_data)
        expected_shape = (1, self.token_ids.shape[1], self.vocab_size)
        self.assertAllEqual(keras.ops.shape(outputs), expected_shape)
        generated_ids = model.generate(self.audio_input, max_new_tokens=3)
        self.assertIsNotNone(generated_ids)
        self.assertEqual(generated_ids.shape[0], 1)

    def test_generate_functionality(self):
        model = MoonshineForConditionalGeneration(**self.init_kwargs)
        batch_audio = np.concatenate(
            [self.audio_input, self.audio_input], axis=0
        )
        generated_ids = model.generate(batch_audio, max_new_tokens=5)
        self.assertEqual(generated_ids.shape[0], 2)
        self.assertGreaterEqual(generated_ids.shape[1], 1)
        test_model = MoonshineForConditionalGeneration(**self.init_kwargs)
        custom_audio = np.ones((1, 16000), dtype="float32")
        generated = test_model.generate(custom_audio, max_new_tokens=10)
        self.assertIsNotNone(generated)

    def test_training_flow(self):
        model = MoonshineForConditionalGeneration(**self.init_kwargs)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        x = self.input_data
        y = keras.ops.array([[3, 8, 4, 6, 2, 0, 0, 0]], dtype="int32")
        model.fit(x, y, batch_size=1, epochs=1, verbose=0)
        trainable_count = sum(
            keras.ops.size(w) for w in model.trainable_weights
        )
        self.assertGreater(trainable_count, 0)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in getattr(MoonshineForConditionalGeneration, "presets", []):
            self.run_preset_test(
                cls=MoonshineForConditionalGeneration,
                preset=preset,
                input_data=self.input_data,
            )
