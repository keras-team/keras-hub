import keras
import numpy as np
import pytest

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_to_text import (
    Qwen3ASRAudioToText,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_to_text_preprocessor import (  # noqa: E501
    Qwen3ASRAudioToTextPreprocessor,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRAudioToTextTest(TestCase):
    def setUp(self):
        self.tokenizer = Qwen3ASRTokenizer(
            vocabulary={
                "<|endoftext|>": 0,
                "<|im_end|>": 1,
                "the": 2,
                "quick": 3,
                "brown": 4,
                "fox": 5,
                "earth": 6,
                "is": 7,
                "round": 8,
                "Ġhello": 9,
            },
            merges=["t h", "th e", "q u", "qu ick"],
        )
        self.vocab_size = 512
        hidden_dim = 32
        audio_token_id = 200
        self.audio_converter = Qwen3ASRAudioConverter(
            num_mels=32,
            sampling_rate=16000,
            max_audio_length=1,
        )
        self.preprocessor = Qwen3ASRAudioToTextPreprocessor(
            audio_converter=self.audio_converter,
            tokenizer=self.tokenizer,
            audio_token_id=audio_token_id,
            decoder_sequence_length=10,
        )
        self.backbone = Qwen3ASRBackbone(
            vocabulary_size=512,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            hidden_dim=hidden_dim,
            intermediate_dim=64,
            num_mel_bins=32,
            encoder_d_model=32,
            encoder_num_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            downsample_hidden_size=16,
            audio_token_id=audio_token_id,
            n_window=40,
            max_source_positions=100,
            dtype="float32",
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        # 10 audio placeholders + 10 text token positions = 20 total.
        self.input_data = {
            "audio_features": np.random.uniform(size=(2, 80, 32)).astype(
                "float32"
            ),
            "token_ids": np.array(
                [
                    [audio_token_id] * 10 + [2, 3, 4, 5, 0, 0, 0, 0, 0, 0],
                    [audio_token_id] * 10 + [2, 3, 4, 5, 6, 7, 0, 0, 0, 0],
                ],
                dtype="int32",
            ),
            "padding_mask": np.array(
                [
                    [1] * 10 + [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1] * 10 + [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                ],
                dtype="int32",
            ),
        }

    def test_output_shape(self):
        model = Qwen3ASRAudioToText(**self.init_kwargs)
        output = model(self.input_data)
        # 10 audio + 10 text = 20 positions.
        self.assertEqual(output.shape, (2, 20, self.vocab_size))

    def test_serialization(self):
        instance = Qwen3ASRAudioToText(**self.init_kwargs)
        self.run_serialization_test(instance=instance)

    def test_generate(self):
        """Test that generate_step produces tokens beyond the prompt."""
        model = Qwen3ASRAudioToText(**self.init_kwargs)
        model.compile(sampler="greedy")

        # Use preprocessed inputs directly. 10 audio placeholders + 4 real
        # text tokens + 6 padding slots for generation = 20 total.
        audio_token_id = 200
        inputs = {
            "audio_features": np.random.uniform(size=(1, 80, 32)).astype(
                "float32"
            ),
            "token_ids": np.array(
                [[audio_token_id] * 10 + [2, 3, 4, 5, 0, 0, 0, 0, 0, 0]],
                dtype="int32",
            ),
            "padding_mask": np.array(
                [[1] * 10 + [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
                dtype="int32",
            ),
        }
        outputs = model.generate_step(inputs)
        # Should return token_ids and padding_mask.
        self.assertIn("token_ids", outputs)
        self.assertIn("padding_mask", outputs)
        # Token IDs shape should match input.
        self.assertEqual(outputs["token_ids"].shape, (1, 20))
        # Generated tokens should be filled in past the prompt.
        generated = keras.ops.convert_to_numpy(outputs["token_ids"])
        # At least one token after the prompt should be non-zero.
        self.assertTrue(np.any(generated[0, 14:] != 0))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3ASRAudioToText,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
