import numpy as np
import pytest

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm import Qwen3ASRCausalLM
from keras_hub.src.models.qwen3_asr.qwen3_asr_preprocessor import (
    Qwen3ASRPreprocessor,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRCausalLMTest(TestCase):
    def setUp(self):
        # Minimal BPE vocab
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = list("abcdefghijklmnopqrstuvwxyzĠĊ")
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])

        # Audio special tokens
        self.vocab += [
            "<|audio_pad|>",
            "<|audio_info|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|audio_start|>",
            "<|audio_end|>",
        ]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])

        self.audio_token_id = self.vocab["<|audio_pad|>"]

        self.tokenizer = Qwen3ASRTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )

        self.audio_converter = Qwen3ASRAudioConverter(
            num_mels=128,
            sampling_rate=16000,
            max_audio_length=1.05,
            n_window=50,
        )

        self.preprocessor = Qwen3ASRPreprocessor(
            tokenizer=self.tokenizer,
            audio_converter=self.audio_converter,
            sequence_length=40,
        )

        self.vocabulary_size = self.tokenizer.vocabulary_size()

        self.backbone = Qwen3ASRBackbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            hidden_dim=32,
            intermediate_dim=64,
            audio_num_mel_bins=128,
            audio_num_layers=2,
            audio_num_attention_heads=4,
            audio_intermediate_dim=64,
            audio_d_model=16,
            audio_n_window=50,
            audio_downsample_hidden_size=8,
            audio_max_position_embeddings=13,
            audio_token_id=self.audio_token_id,
        )

        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }

        self.train_data = (
            {
                "prompts": [" airplane at airport", " airplane"],
                "audio": np.random.uniform(size=(2, 16000)),
                "responses": [" indeed", " yes"],
            },
        )
        # Preprocess to get input_data for saving test
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Qwen3ASRCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 40, self.vocabulary_size),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3ASRCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_generate(self):
        causal_lm = Qwen3ASRCausalLM(**self.init_kwargs)

        audio_input = np.random.uniform(size=(16000,))
        inputs = {
            "prompts": " airplane",
            "audio": audio_input,
        }

        output = causal_lm.generate(inputs)
        self.assertIsInstance(output, str)

        # Test batch generate
        batch_inputs = {
            "prompts": [" airplane", " airplane"],
            "audio": np.random.uniform(size=(2, 16000)),
        }
        batch_output = causal_lm.generate(batch_inputs)
        self.assertEqual(len(batch_output), 2)
        self.assertIsInstance(batch_output[0], str)
