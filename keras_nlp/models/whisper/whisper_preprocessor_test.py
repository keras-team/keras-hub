# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from keras_nlp.models.whisper.whisper_audio_feature_extractor import (
    WhisperAudioFeatureExtractor,
)
from keras_nlp.models.whisper.whisper_preprocessor import WhisperPreprocessor
from keras_nlp.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_nlp.tests.test_case import TestCase


class WhisperPreprocessorTest(TestCase):
    def setUp(self):
        self.audio_feature_extractor = WhisperAudioFeatureExtractor(
            num_mels=80,
            num_fft_bins=400,
            stride=100,
            sampling_rate=100,
            max_audio_length=5,
        )
        self.vocab = ["air", "Ġair", "plane", "Ġat", "port"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.special_tokens = {
            "<|startoftranscript|>": 9,
            "<|endoftext|>": 10,
            "<|notimestamps|>": 11,
            "<|transcribe|>": 12,
            "<|translate|>": 13,
        }
        self.language_tokens = {
            "<|en|>": 14,
            "<|fr|>": 15,
        }
        self.tokenizer = WhisperTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
            special_tokens=self.special_tokens,
            language_tokens=self.language_tokens,
        )
        self.init_kwargs = {
            "audio_feature_extractor": self.audio_feature_extractor,
            "tokenizer": self.tokenizer,
            "decoder_sequence_length": 12,
            "language": "<|en|>",
            "task": "translate",
        }
        self.input_data = {
            "encoder_audio": np.ones((2, 200)),
            "decoder_text": [" airplane at airport", " airplane at"],
        }

    def test_feature_extractor_basics(self):
        self.run_preprocessor_test(
            cls=WhisperPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            token_id_key="decoder_token_ids",
        )

    def test_sequence_length_override(self):
        input_data = {
            "encoder_audio": np.ones((200,)),
            "decoder_text": " airplane at airport",
        }
        preprocessor = WhisperPreprocessor(**self.init_kwargs)
        x = preprocessor(input_data, decoder_sequence_length=6)
        self.assertAllEqual(x["decoder_token_ids"], [9, 14, 13, 11, 1, 10])
