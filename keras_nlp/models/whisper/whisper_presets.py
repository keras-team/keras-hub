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
MULTILINGUAL_SPECIAL_TOKENS = {
    "<|startoftranscript|>": 50258,
    "<|endoftext|>": 50257,
    "<|notimestamps|>": 50363,
    "<|translate|>": 50359,
    "<|transcribe|>": 50358,
}

ENGLISH_SPECIAL_TOKENS = {
    "<|startoftranscript|>": 50257,
    "<|endoftext|>": 50256,
    "<|notimestamps|>": 50362,
    "<|translate|>": 50358,
    "<|transcribe|>": 50357,
}

AUDIO_FEATURE_EXTRACTOR_CONFIG = {
    "num_mels": 80,
    "num_fft_bins": 400,
    "stride": 160,
    "sampling_rate": 16000,
    "max_audio_length": 30,
}

LANGUAGE_TOKENS = {
    "<|af|>": 50327,
    "<|am|>": 50334,
    "<|ar|>": 50272,
    "<|as|>": 50350,
    "<|az|>": 50304,
    "<|ba|>": 50355,
    "<|be|>": 50330,
    "<|bg|>": 50292,
    "<|bn|>": 50302,
    "<|bo|>": 50347,
    "<|br|>": 50309,
    "<|bs|>": 50315,
    "<|ca|>": 50270,
    "<|cs|>": 50283,
    "<|cy|>": 50297,
    "<|da|>": 50285,
    "<|de|>": 50261,
    "<|el|>": 50281,
    "<|en|>": 50259,
    "<|es|>": 50262,
    "<|et|>": 50307,
    "<|eu|>": 50310,
    "<|fa|>": 50300,
    "<|fi|>": 50277,
    "<|fo|>": 50338,
    "<|fr|>": 50265,
    "<|gl|>": 50319,
    "<|gu|>": 50333,
    "<|haw|>": 50352,
    "<|ha|>": 50354,
    "<|he|>": 50279,
    "<|hi|>": 50276,
    "<|hr|>": 50291,
    "<|ht|>": 50339,
    "<|hu|>": 50286,
    "<|hy|>": 50312,
    "<|id|>": 50275,
    "<|is|>": 50311,
    "<|it|>": 50274,
    "<|ja|>": 50266,
    "<|jw|>": 50356,
    "<|ka|>": 50329,
    "<|kk|>": 50316,
    "<|km|>": 50323,
    "<|kn|>": 50306,
    "<|ko|>": 50264,
    "<|la|>": 50294,
    "<|lb|>": 50345,
    "<|ln|>": 50353,
    "<|lo|>": 50336,
    "<|lt|>": 50293,
    "<|lv|>": 50301,
    "<|mg|>": 50349,
    "<|mi|>": 50295,
    "<|mk|>": 50308,
    "<|ml|>": 50296,
    "<|mn|>": 50314,
    "<|mr|>": 50320,
    "<|ms|>": 50282,
    "<|mt|>": 50343,
    "<|my|>": 50346,
    "<|ne|>": 50313,
    "<|nl|>": 50271,
    "<|nn|>": 50342,
    "<|no|>": 50288,
    "<|oc|>": 50328,
    "<|pa|>": 50321,
    "<|pl|>": 50269,
    "<|ps|>": 50340,
    "<|pt|>": 50267,
    "<|ro|>": 50284,
    "<|ru|>": 50263,
    "<|sa|>": 50344,
    "<|sd|>": 50332,
    "<|si|>": 50322,
    "<|sk|>": 50298,
    "<|sl|>": 50305,
    "<|sn|>": 50324,
    "<|so|>": 50326,
    "<|sq|>": 50317,
    "<|sr|>": 50303,
    "<|su|>": 50357,
    "<|sv|>": 50273,
    "<|sw|>": 50318,
    "<|ta|>": 50287,
    "<|te|>": 50299,
    "<|tg|>": 50331,
    "<|th|>": 50289,
    "<|tk|>": 50341,
    "<|tl|>": 50348,
    "<|tr|>": 50268,
    "<|tt|>": 50351,
    "<|uk|>": 50280,
    "<|ur|>": 50290,
    "<|uz|>": 50337,
    "<|vi|>": 50278,
    "<|yi|>": 50335,
    "<|yo|>": 50325,
    "<|zh|>": 50260,
}

# Metadata for loading pretrained model weights.
backbone_presets = {
    "whisper_tiny_en": {
        "metadata": {
            "description": (
                "4-layer Whisper model. Trained on 438,000 hours of labelled "
                "English speech data."
            ),
            "params": 37184256,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51864,
            "num_layers": 4,
            "num_heads": 6,
            "hidden_dim": 384,
            "intermediate_dim": 1536,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": ENGLISH_SPECIAL_TOKENS,
            "language_tokens": None,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_tiny_en/v1/model.h5",
        "weights_hash": "3dc3768ac48ec90b1029fbf52ffbacc7",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_tiny_en/v1/vocab.json",
        "vocabulary_hash": "22377f841debacb023848b3468ea3281",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_tiny_en/v1/merges.txt",
        "merges_hash": "093ecf3f30371012f2e96fcfb10ea6ab",
    },
    "whisper_base_en": {
        "metadata": {
            "description": (
                "6-layer Whisper model. Trained on 438,000 hours of labelled "
                "English speech data."
            ),
            "params": 124439808,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51864,
            "num_layers": 6,
            "num_heads": 8,
            "hidden_dim": 512,
            "intermediate_dim": 2048,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": ENGLISH_SPECIAL_TOKENS,
            "language_tokens": None,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_en/v1/model.h5",
        "weights_hash": "799d3c143993d42f7446bafbc0f46d7d",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_en/v1/vocab.json",
        "vocabulary_hash": "22377f841debacb023848b3468ea3281",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_en/v1/merges.txt",
        "merges_hash": "093ecf3f30371012f2e96fcfb10ea6ab",
    },
    "whisper_small_en": {
        "metadata": {
            "description": (
                "12-layer Whisper model. Trained on 438,000 hours of labelled "
                "English speech data."
            ),
            "params": 241734144,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51864,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": ENGLISH_SPECIAL_TOKENS,
            "language_tokens": None,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_en/v1/model.h5",
        "weights_hash": "b75a89225e20019d85ff5f1c362f8a49",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_en/v1/vocab.json",
        "vocabulary_hash": "22377f841debacb023848b3468ea3281",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_en/v1/merges.txt",
        "merges_hash": "093ecf3f30371012f2e96fcfb10ea6ab",
    },
    "whisper_medium_en": {
        "metadata": {
            "description": (
                "24-layer Whisper model. Trained on 438,000 hours of labelled "
                "English speech data."
            ),
            "params": 763856896,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51864,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": ENGLISH_SPECIAL_TOKENS,
            "language_tokens": None,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_medium_en/v1/model.h5",
        "weights_hash": "107184882d1cc65926815e4cc50dc5f3",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_medium_en/v1/vocab.json",
        "vocabulary_hash": "22377f841debacb023848b3468ea3281",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_medium_en/v1/merges.txt",
        "merges_hash": "093ecf3f30371012f2e96fcfb10ea6ab",
    },
    "whisper_tiny_multi": {
        "metadata": {
            "description": (
                "4-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 37760640,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51865,
            "num_layers": 4,
            "num_heads": 6,
            "hidden_dim": 384,
            "intermediate_dim": 1536,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": MULTILINGUAL_SPECIAL_TOKENS,
            "language_tokens": LANGUAGE_TOKENS,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_tiny_multi/v1/model.h5",
        "weights_hash": "b1279a81001ad5eb35970d1aea706396",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_tiny_multi/v1/vocab.json",
        "vocabulary_hash": "1b87ed3e3ecd9ccfdca74e64cbe81d68",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_tiny_multi/v1/merges.txt",
        "merges_hash": "c7f01d4100f6211417988889bf35ccd8",
    },
    "whisper_base_multi": {
        "metadata": {
            "description": (
                "6-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 72593920,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51865,
            "num_layers": 6,
            "num_heads": 8,
            "hidden_dim": 512,
            "intermediate_dim": 2048,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": MULTILINGUAL_SPECIAL_TOKENS,
            "language_tokens": LANGUAGE_TOKENS,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_multi/v1/model.h5",
        "weights_hash": "5208396e2d5efac43114a4a3d4f583ab",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_multi/v1/vocab.json",
        "vocabulary_hash": "1b87ed3e3ecd9ccfdca74e64cbe81d68",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_multi/v1/merges.txt",
        "merges_hash": "c7f01d4100f6211417988889bf35ccd8",
    },
    "whisper_small_multi": {
        "metadata": {
            "description": (
                "12-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 241734912,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51865,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": MULTILINGUAL_SPECIAL_TOKENS,
            "language_tokens": LANGUAGE_TOKENS,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_multi/v1/model.h5",
        "weights_hash": "c90c6a895e522056b77b924b6e907ed8",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_multi/v1/vocab.json",
        "vocabulary_hash": "1b87ed3e3ecd9ccfdca74e64cbe81d68",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_base_multi/v1/merges.txt",
        "merges_hash": "c7f01d4100f6211417988889bf35ccd8",
    },
    "whisper_medium_multi": {
        "metadata": {
            "description": (
                "24-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 763857920,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51865,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": MULTILINGUAL_SPECIAL_TOKENS,
            "language_tokens": LANGUAGE_TOKENS,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_medium_multi/v1/model.h5",
        "weights_hash": "6f993f732fe397e9c5e3a96a9505a3a9",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_medium_multi/v1/vocab.json",
        "vocabulary_hash": "1b87ed3e3ecd9ccfdca74e64cbe81d68",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_medium_multi/v1/merges.txt",
        "merges_hash": "c7f01d4100f6211417988889bf35ccd8",
    },
    "whisper_large_multi": {
        "metadata": {
            "description": (
                "32-layer Whisper model. Trained on 680,000 hours of labelled "
                "multilingual speech data."
            ),
            "params": 1543304960,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51865,
            "num_layers": 32,
            "num_heads": 20,
            "hidden_dim": 1280,
            "intermediate_dim": 5120,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": MULTILINGUAL_SPECIAL_TOKENS,
            "language_tokens": LANGUAGE_TOKENS,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_large_multi/v1/model.h5",
        "weights_hash": "ccab1c93c5739007868ae73fe025806d",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_large_multi/v1/vocab.json",
        "vocabulary_hash": "1b87ed3e3ecd9ccfdca74e64cbe81d68",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_large_multi/v1/merges.txt",
        "merges_hash": "c7f01d4100f6211417988889bf35ccd8",
    },
    "whisper_large_multi_v2": {
        "metadata": {
            "description": (
                "32-layer Whisper model. Trained for 2.5 epochs on 680,000  "
                "hours of labelled multilingual speech data. An improved "
                "of `whisper_large_multi`."
            ),
            "params": 1543304960,
            "official_name": "Whisper",
            "path": "whisper",
            "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        },
        "config": {
            "vocabulary_size": 51865,
            "num_layers": 32,
            "num_heads": 20,
            "hidden_dim": 1280,
            "intermediate_dim": 5120,
            "num_mels": 80,
            "dropout": 0.0,
            "max_encoder_sequence_length": 3000,
            "max_decoder_sequence_length": 448,
        },
        "audio_feature_extractor_config": AUDIO_FEATURE_EXTRACTOR_CONFIG,
        "preprocessor_config": {
            "special_tokens": MULTILINGUAL_SPECIAL_TOKENS,
            "language_tokens": LANGUAGE_TOKENS,
        },
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/whisper_large_multi_v2/v1/model.h5",
        "weights_hash": "ca157162ec9c3329a659388528a3af88",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/whisper_large_multi_v2/v1/vocab.json",
        "vocabulary_hash": "1b87ed3e3ecd9ccfdca74e64cbe81d68",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/whisper_large_multi_v2/v1/merges.txt",
        "merges_hash": "c7f01d4100f6211417988889bf35ccd8",
    },
}
