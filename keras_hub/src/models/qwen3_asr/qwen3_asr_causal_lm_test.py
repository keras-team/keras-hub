import numpy as np
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRAudioEncoder,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm import Qwen3ASRCausalLM
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRCausalLMTest(TestCase):
    def setUp(self):
        self.vocabulary_size = 10
        self.audio_encoder = Qwen3ASRAudioEncoder(
            d_model=8,
            encoder_layers=1,
            encoder_attention_heads=1,
            encoder_ffn_dim=16,
            downsample_hidden_size=4,
            num_mel_bins=8,
            n_window=5,
            max_position_embeddings=100,
            output_dim=16,
        )
        self.backbone = Qwen3ASRBackbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=16,
            head_dim=2,
            intermediate_dim=8,
            audio_encoder=self.audio_encoder,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": None,
        }

        seq_len = 10
        chunk_len = 10
        num_chunks = 3
        total_mels = num_chunks * chunk_len

        self.input_data = {
            "token_ids": np.ones((2, seq_len), dtype="int32"),
            "padding_mask": np.ones((2, seq_len), dtype="int32"),
            "audio_mel": np.ones((2, total_mels, 8), dtype="float32"),
            "audio_mel_mask": np.ones((2, total_mels), dtype="int32"),
            "audio_indices": np.array(
                [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype="int32"
            ),
        }

    def test_causal_lm_basics(self):
        model = Qwen3ASRCausalLM(**self.init_kwargs)
        output = model(self.input_data)
        self.assertEqual(ops.shape(output), (2, 10, self.vocabulary_size))

    def test_generate_multimodal(self):
        model = Qwen3ASRCausalLM(**self.init_kwargs)
        outputs = model.generate(self.input_data, stop_token_ids=None)

        self.assertTrue("token_ids" in outputs)
        self.assertTrue("padding_mask" in outputs)

    def test_score_multimodal(self):
        model = Qwen3ASRCausalLM(**self.init_kwargs)
        audio_embeddings = self.audio_encoder(
            self.input_data["audio_mel"], self.input_data["audio_mel_mask"]
        )

        logits = model.score(
            token_ids=self.input_data["token_ids"],
            padding_mask=self.input_data["padding_mask"],
            audio_embeddings=audio_embeddings,
            audio_indices=self.input_data["audio_indices"],
            scoring_mode="logits",
        )
        self.assertEqual(ops.shape(logits), (2, 10, self.vocabulary_size))
