import tempfile

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm import Qwen3ASRCausalLM
from keras_hub.src.tests.test_case import TestCase


class TestQwen3ASRConverter(TestCase):
    @pytest.mark.extra_large
    def test_backbone_from_hf_preset(self):
        model = Qwen3ASRBackbone.from_preset(
            "hf://Qwen/Qwen3-ASR-0.6B-hf",
            load_weights=False,
        )
        self.assertEqual(model.hidden_dim, 1024)
        self.assertEqual(model.num_layers, 28)
        self.assertEqual(model.audio_encoder.num_layers, 18)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://Qwen/Qwen3-ASR-0.6B-hf",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3ASRBackbone)

    @pytest.mark.extra_large
    def test_causal_lm_from_hf_preset(self):
        model = Qwen3ASRCausalLM.from_preset(
            "hf://Qwen/Qwen3-ASR-0.6B-hf",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3ASRCausalLM)
        self.assertEqual(model.backbone.hidden_dim, 1024)

    def test_numerical_parity(self):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        # Tiny configs
        vocab_size = 100
        audio_token_id = 99

        text_cfg = transformers.Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            tie_word_embeddings=True,
            rms_norm_eps=1e-5,
        )
        audio_cfg = transformers.Qwen3ASREncoderConfig(
            num_mel_bins=20,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            d_model=16,
            n_window=50,
            max_position_embeddings=13,
            downsample_hidden_size=4,
            output_dim=16,
        )
        cfg = transformers.Qwen3ASRConfig(
            text_config=text_cfg,
            audio_config=audio_cfg,
            audio_token_id=audio_token_id,
            timestamp_token_id=98,
            pad_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=True,
        )

        torch.manual_seed(0)
        hf_model = transformers.Qwen3ASRForConditionalGeneration(cfg).eval()

        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            keras_model = Qwen3ASRBackbone.from_preset(preset_dir)

        # Inputs
        batch_size = 1
        audio_len = 100  # 1 chunk
        audio_token_len = 13  # output of audio encoder for 1 chunk

        np.random.seed(0)
        audio_mel_keras = np.random.rand(batch_size, audio_len, 20).astype(
            "float32"
        )
        audio_mel_hf = np.transpose(audio_mel_keras, (0, 2, 1))
        audio_mask = np.ones((batch_size, audio_len), dtype="int32")

        # 1 start token, 13 audio tokens, 1 end token
        token_ids = np.array(
            [[1] + [audio_token_id] * audio_token_len + [2]], dtype="int32"
        )
        padding_mask = np.ones_like(token_ids)

        # Keras forward
        keras_inputs = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "audio_mel": audio_mel_keras,
            "audio_mask": audio_mask,
        }
        keras_out = ops.convert_to_numpy(keras_model(keras_inputs))

        # HF forward
        with torch.no_grad():
            hf_out = (
                hf_model.model(
                    input_ids=torch.tensor(token_ids),
                    attention_mask=torch.tensor(padding_mask),
                    input_features=torch.tensor(audio_mel_hf),
                    input_features_mask=torch.tensor(audio_mask),
                )
                .last_hidden_state.detach()
                .cpu()
                .numpy()
            )

        self.assertEqual(keras_out.shape, hf_out.shape)
        self.assertAllClose(keras_out, hf_out, atol=1e-3, rtol=1e-3)
