import copy
from unittest.mock import patch

import keras
import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops
from packaging import version

from keras_hub.src.models.gemma3n.gemma3n_audio_converter import (
    Gemma3nAudioConverter,
)
from keras_hub.src.models.gemma3n.gemma3n_audio_encoder import (
    Gemma3nAudioEncoder,
)
from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
from keras_hub.src.models.gemma3n.gemma3n_causal_lm import Gemma3nCausalLM
from keras_hub.src.models.gemma3n.gemma3n_causal_lm_preprocessor import (
    Gemma3nCausalLMPreprocessor,
)
from keras_hub.src.models.gemma3n.gemma3n_image_converter import (
    Gemma3nImageConverter,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import (
    convert_arch_def_to_stackwise,
)
from keras_hub.src.tests.mocks.mock_gemma3n_tokenizer import (
    MockGemma3nTokenizer,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import fused_attention_op_available
from keras_hub.src.utils.keras_utils import gpu_supports_fused_attention_op
from keras_hub.src.utils.keras_utils import running_on_gpu


@pytest.mark.skipif(
    version.parse(keras.__version__) > version.parse("3.8.0"),
    reason=("Some facets of Gemma3nCausalLM are unsupported in keras > 3.8.0"),
)
class Gemma3nCausalLMTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.tokenizer = MockGemma3nTokenizer()

        # === Vision Encoder ===
        vision_arch_def = [["er_r1_k3_s1_e1_c8"]]
        stackwise_params = convert_arch_def_to_stackwise(vision_arch_def)
        vision_encoder = MobileNetV5Backbone(
            **stackwise_params,
            num_features=4,
            image_shape=(16, 16, 3),
            use_msfa=False,
        )

        # === Audio Encoder ===
        audio_encoder = Gemma3nAudioEncoder(
            hidden_size=4,
            input_feat_size=16,
            sscp_conv_channel_size=[2, 4],
            sscp_conv_kernel_size=[(1, 1), (1, 1)],
            sscp_conv_stride_size=[(2, 2), (2, 2)],
            sscp_conv_group_norm_eps=1e-5,
            conf_num_hidden_layers=1,
            rms_norm_eps=1e-6,
            gradient_clipping=1.0,
            conf_residual_weight=0.5,
            conf_num_attention_heads=1,
            conf_attention_chunk_size=2,
            conf_attention_context_right=1,
            conf_attention_context_left=1,
            conf_attention_logit_cap=50.0,
            conf_conv_kernel_size=3,
            conf_reduction_factor=1,
        )

        # === Text-Only ===
        self.text_preprocessor = Gemma3nCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=None,
            audio_converter=None,
            sequence_length=20,
            max_images_per_prompt=0,
            num_vision_tokens_per_image=0,
            max_audios_per_prompt=0,
            num_audio_tokens_per_audio=0,
        )
        text_backbone_init_kwargs = {
            "text_vocab_size": self.text_preprocessor.tokenizer.vocabulary_size(),  # noqa: E501
            "text_hidden_size": 4,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": [8],
            "hidden_activation": "gelu_approximate",
            "layer_types": ["full_attention"],
            "sliding_window": 4,
            "rope_theta": 10000.0,
            "max_position_embeddings": 20,
            "vocab_size_per_layer_input": 10,
            "hidden_size_per_layer_input": 2,
            "altup_num_inputs": 2,
            "laurel_rank": 1,
        }
        self.text_backbone = Gemma3nBackbone(**text_backbone_init_kwargs)
        self.text_init_kwargs = {
            "preprocessor": self.text_preprocessor,
            "backbone": self.text_backbone,
        }
        self.text_train_data = (
            {
                "prompts": ["the quick brown fox", "the quick brown fox"],
                "responses": ["the earth is round", "the earth is round"],
            },
        )
        self.text_input_data = self.text_preprocessor(*self.text_train_data)[0]

        # === Vision + Text ===
        self.image_converter = Gemma3nImageConverter(
            image_size=(16, 16),
        )
        self.vision_preprocessor = Gemma3nCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            audio_converter=None,
            sequence_length=20,
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
            max_audios_per_prompt=0,
            num_audio_tokens_per_audio=0,
        )
        vision_backbone_init_kwargs = copy.deepcopy(text_backbone_init_kwargs)
        vision_backbone_init_kwargs["vision_encoder_config"] = (
            vision_encoder.get_config()
        )
        vision_backbone_init_kwargs["vision_hidden_size"] = 8
        self.vision_backbone = Gemma3nBackbone(**vision_backbone_init_kwargs)
        self.vision_init_kwargs = {
            "preprocessor": self.vision_preprocessor,
            "backbone": self.vision_backbone,
        }
        self.vision_train_data = (
            {
                "prompts": [
                    "the quick brown fox <start_of_image>",
                    "the quick brown fox",
                ],
                "responses": ["the earth is round", "the earth is round"],
                "images": [np.ones((8, 8, 3)), np.ones((8, 8, 3))],
            },
        )
        self.vision_input_data = self.vision_preprocessor(
            *self.vision_train_data
        )[0]

        # === Audio + Text ===
        self.audio_converter = Gemma3nAudioConverter(
            feature_size=16,
            sampling_rate=16000,
        )
        self.audio_preprocessor = Gemma3nCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=None,
            audio_converter=self.audio_converter,
            sequence_length=20,
            max_images_per_prompt=0,
            num_vision_tokens_per_image=0,
            max_audios_per_prompt=2,
            num_audio_tokens_per_audio=3,
        )
        audio_backbone_init_kwargs = copy.deepcopy(text_backbone_init_kwargs)
        audio_backbone_init_kwargs["audio_encoder_config"] = (
            audio_encoder.get_config()
        )
        audio_backbone_init_kwargs["audio_hidden_size"] = 4
        self.audio_backbone = Gemma3nBackbone(**audio_backbone_init_kwargs)
        self.audio_init_kwargs = {
            "preprocessor": self.audio_preprocessor,
            "backbone": self.audio_backbone,
        }
        self.audio_train_data = (
            {
                "prompts": [
                    "the quick <start_of_audio>",
                    "the quick brown fox",
                ],
                "responses": ["brown", "the earth is round"],
                "audios": [np.ones((16000,)), np.ones((16000,))],
            },
        )
        self.audio_input_data = self.audio_preprocessor(*self.audio_train_data)[
            0
        ]

        # === Multimodal (Vision + Audio + Text) ===
        self.multimodal_preprocessor = Gemma3nCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            audio_converter=self.audio_converter,
            sequence_length=30,
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
            max_audios_per_prompt=2,
            num_audio_tokens_per_audio=3,
        )
        multimodal_backbone_init_kwargs = copy.deepcopy(
            text_backbone_init_kwargs
        )
        multimodal_backbone_init_kwargs["vision_encoder_config"] = (
            vision_encoder.get_config()
        )
        multimodal_backbone_init_kwargs["vision_hidden_size"] = 8
        multimodal_backbone_init_kwargs["audio_encoder_config"] = (
            audio_encoder.get_config()
        )
        multimodal_backbone_init_kwargs["audio_hidden_size"] = 4
        multimodal_backbone_init_kwargs["max_position_embeddings"] = 30
        self.multimodal_backbone = Gemma3nBackbone(
            **multimodal_backbone_init_kwargs
        )
        self.multimodal_init_kwargs = {
            "preprocessor": self.multimodal_preprocessor,
            "backbone": self.multimodal_backbone,
        }
        self.multimodal_train_data = (
            {
                "prompts": [
                    "image <start_of_image> audio <start_of_audio>",
                    "the quick brown fox",
                ],
                "responses": ["test", "the earth is round"],
                "images": [np.ones((8, 8, 3)), np.ones((8, 8, 3))],
                "audios": [np.ones((16000,)), np.ones((16000,))],
            },
        )
        self.multimodal_input_data = self.multimodal_preprocessor(
            *self.multimodal_train_data
        )[0]

    @parameterized.named_parameters(
        ("text_only", "text_only"),
        ("vision_text", "vision_text"),
        ("audio_text", "audio_text"),
        ("multimodal", "multimodal"),
    )
    def test_causal_lm_basics(self, modality_type):
        if modality_type == "text_only":
            init_kwargs = self.text_init_kwargs
            train_data = self.text_train_data
            expected_vocab_size = self.tokenizer.vocabulary_size()
        elif modality_type == "vision_text":
            init_kwargs = self.vision_init_kwargs
            train_data = self.vision_train_data
            expected_vocab_size = self.tokenizer.vocabulary_size()
        elif modality_type == "audio_text":
            init_kwargs = self.audio_init_kwargs
            train_data = self.audio_train_data
            expected_vocab_size = self.tokenizer.vocabulary_size()
        else:  # multimodal
            init_kwargs = self.multimodal_init_kwargs
            train_data = self.multimodal_train_data
            expected_vocab_size = self.tokenizer.vocabulary_size()
        self.run_task_test(
            cls=Gemma3nCausalLM,
            init_kwargs=init_kwargs,
            train_data=train_data,
            expected_output_shape=(
                2,
                20 if modality_type != "multimodal" else 30,
                expected_vocab_size,
            ),
        )

    def test_text_flash_attention_call(self):
        if (
            keras.config.backend() != "jax"
            or not fused_attention_op_available()
            or not gpu_supports_fused_attention_op()
        ):
            self.skipTest("`flash_attention` testing requires the JAX backend.")

        with patch("keras.src.backend.nn.dot_product_attention") as mock_func:
            causal_lm = Gemma3nCausalLM(**self.text_init_kwargs)
            causal_lm.generate("the quick brown fox")
            if running_on_gpu():
                mock_func.assert_called()
            else:
                mock_func.assert_not_called()

    def test_text_early_stopping(self):
        causal_lm = Gemma3nCausalLM(**self.text_init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.text_preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the quick brown fox", "the quick"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_text_multitoken_stopping(self):
        causal_lm = Gemma3nCausalLM(**self.text_init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.text_preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the quick brown fox", "the quick"]
            output = causal_lm.generate(prompt, stop_token_ids=(3,))
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_text_generate_compilation(self):
        causal_lm = Gemma3nCausalLM(**self.text_init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate("the quick brown fox")
        first_fn = causal_lm.generate_function
        causal_lm.generate("the quick brown fox")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    def test_vision_generate(self):
        causal_lm = Gemma3nCausalLM(**self.vision_init_kwargs)
        inputs = {
            "prompts": "this is a lily <start_of_image>",
            "images": np.ones((8, 8, 3), dtype="float32"),
        }
        output = causal_lm.generate(inputs)
        self.assertIsInstance(output, str)

    def test_audio_generate(self):
        causal_lm = Gemma3nCausalLM(**self.audio_init_kwargs)
        inputs = {
            "prompts": "transcribe this <start_of_audio>",
            "audios": np.ones((16000,), dtype="float32"),
        }
        output = causal_lm.generate(inputs)
        self.assertIsInstance(output, str)

    def test_multimodal_generate(self):
        causal_lm = Gemma3nCausalLM(**self.multimodal_init_kwargs)
        inputs = {
            "prompts": "image <start_of_image> audio <start_of_audio>",
            "images": np.ones((8, 8, 3), dtype="float32"),
            "audios": np.ones((16000,), dtype="float32"),
        }
        output = causal_lm.generate(inputs)
        self.assertIsInstance(output, str)
