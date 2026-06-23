import importlib.util
import os
import shutil
import unittest

import numpy as np

_LITERT_TORCH_AVAILABLE = (
    importlib.util.find_spec("litert_torch") is not None
)
_LITERT_LM_BUILDER_AVAILABLE = (
    importlib.util.find_spec("litert_lm_builder") is not None
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
from keras_hub.src.utils.preset_utils import TOKENIZER_ASSET_DIR



@unittest.skipIf(
    not _LITERT_TORCH_AVAILABLE,
    "Requires litert-torch.",
)
@unittest.skipIf(
    not _LITERT_LM_BUILDER_AVAILABLE,
    "Requires litert-lm-builder.",
)
class TestGemma3nLiteRTLmExport(TestCase):
    def setUp(self):
        self.tokenizer = MockGemma3nTokenizer()
        self.proto = os.path.join(
            self.get_test_data_dir(), "gemma_test_vocab.spm"
        )
        self.tokenizer.file_assets = ["vocabulary.spm"]

        def _save_to_preset(preset_dir):
            asset_dir = os.path.join(preset_dir, TOKENIZER_ASSET_DIR)
            os.makedirs(asset_dir, exist_ok=True)
            shutil.copy(self.proto, os.path.join(asset_dir, "vocabulary.spm"))

        self.tokenizer.save_to_preset = _save_to_preset

    def _build_text_only_model(self):
        preprocessor = Gemma3nCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=None,
            audio_converter=None,
            sequence_length=20,
            max_images_per_prompt=0,
            num_vision_tokens_per_image=0,
            max_audios_per_prompt=0,
            num_audio_tokens_per_audio=0,
        )
        backbone = Gemma3nBackbone(
            text_vocab_size=self.tokenizer.vocabulary_size(),
            text_hidden_size=8,
            num_hidden_layers=1,
            pad_token_id=0,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            intermediate_size=[16],
            hidden_activation="gelu_approximate",
            layer_types=["full_attention"],
            sliding_window=4,
            rope_theta=10000.0,
            max_position_embeddings=20,
            vocab_size_per_layer_input=10,
            hidden_size_per_layer_input=2,
            altup_num_inputs=2,
            laurel_rank=1,
        )
        return Gemma3nCausalLM(preprocessor=preprocessor, backbone=backbone)

    def _build_vision_text_model(self):
        image_converter = Gemma3nImageConverter(image_size=(16, 16))
        preprocessor = Gemma3nCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            audio_converter=None,
            sequence_length=20,
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
            max_audios_per_prompt=0,
            num_audio_tokens_per_audio=0,
        )
        vision_arch_def = [["er_r1_k3_s1_e1_c8"]]
        stackwise_params = convert_arch_def_to_stackwise(vision_arch_def)
        vision_encoder = MobileNetV5Backbone(
            **stackwise_params,
            num_features=4,
            image_shape=(16, 16, 3),
            use_msfa=False,
        )
        backbone = Gemma3nBackbone(
            text_vocab_size=self.tokenizer.vocabulary_size(),
            text_hidden_size=8,
            num_hidden_layers=1,
            pad_token_id=0,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            intermediate_size=[16],
            hidden_activation="gelu_approximate",
            layer_types=["full_attention"],
            sliding_window=4,
            rope_theta=10000.0,
            max_position_embeddings=20,
            vocab_size_per_layer_input=10,
            hidden_size_per_layer_input=2,
            altup_num_inputs=2,
            laurel_rank=1,
            vision_encoder_config=vision_encoder.get_config(),
            vision_hidden_size=8,
        )
        return Gemma3nCausalLM(preprocessor=preprocessor, backbone=backbone)

    def _set_random_weights(self, model):
        rng = np.random.default_rng(42)
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        model.set_weights(weights)

    def test_gemma3n_litertlm_export_baked_in(self):
        model = self._build_text_only_model()
        self._set_random_weights(model)
        input_data = np.array([[1, 2, 3, 4]], dtype=np.int32)
        self.run_litertlm_export_test(
            model=model,
            input_data=input_data,
            prefill_seq_len=4,
            verify_model_type="gemma3n",
            verify_numerics=False,
            verify_generation=False,
        )

    def test_gemma3n_litertlm_export_baked_in_vision(self):
        model = self._build_vision_text_model()
        self._set_random_weights(model)
        # `input_data=None` skips the text-only numeric parity check; the
        # exporter builds its own vision sample inputs from the vision config.
        self.run_litertlm_export_test(
            model=model,
            input_data=None,
            prefill_seq_len=20,
            verify_model_type="gemma3n",
            verify_numerics=False,
            verify_generation=False,
        )

    @unittest.skip(
        "Gemma3n separate vision encoder export is not yet supported. "
        "MobileNetV5 does not expose a single projected vision dimension, "
        "and Gemma3n applies reshape / sqrt-scaling / embed_vision inside "
        "the backbone after the encoder."
    )
    def test_gemma3n_litertlm_export_separate_vision_encoder(self):
        model = self._build_vision_text_model()
        self._set_random_weights(model)
        self.run_litertlm_export_test(
            model=model,
            input_data=None,
            prefill_seq_len=20,
            verify_model_type="gemma3n",
            verify_numerics=False,
            verify_generation=False,
            separate_vision_encoder=True,
        )
