import importlib.util
import os
import unittest

import numpy as np
import tensorflow as tf

_LITERT_TORCH_AVAILABLE = (
    importlib.util.find_spec("litert_torch") is not None
)
_LITERT_LM_BUILDER_AVAILABLE = (
    importlib.util.find_spec("litert_lm_builder") is not None
)

from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM
from keras_hub.src.models.gemma4.gemma4_causal_lm_preprocessor import (
    Gemma4CausalLMPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


@unittest.skipIf(
    not _LITERT_TORCH_AVAILABLE,
    "LiteRT-LM export requires `litert-torch`. "
    "Install it with: pip install litert-torch",
)
@unittest.skipIf(
    not _LITERT_LM_BUILDER_AVAILABLE,
    "LiteRT-LM export requires `litert-lm-builder`. "
    "Install it with: pip install litert-lm-builder",
)
class TestGemma4LiteRTLmExport(TestCase):
    def _build_tiny_gemma4_multimodal_model(self):
        """Build a tiny multimodal Gemma4CausalLM for LiteRT-LM export tests."""
        tokenizer = MockGemma4Tokenizer()

        # The exporter validates that the tokenizer exposes a SentencePiece
        # asset.  The mock tokenizer does not have a real vocab file, so we
        # point it to the test vocab and monkey-patch the preset save path.
        proto = os.path.join(self.get_test_data_dir(), "gemma_test_vocab.spm")
        tokenizer.file_assets = ["vocabulary.spm"]

        def _save_to_preset(preset_dir):
            import shutil

            from keras_hub.src.utils.preset_utils import TOKENIZER_ASSET_DIR

            asset_dir = os.path.join(preset_dir, TOKENIZER_ASSET_DIR)
            os.makedirs(asset_dir, exist_ok=True)
            shutil.copy(proto, os.path.join(asset_dir, "vocabulary.spm"))

        tokenizer.save_to_preset = _save_to_preset

        image_converter = Gemma4ImageConverter(
            image_size=(16, 16), patch_size=4
        )
        preprocessor = Gemma4CausalLMPreprocessor(
            image_converter=image_converter,
            tokenizer=tokenizer,
            sequence_length=20,
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
            # Disable dummy audio tensors so the backbone inputs stay in sync
            # with the preprocessor outputs (this test exercises vision only).
            audio_input_feat_size=0,
        )

        vision_encoder = Gemma4VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone = Gemma4Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vision_encoder=vision_encoder,
        )
        model = Gemma4CausalLM(preprocessor=preprocessor, backbone=backbone)

        # Preprocessed vision + text input used to build the model and to
        # validate the exported signatures.
        train_data = {
            "prompts": tf.constant(["the quick brown fox <|image|>"]),
            "responses": tf.constant(["the earth is round"]),
            "pixel_values": tf.constant(
                np.ones([1, 2, 16, 3 * 4 * 4], dtype="float32")
            ),
            "pixel_position_ids": tf.constant(
                np.ones([1, 2, 16, 2], dtype="int32")
            ),
        }
        input_data = preprocessor(train_data)[0]

        # Build the model and set deterministic random weights.
        _ = model(input_data)
        rng = np.random.default_rng(42)
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        model.set_weights(weights)

        return model, input_data

    def test_gemma4_litertlm_export_baked_in(self):
        """Export a tiny Gemma4 with the vision encoder baked into prefill."""
        import keras

        if keras.config.backend() != "torch":
            self.skipTest("LiteRT-LM export requires the PyTorch backend.")

        model, input_data = self._build_tiny_gemma4_multimodal_model()

        self.run_litertlm_export_test(
            model=model,
            input_data=input_data,
            prefill_seq_len=20,
            verify_model_type="gemma4",
            verify_numerics=False,
            verify_generation=False,
        )

    def test_gemma4_litertlm_export_separate_vision_encoder(self):
        """Export Gemma4 with separate vision encoder/adapter models."""
        import keras

        if keras.config.backend() != "torch":
            self.skipTest("LiteRT-LM export requires the PyTorch backend.")

        model, input_data = self._build_tiny_gemma4_multimodal_model()

        self.run_litertlm_export_test(
            model=model,
            input_data=input_data,
            prefill_seq_len=20,
            verify_model_type="gemma4",
            verify_numerics=False,
            verify_generation=False,
            separate_vision_encoder=True,
        )

        # Extract all TFLite models and verify the multimodal signatures.
        litertlm_path = os.path.join(self.get_temp_dir(), "model.litertlm")
        interpreters = self._extract_litertlm_tflite_interpreters(litertlm_path)

        all_signatures = {}
        for interpreter in interpreters:
            all_signatures.update(interpreter._get_full_signature_list())

        signature_names = set(all_signatures.keys())
        self.assertIn("prefill", signature_names)
        self.assertIn("decode", signature_names)
        self.assertIn("vision_encoder", signature_names)
        self.assertIn("vision_adapter", signature_names)

        prefill_inputs = set(all_signatures["prefill"]["inputs"])
        self.assertNotIn("images", prefill_inputs)
        self.assertNotIn("pixel_values", prefill_inputs)
        self.assertNotIn("pixel_position_ids", prefill_inputs)
        self.assertIn("mm_embedding", prefill_inputs)

        vision_encoder_inputs = set(all_signatures["vision_encoder"]["inputs"])
        self.assertIn("pixel_values", vision_encoder_inputs)
        self.assertIn("pixel_position_ids", vision_encoder_inputs)
