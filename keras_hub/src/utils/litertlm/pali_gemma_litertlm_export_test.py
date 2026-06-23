import importlib.util
import os
import unittest

import numpy as np

_LITERT_TORCH_AVAILABLE = (
    importlib.util.find_spec("litert_torch") is not None
)
_LITERT_LM_BUILDER_AVAILABLE = (
    importlib.util.find_spec("litert_lm_builder") is not None
)

from keras_hub.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_hub.src.models.pali_gemma.pali_gemma_causal_lm import (
    PaliGemmaCausalLM,
)
from keras_hub.src.models.pali_gemma.pali_gemma_causal_lm_preprocessor import (
    PaliGemmaCausalLMPreprocessor,
)
from keras_hub.src.models.pali_gemma.pali_gemma_image_converter import (
    PaliGemmaImageConverter,
)
from keras_hub.src.models.pali_gemma.pali_gemma_tokenizer import (
    PaliGemmaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase



@unittest.skipIf(
    not _LITERT_TORCH_AVAILABLE,
    "Requires litert-torch.",
)
@unittest.skipIf(
    not _LITERT_LM_BUILDER_AVAILABLE,
    "Requires litert-lm-builder.",
)
class TestPaliGemmaLiteRTLmExport(TestCase):
    def _build_tiny_model(self):
        proto = os.path.join(self.get_test_data_dir(), "gemma_test_vocab.spm")
        tokenizer = PaliGemmaTokenizer(proto=proto)
        image_converter = PaliGemmaImageConverter(image_size=(16, 16))
        preprocessor = PaliGemmaCausalLMPreprocessor(
            tokenizer=tokenizer,
            image_converter=image_converter,
            sequence_length=8,
        )
        backbone = PaliGemmaBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vit_patch_size=4,
            vit_num_layers=1,
            vit_num_heads=2,
            vit_hidden_dim=8,
            vit_intermediate_dim=16,
        )
        model = PaliGemmaCausalLM(
            preprocessor=preprocessor,
            backbone=backbone,
        )

        # Set random weights for determinism.
        rng = np.random.default_rng(42)
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        model.set_weights(weights)
        return model

    def test_pali_gemma_litertlm_export_baked_in(self):
        model = self._build_tiny_model()
        input_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        self.run_litertlm_export_test(
            model=model,
            input_data=input_data,
            prefill_seq_len=8,
            verify_model_type="generic_model",
            verify_numerics=False,
            verify_generation=True,
            generation_max_tokens=4,
        )

    def test_pali_gemma_litertlm_export_separate_vision_encoder(self):
        import struct

        import tensorflow as tf
        from litert_lm_builder import litertlm_core as core

        model = self._build_tiny_model()
        input_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        self.run_litertlm_export_test(
            model=model,
            input_data=input_data,
            prefill_seq_len=8,
            verify_model_type="generic_model",
            verify_numerics=False,
            verify_generation=False,
            separate_vision_encoder=True,
        )

        # Verify the bundle contains the separate vision encoder/adapter
        # signatures, and that the prefill signature consumes ``mm_embedding``
        # instead of raw images.
        path = os.path.join(self.get_temp_dir(), "model.litertlm")
        self.assertTrue(os.path.exists(path))

        with open(path, "rb") as f:
            data = f.read()
        header_end = struct.unpack("<Q", data[24:32])[0]
        metadata = core.schema.LiteRTLMMetaData.GetRootAsLiteRTLMMetaData(
            data[32:header_end], 0
        )

        all_signatures = {}
        for i in range(metadata.SectionMetadata().ObjectsLength()):
            obj = metadata.SectionMetadata().Objects(i)
            if (
                core.any_section_data_type_to_string(obj.DataType())
                != "TFLiteModel"
            ):
                continue
            tflite_data = data[obj.BeginOffset() : obj.EndOffset()]
            tflite_path = os.path.join(
                self.get_temp_dir(),
                f"separate_vision_{len(all_signatures)}.tflite",
            )
            with open(tflite_path, "wb") as f:
                f.write(tflite_data)
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
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
        self.assertIn("images", vision_encoder_inputs)
