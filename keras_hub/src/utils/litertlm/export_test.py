import os
import struct

import numpy as np
import tensorflow as tf
import torch

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.litertlm.export import export_to_litertlm


class TestLiteRTLmExport(TestCase):
    def test_export_tiny_gemma(self):
        proto = os.path.join(
            self.get_test_data_dir(), "gemma_test_vocab.spm"
        )
        tokenizer = GemmaTokenizer(proto=proto)

        backbone = GemmaBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=32,
            head_dim=8,
            intermediate_dim=64,
            max_sequence_length=8,
        )
        preprocessor = GemmaCausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=8
        )
        model = GemmaCausalLM(backbone=backbone, preprocessor=preprocessor)

        # Set random weights for determinism.
        rng = np.random.default_rng(42)
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        model.set_weights(weights)

        path = os.path.join(self.get_temp_dir(), "test.litertlm")
        export_to_litertlm(model, path, prefill_seq_len=8)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_export_outputs_match_keras(self):
        """Verify that exported TFLite outputs match Keras eager outputs."""
        proto = os.path.join(
            self.get_test_data_dir(), "gemma_test_vocab.spm"
        )
        tokenizer = GemmaTokenizer(proto=proto)

        backbone = GemmaBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=32,
            head_dim=8,
            intermediate_dim=64,
            max_sequence_length=8,
        )
        preprocessor = GemmaCausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=8
        )
        model = GemmaCausalLM(backbone=backbone, preprocessor=preprocessor)

        rng = np.random.default_rng(42)
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        model.set_weights(weights)

        # Export
        litertlm_path = os.path.join(self.get_temp_dir(), "verify.litertlm")
        export_to_litertlm(model, litertlm_path, prefill_seq_len=8)

        # Extract TFLite
        with open(litertlm_path, "rb") as f:
            data = f.read()
        header_end = struct.unpack("<Q", data[24:32])[0]
        import ai_edge_litert.internal.litertlm_core as core

        metadata_buf = data[32:header_end]
        metadata = core.schema.LiteRTLMMetaData.GetRootAsLiteRTLMMetaData(
            metadata_buf, 0
        )
        tflite_path = os.path.join(self.get_temp_dir(), "verify.tflite")
        for i in range(metadata.SectionMetadata().ObjectsLength()):
            obj = metadata.SectionMetadata().Objects(i)
            if core.any_section_data_type_to_string(obj.DataType()) == "TFLiteModel":
                tflite_data = data[obj.BeginOffset() : obj.EndOffset()]
                with open(tflite_path, "wb") as f:
                    f.write(tflite_data)

        interpreter = tf.lite.Interpreter(model_path=tflite_path)

        B, T, L = 1, 8, 2
        H = backbone.num_key_value_heads
        D = backbone.head_dim
        tokens_np = np.arange(11, 11 + T, dtype=np.int32).reshape(B, T) % 11
        cache_keras = np.zeros((B, L, 2, T, H, D), dtype=np.float32)

        # Keras prefill
        with torch.no_grad():
            keras_logits, _, keras_cache = model.call_with_cache(
                torch.from_numpy(tokens_np),
                torch.from_numpy(cache_keras),
                0,
            )
        keras_logits = keras_logits.numpy()
        keras_cache = keras_cache.numpy()

        # TFLite prefill
        prefill_runner = interpreter.get_signature_runner("prefill")
        prefill_inputs = {
            "tokens": tokens_np,
            "input_pos": np.arange(T, dtype=np.int32),
            "mask": np.ones((B, 1, T, T), dtype=np.float32),
        }
        for i in range(L):
            prefill_inputs[f"kv_cache_k_{i}"] = cache_keras[:, i, 0, ...]
            prefill_inputs[f"kv_cache_v_{i}"] = cache_keras[:, i, 1, ...]
        tflite_prefill_out = prefill_runner(**prefill_inputs)

        # Compare prefill logits
        self.assertAllClose(
            keras_logits,
            tflite_prefill_out["logits"],
            atol=1e-4,
            rtol=1e-4,
        )

        # Compare prefill KV caches
        for i in range(L):
            self.assertAllClose(
                keras_cache[:, i, 0, ...],
                tflite_prefill_out[f"kv_cache_k_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )
            self.assertAllClose(
                keras_cache[:, i, 1, ...],
                tflite_prefill_out[f"kv_cache_v_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )

        # Keras decode at position 3
        decode_pos = 3
        decode_token = tokens_np[:, decode_pos : decode_pos + 1].copy()
        with torch.no_grad():
            keras_logits_dec, _, keras_cache_dec = model.call_with_cache(
                torch.from_numpy(decode_token),
                torch.from_numpy(keras_cache),
                decode_pos,
            )
        keras_logits_dec = keras_logits_dec.numpy()
        keras_cache_dec = keras_cache_dec.numpy()

        # TFLite decode
        decode_runner = interpreter.get_signature_runner("decode")
        decode_inputs = {
            "tokens": decode_token,
            "input_pos": np.array([decode_pos], dtype=np.int32),
            "mask": np.ones((B, 1, 1, T), dtype=np.float32),
        }
        for i in range(L):
            decode_inputs[f"kv_cache_k_{i}"] = tflite_prefill_out[
                f"kv_cache_k_{i}"
            ]
            decode_inputs[f"kv_cache_v_{i}"] = tflite_prefill_out[
                f"kv_cache_v_{i}"
            ]
        tflite_dec_out = decode_runner(**decode_inputs)

        # Compare decode logits
        self.assertAllClose(
            keras_logits_dec,
            tflite_dec_out["logits"],
            atol=1e-4,
            rtol=1e-4,
        )

        # Compare decode KV caches
        for i in range(L):
            self.assertAllClose(
                keras_cache_dec[:, i, 0, ...],
                tflite_dec_out[f"kv_cache_k_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )
            self.assertAllClose(
                keras_cache_dec[:, i, 1, ...],
                tflite_dec_out[f"kv_cache_v_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )
