from keras_hub.src.models.llama.llama_attention import LlamaAttention
from keras_hub.src.models.llama.llama_decoder import LlamaTransformerDecoder
from keras_hub.src.tests.test_case import TestCase

ROPE_KWARGS = {
    "rope_position_scaling_factor": 2.0,
    "rope_frequency_adjustment_factor": 8.0,
    "rope_low_freq_factor": 1.0,
    "rope_high_freq_factor": 4.0,
    "rope_pretraining_sequence_length": 8192,
}


class LlamaAttentionTest(TestCase):
    def test_config_serialization(self):
        layer = LlamaAttention(num_query_heads=4, num_key_value_heads=2)
        self.run_serialization_test(layer)

    def test_get_config_preserves_rope_args(self):
        layer = LlamaAttention(
            num_query_heads=4, num_key_value_heads=2, **ROPE_KWARGS
        )
        revived = LlamaAttention.from_config(layer.get_config())
        for name, value in ROPE_KWARGS.items():
            self.assertEqual(getattr(revived, name), value)


class LlamaTransformerDecoderTest(TestCase):
    def test_config_serialization(self):
        layer = LlamaTransformerDecoder(
            intermediate_dim=16, num_query_heads=4, num_key_value_heads=2
        )
        self.run_serialization_test(layer)

    def test_get_config_preserves_rope_args(self):
        layer = LlamaTransformerDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            **ROPE_KWARGS,
        )
        revived = LlamaTransformerDecoder.from_config(layer.get_config())
        for name, value in ROPE_KWARGS.items():
            self.assertEqual(getattr(revived, name), value)
