import numpy as np
import pytest

from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_backbone import Qwen3OmniMoeBackbone


class Qwen3OmniMoeBackboneTest:
    def test_basic(self):
        # Test basic functionality
        model = Qwen3OmniMoeBackbone(
            vocabulary_size=151936,
            num_layers=32,
            num_query_heads=32,
            num_key_value_heads=4,
            hidden_dim=4096,
            intermediate_dim=11008,
            num_experts=8,
            num_experts_per_tok=2,
            head_dim=128,
            max_sequence_length=32768,
        )

        # Test forward pass
        token_ids = np.ones((1, 12), dtype="int32")
        padding_mask = np.ones((1, 12), dtype="bool")
        
        outputs = model(
            token_ids=token_ids,
            attention_mask=padding_mask,
        )
        
        assert outputs.shape == (1, 12, 4096)

    def test_from_preset(self):
        # Test loading from preset
        model = Qwen3OmniMoeBackbone.from_preset("qwen3_omni_moe_7b")
        
        # Test forward pass
        token_ids = np.ones((1, 12), dtype="int32")
        padding_mask = np.ones((1, 12), dtype="bool")
        
        outputs = model(
            token_ids=token_ids,
            attention_mask=padding_mask,
        )
        
        assert outputs.shape == (1, 12, 4096)

    def test_cache(self):
        # Test caching functionality
        model = Qwen3OmniMoeBackbone(
            vocabulary_size=151936,
            num_layers=32,
            num_query_heads=32,
            num_key_value_heads=4,
            hidden_dim=4096,
            intermediate_dim=11008,
            num_experts=8,
            num_experts_per_tok=2,
            head_dim=128,
            max_sequence_length=32768,
        )

        # First forward pass
        token_ids = np.ones((1, 12), dtype="int32")
        padding_mask = np.ones((1, 12), dtype="bool")
        
        outputs1 = model(
            token_ids=token_ids,
            attention_mask=padding_mask,
        )
        
        # Second forward pass with cache
        token_ids2 = np.ones((1, 1), dtype="int32")
        padding_mask2 = np.ones((1, 1), dtype="bool")
        
        outputs2 = model(
            token_ids=token_ids2,
            attention_mask=padding_mask2,
        )
        
        assert outputs1.shape == (1, 12, 4096)
        assert outputs2.shape == (1, 1, 4096)
