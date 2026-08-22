import numpy as np
import pytest
import torch
from transformers import AutoModel

from keras_hub.src.models.modernbert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.convert_modern_bert import (
    convert_backbone_config,
)
from keras_hub.src.utils.transformers.convert_modern_bert import convert_weights
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader


class TestConvertModernBert(TestCase):
    @pytest.mark.extra_large
    def test_convert_modern_bert_base(self):
        hf_model_id = "answerdotai/ModernBERT-base"
        preset = f"hf://{hf_model_id}"

        hf_model = AutoModel.from_pretrained(hf_model_id)
        hf_model.eval()

        # Convert configuration and instantiate backbone
        hf_config = hf_model.config.to_dict()
        keras_config = convert_backbone_config(hf_config)
        backbone = ModernBertBackbone(**keras_config)

        # Port weights
        with SafetensorLoader(preset) as loader:
            convert_weights(backbone, loader, hf_config)

        # Generate deterministic dummy inputs
        np.random.seed(42)
        batch_size = 2
        seq_len = 16
        token_ids = np.random.randint(
            100, hf_config["vocab_size"] - 100, size=(batch_size, seq_len)
        )
        padding_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        # Run Hugging Face forward pass
        with torch.no_grad():
            hf_outputs = (
                hf_model(
                    input_ids=torch.from_numpy(token_ids),
                    attention_mask=torch.from_numpy(padding_mask),
                )
                .last_hidden_state.cpu()
                .numpy()
            )

        # Run KerasHub forward pass
        keras_inputs = {
            "token_ids": token_ids,
            "padding_mask": padding_mask.astype(bool),
        }
        keras_outputs = backbone(keras_inputs)

        if hasattr(keras_outputs, "numpy"):
            keras_outputs = keras_outputs.numpy()

        self.assertAllClose(hf_outputs, keras_outputs, atol=1e-4, rtol=1e-4)
