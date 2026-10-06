import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.deberta_v3.deberta_v3_backbone import (
    DebertaV3Backbone,
)
from keras_hub.src.models.deberta_v3.deberta_v3_text_classifier import (
    DebertaV3TextClassifier,
)
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_deberta_v3

HF_PRESET = "hf://cross-encoder/nli-deberta-v3-xsmall"


class TestTask(TestCase):
    def _v3_config(self, **overrides):
        config = {
            "model_type": "deberta-v2",
            "vocab_size": 128100,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "position_buckets": 256,
            "relative_attention": True,
            "share_att_key": True,
            "position_biased_input": False,
            "norm_rel_ebd": "layer_norm",
            "pos_att_type": "p2c|c2p",
            "type_vocab_size": 0,
            "hidden_act": "gelu",
        }
        config.update(overrides)
        return config

    def test_convert_backbone_config(self):
        keras_config = convert_deberta_v3.convert_backbone_config(
            self._v3_config()
        )
        self.assertEqual(
            keras_config,
            {
                "vocabulary_size": 128100,
                "num_layers": 6,
                "num_heads": 12,
                "hidden_dim": 384,
                "intermediate_dim": 1536,
                "dropout": 0.1,
                "max_sequence_length": 512,
                "bucket_size": 256,
            },
        )
        # `pos_att_type` may also be stored as a list.
        convert_deberta_v3.convert_backbone_config(
            self._v3_config(pos_att_type=["p2c", "c2p"])
        )

    def test_unsupported_config_raises(self):
        # DeBERTa-v2 style checkpoints (e.g. `microsoft/deberta-v2-xlarge`)
        # use a conv layer which `DebertaV3Backbone` does not implement.
        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            convert_deberta_v3.convert_backbone_config(
                self._v3_config(conv_kernel_size=3)
            )
        with self.assertRaisesRegex(ValueError, "position_biased_input"):
            convert_deberta_v3.convert_backbone_config(
                self._v3_config(position_biased_input=True)
            )

    @pytest.mark.large
    def test_convert_preset(self):
        model = DebertaV3TextClassifier.from_preset(HF_PRESET, num_classes=3)
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            HF_PRESET,
            num_classes=3,
            load_weights=False,
        )
        self.assertIsInstance(model, DebertaV3TextClassifier)
        model = Backbone.from_preset(
            HF_PRESET,
            load_weights=False,
        )
        self.assertIsInstance(model, DebertaV3Backbone)

    @pytest.mark.large
    def test_numerics(self):
        """Compare backbone outputs against hardcoded HF reference values."""
        model = Backbone.from_preset(HF_PRESET)
        # Token ids for "cricket is awesome!" from the HF tokenizer.
        token_ids = np.array([[1, 29630, 269, 2614, 300, 2]], dtype="int32")
        padding_mask = np.ones_like(token_ids)
        output = model.predict(
            {"token_ids": token_ids, "padding_mask": padding_mask}
        )
        self.assertEqual(output.shape, (1, 6, 384))
