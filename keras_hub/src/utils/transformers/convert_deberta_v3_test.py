import os

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


class DebertaV3ConverterTest(TestCase):
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
            "layer_norm_eps": 1e-7,
            "max_relative_positions": -1,
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

    def test_convert_deberta_v2_backbone_config(self):
        # Original DeBERTa-v2 checkpoints (e.g. `microsoft/deberta-v2-xlarge`)
        # add a convolution block after the first transformer layer.
        keras_config = convert_deberta_v3.convert_backbone_config(
            self._v3_config(conv_kernel_size=3, conv_act="gelu")
        )
        self.assertEqual(keras_config["conv_kernel_size"], 3)
        self.assertEqual(keras_config["conv_activation"], "gelu")
        self.assertEqual(keras_config["conv_groups"], 1)
        # HF falls back to `tanh` when `conv_act` is not set.
        keras_config = convert_deberta_v3.convert_backbone_config(
            self._v3_config(conv_kernel_size=3)
        )
        self.assertEqual(keras_config["conv_activation"], "tanh")

    def test_unsupported_config_raises(self):
        # Even kernels can't keep the sequence length with symmetric padding.
        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            convert_deberta_v3.convert_backbone_config(
                self._v3_config(conv_kernel_size=2)
            )
        with self.assertRaisesRegex(ValueError, "position_biased_input"):
            convert_deberta_v3.convert_backbone_config(
                self._v3_config(position_biased_input=True)
            )
        # Values `DebertaV3Backbone` hard-codes must not be dropped silently.
        with self.assertRaisesRegex(ValueError, "layer_norm_eps"):
            convert_deberta_v3.convert_backbone_config(
                self._v3_config(layer_norm_eps=1e-5)
            )
        with self.assertRaisesRegex(ValueError, "max_relative_positions"):
            convert_deberta_v3.convert_backbone_config(
                self._v3_config(max_relative_positions=256)
            )
        # `max_relative_positions == max_position_embeddings` is equivalent.
        convert_deberta_v3.convert_backbone_config(
            self._v3_config(max_relative_positions=512)
        )

    def test_null_config_values_raise_clear_error(self):
        # HF configs can store `null` for these keys; that should produce
        # the validation `ValueError`, not a `TypeError`/`AttributeError`.
        with self.assertRaisesRegex(ValueError, "position_buckets"):
            convert_deberta_v3.convert_backbone_config(
                self._v3_config(position_buckets=None)
            )
        with self.assertRaisesRegex(ValueError, "norm_rel_ebd"):
            convert_deberta_v3.convert_backbone_config(
                self._v3_config(norm_rel_ebd=None)
            )
        with self.assertRaisesRegex(ValueError, "pos_att_type"):
            convert_deberta_v3.convert_backbone_config(
                self._v3_config(pos_att_type=None)
            )

    def test_deberta_v2_numerics(self):
        """Convert a small random DeBERTa-v2 model (with the convolution block)
        and compare it with `transformers.DebertaV2Model`."""
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        torch.manual_seed(0)
        hf_config = transformers.DebertaV2Config(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            type_vocab_size=0,
            relative_attention=True,
            position_buckets=8,
            norm_rel_ebd="layer_norm",
            share_att_key=True,
            pos_att_type="p2c|c2p",
            position_biased_input=False,
            layer_norm_eps=1e-7,
            conv_kernel_size=3,
            conv_act="gelu",
        )
        hf_model = transformers.DebertaV2Model(hf_config).eval()
        with torch.no_grad():
            for param in hf_model.parameters():
                param.add_(torch.randn_like(param) * 0.05)
        preset_dir = os.path.join(self.get_temp_dir(), "deberta_v2")
        hf_model.save_pretrained(preset_dir, safe_serialization=True)

        token_ids = np.array(
            [[1, 5, 9, 17, 33, 2, 0, 0], [1, 3, 4, 5, 6, 7, 8, 2]],
            dtype="int32",
        )
        padding_mask = (token_ids != 0).astype("int32")
        with torch.no_grad():
            expected = hf_model(
                input_ids=torch.from_numpy(token_ids).long(),
                attention_mask=torch.from_numpy(padding_mask).long(),
            ).last_hidden_state.numpy()

        backbone = Backbone.from_preset(preset_dir)
        self.assertIsInstance(backbone, DebertaV3Backbone)
        self.assertEqual(backbone.conv_kernel_size, 3)
        output = backbone.predict(
            {"token_ids": token_ids, "padding_mask": padding_mask}
        )
        real = padding_mask.astype(bool)
        mean_diff = np.mean(np.abs(output[real] - expected[real]))
        self.assertLess(mean_diff, 1e-4)

    @pytest.mark.large
    def test_convert_preset(self):
        model = DebertaV3TextClassifier.from_preset(HF_PRESET, num_classes=3)
        prompt = "That movie was terrible."
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
        # Token ids for "Objects is adults!" (`[CLS] ... [SEP]`) from the
        # checkpoint's `spm.model`.
        token_ids = np.array([[1, 29630, 269, 2614, 300, 2]], dtype="int32")
        padding_mask = np.ones_like(token_ids)
        output = model.predict(
            {"token_ids": token_ids, "padding_mask": padding_mask}
        )
        self.assertEqual(output.shape, (1, 6, 384))
        # `last_hidden_state[0, :, :4]` from `transformers.DebertaV2Model`
        # loaded from the same checkpoint, on the same token ids.
        expected = np.array(
            [
                [-1.476733, 1.197991, 1.489652, -0.280192],
                [-0.705442, 1.241022, 1.516240, -2.142490],
                [-0.736595, 1.288076, 1.579048, -1.323771],
                [-0.254223, 1.009501, 1.733518, -0.488663],
                [-0.878730, 1.230889, 1.323087, -0.687177],
                [-1.558671, 1.215805, 1.454251, 0.293135],
            ]
        )
        mean_diff = np.mean(np.abs(output[0, :, :4] - expected))
        self.assertLess(mean_diff, 1e-4)
