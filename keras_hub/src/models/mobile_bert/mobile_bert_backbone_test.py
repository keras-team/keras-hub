import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class MobileBertBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "embedding_dim": 8,
            "intra_bottleneck_dim": 8,
            "intermediate_dim": 16,
            "max_sequence_length": 16,
            "num_segments": 2,
            "use_pooler_dense": True,
        }
        self.input_data = {
            "token_ids": np.ones((2, 4), dtype="int32"),
            "segment_ids": np.array(
                [[0, 0, 1, 1], [0, 1, 1, 0]], dtype="int32"
            ),
            "padding_mask": np.array(
                [[1, 1, 1, 1], [1, 1, 1, 0]], dtype="int32"
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MobileBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 4, 16),
                "pooled_output": (2, 16),
            },
            variable_length_data=(
                {
                    "token_ids": np.ones((2, 8), dtype="int32"),
                    "segment_ids": np.array(
                        [
                            [0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 1, 1, 1, 1, 0],
                        ],
                        dtype="int32",
                    ),
                    "padding_mask": np.array(
                        [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0]],
                        dtype="int32",
                    ),
                },
            ),
            run_mixed_precision_check=True,
            run_quantization_check=True,
        )
        fp16_model = MobileBertBackbone(
            **self.init_kwargs, dtype="mixed_float16"
        )
        fp16_outputs = fp16_model(self.input_data)
        self.assertTrue(
            np.isfinite(
                ops.convert_to_numpy(fp16_outputs["sequence_output"])
            ).all()
        )
        self.assertTrue(
            np.isfinite(
                ops.convert_to_numpy(fp16_outputs["pooled_output"])
            ).all()
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_backbone_without_pooler_dense(self):
        kwargs = self.init_kwargs.copy()
        kwargs["use_pooler_dense"] = False
        self.run_backbone_test(
            cls=MobileBertBackbone,
            init_kwargs=kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 4, 16),
                "pooled_output": (2, 16),
            },
        )
        model = MobileBertBackbone(**kwargs)
        outputs = model(self.input_data)

        sequence_output = outputs["sequence_output"]
        pooled_output = outputs["pooled_output"]

        # When use_pooler_dense is False, pooled_output is just the [CLS] token
        self.assertAllClose(pooled_output, sequence_output[:, 0, :])

    def test_backbone_layer_norm(self):
        kwargs = {
            **self.init_kwargs,
            "normalization_type": "layer_norm",
            "layer_norm_epsilon": 1e-5,
        }
        self.run_backbone_test(
            cls=MobileBertBackbone,
            init_kwargs=kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 4, 16),
                "pooled_output": (2, 16),
            },
        )
        model = MobileBertBackbone(**kwargs)
        self.assertIsInstance(
            model.embeddings_layer_norm, keras.layers.LayerNormalization
        )
        self.assertEqual(model.embeddings_layer_norm.epsilon, 1e-5)
        self.assertEqual(
            model.transformer_layers[0].attention.output_layer_norm.epsilon,
            1e-5,
        )
        restored = MobileBertBackbone.from_config(model.get_config())
        self.assertEqual(restored.normalization_type, "layer_norm")

    def test_segment_ids_affect_output(self):
        keras.utils.set_random_seed(0)
        model = MobileBertBackbone(**self.init_kwargs)
        inputs_zero = {
            **self.input_data,
            "segment_ids": np.zeros((2, 4), dtype="int32"),
        }
        out_zero = model(inputs_zero)["sequence_output"]
        out_mixed = model(self.input_data)["sequence_output"]
        self.assertNotAllClose(out_zero, out_mixed)

    def test_invalid_normalization_type(self):
        with self.assertRaisesRegex(ValueError, "normalization_type"):
            MobileBertBackbone(
                **{**self.init_kwargs, "normalization_type": "rms_norm"}
            )

    def test_invalid_num_ffn(self):
        with self.assertRaisesRegex(ValueError, "num_feedforward_networks"):
            MobileBertBackbone(
                **{**self.init_kwargs, "num_feedforward_networks": 0}
            )

    def test_rejects_indivisible_heads(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            MobileBertBackbone(**{**self.init_kwargs, "num_heads": 3})

    def test_preset_metadata_path(self):
        # keras.io links presets under `/api/models/<path>/`, which must match
        # the package directory `mobile_bert`.
        self.assertEqual(
            MobileBertBackbone.presets["mobilebert_uncased"]["metadata"][
                "path"
            ],
            "mobile_bert",
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MobileBertBackbone,
            preset="mobilebert_uncased",
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 4, 512),
                "pooled_output": (2, 512),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MobileBertBackbone.presets:
            self.run_preset_test(
                cls=MobileBertBackbone,
                preset=preset,
                input_data=self.input_data,
            )
