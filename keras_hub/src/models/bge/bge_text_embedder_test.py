import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.bge.bge_backbone import BgeBackbone
from keras_hub.src.models.bge.bge_text_embedder import BgeTextEmbedder
from keras_hub.src.models.bge.bge_text_embedder_preprocessor import (
    BgeTextEmbedderPreprocessor,
)
from keras_hub.src.models.bge.bge_tokenizer import BgeTokenizer
from keras_hub.src.tests.test_case import TestCase


class BgeTextEmbedderTest(TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = BgeTextEmbedderPreprocessor(
            BgeTokenizer(vocabulary=self.vocab),
            sequence_length=8,
        )
        self.backbone = BgeBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=self.preprocessor.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.input_data = ["the quick brown fox.", "the fox."]
        self.preprocessed_data = self.preprocessor(self.input_data)

    def test_embedder_basics(self):
        embedder = BgeTextEmbedder(**self.init_kwargs)
        # Serialization roundtrip.
        self.run_serialization_test(embedder)
        # Eager call with preprocessed inputs.
        output = embedder(self.preprocessed_data)
        self.assertEqual(output.shape, (2, 4))
        # Compiled predict requires no preprocessor (to avoid re-tokenizing).
        embedder_no_prep = BgeTextEmbedder(
            backbone=self.backbone, preprocessor=None
        )
        output_predict = embedder_no_prep.predict(self.preprocessed_data)
        self.assertAllClose(output, output_predict)

    def test_output_is_l2_normalized(self):
        """All output vectors must have unit L2 norm."""
        embedder = BgeTextEmbedder(**self.init_kwargs)
        outputs = embedder(self.preprocessed_data)
        norms = ops.sqrt(ops.sum(ops.power(outputs, 2), axis=-1))
        self.assertAllClose(norms, ops.ones_like(norms), atol=1e-5)

    def test_without_preprocessor(self):
        """Model must accept pre-tokenized dict inputs
        when preprocessor=None."""
        embedder = BgeTextEmbedder(
            backbone=self.backbone,
            preprocessor=None,
        )
        outputs = embedder(self.preprocessed_data)
        self.assertEqual(outputs.shape, (2, 4))

    def test_dot_product_equals_cosine_similarity(self):
        """After L2 normalization, dot product must equal cosine similarity."""
        embedder = BgeTextEmbedder(**self.init_kwargs)
        outputs = ops.convert_to_numpy(embedder(self.preprocessed_data))
        # dot(a, b) == cosine_sim(a, b) when both are unit vectors
        dot = float(np.dot(outputs[0], outputs[1]))
        norms = np.linalg.norm(outputs, axis=-1)
        cosine = float(np.dot(outputs[0], outputs[1]) / (norms[0] * norms[1]))
        self.assertAlmostEqual(dot, cosine, places=5)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BgeTextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.preprocessed_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        # Token IDs for: [CLS] i love machine learning and nl ##p [SEP]
        self.run_preset_test(
            cls=BgeTextEmbedder,
            preset="bge_small_en_v1.5",
            input_data={
                "token_ids": ops.array(
                    [[101, 1045, 2293, 3698, 4083, 1998, 17953, 2361, 102]],
                    dtype="int32",
                ),
                "segment_ids": ops.zeros((1, 9), dtype="int32"),
                "padding_mask": ops.ones((1, 9), dtype="int32"),
            },
            expected_output_shape=(1, 384),
            expected_partial_output=ops.array(
                [-0.05340093, -0.02628993, 0.02447508, -0.02452144, 0.04503455]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        input_data = {
            "token_ids": ops.array(
                [[101, 1045, 2293, 3698, 4083, 102]], dtype="int32"
            ),
            "segment_ids": ops.zeros((1, 6), dtype="int32"),
            "padding_mask": ops.ones((1, 6), dtype="int32"),
        }
        for preset in BgeTextEmbedder.presets:
            self.run_preset_test(
                cls=BgeTextEmbedder,
                preset=preset,
                input_data=input_data,
            )
