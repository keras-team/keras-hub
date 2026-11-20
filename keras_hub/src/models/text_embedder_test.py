from unittest import mock

import keras
import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.text_embedder import TextEmbedder
from keras_hub.src.tests.test_case import TestCase


class SimpleTextEmbedder(TextEmbedder):
    def __init__(self, preprocessor=None, **kwargs):
        self.preprocessor = preprocessor
        inputs = keras.Input(shape=(4,))
        outputs = keras.layers.Dense(3)(inputs)
        super().__init__(inputs, outputs, **kwargs)


# Integer-valued floats, so every product and sum is exact in float32.
QUERY = [[1.0, 0.0, 2.0, 1.0], [0.0, 3.0, 1.0, 2.0]]
DOCUMENTS = [
    [1.0, 1.0, 0.0, 2.0],
    [2.0, 0.0, 1.0, 0.0],
    [0.0, 4.0, 2.0, 1.0],
]
# `QUERY @ DOCUMENTS.T`, shape `(num_queries, num_documents)`.
SIMILARITY = [[3.0, 4.0, 5.0], [7.0, 1.0, 16.0]]


class TextEmbedderTest(TestCase):
    @parameterized.named_parameters(
        ("numpy", lambda x: np.array(x, dtype="float32")),
        ("nested_list", lambda x: x),
        ("tensor", lambda x: ops.convert_to_tensor(x, dtype="float32")),
    )
    def test_similarity(self, convert):
        embedder = SimpleTextEmbedder()
        self.assertAllClose(
            embedder.similarity(convert(QUERY), convert(DOCUMENTS)),
            np.array(SIMILARITY, dtype="float32"),
        )

    def test_encode_documents_wraps_string_and_forwards_kwargs(self):
        embedder = SimpleTextEmbedder()
        embeddings = np.ones((1, 3), dtype="float32")
        with mock.patch.object(
            SimpleTextEmbedder, "predict", return_value=embeddings
        ) as predict:
            # A single string is wrapped in a list.
            self.assertIs(
                embedder.encode_documents("a", verbose=0, batch_size=2),
                embeddings,
            )
            predict.assert_called_once_with(["a"], verbose=0, batch_size=2)
            predict.reset_mock()
            # A list is passed through as is.
            documents = ["a", "b"]
            self.assertIs(embedder.encode_documents(documents), embeddings)
            predict.assert_called_once_with(documents)
            self.assertIs(predict.call_args.args[0], documents)

    def test_encode_documents_matches_predict(self):
        embedder = SimpleTextEmbedder()
        x = np.array(
            [[1.0, 0.0, 2.0, 1.0], [0.0, 3.0, 1.0, 2.0], [2.0, 1.0, 0.0, 4.0]],
            dtype="float32",
        )
        documents = embedder.encode_documents(x)
        self.assertAllClose(documents, embedder.predict(x))
        self.assertEqual(tuple(ops.shape(documents)), (3, 3))
        self.assertEqual(
            tuple(ops.shape(embedder.similarity(documents, documents))),
            (3, 3),
        )
