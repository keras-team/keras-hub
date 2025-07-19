import unittest

import keras
import numpy as np

import keras_hub


@unittest.skipIf(
    keras.backend.backend() == "openvino",
    "Skip for non-trainable backends like OpenVINO",
)
class BasicUsageTest(unittest.TestCase):
    def test_transformer(self):
        # Tokenize some inputs with a binary label.
        vocab = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        sentences = ["The quick brown fox jumped.", "The fox slept."]
        tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
            vocabulary=vocab,
            sequence_length=10,
        )
        x, y = tokenizer(sentences), np.array([1, 0])

        # Create a tiny transformer.
        inputs = keras.Input(shape=(None,), dtype="int32")
        outputs = keras_hub.layers.TokenAndPositionEmbedding(
            vocabulary_size=len(vocab),
            sequence_length=10,
            embedding_dim=16,
        )(inputs)
        outputs = keras_hub.layers.TransformerEncoder(
            num_heads=4,
            intermediate_dim=32,
        )(outputs)
        outputs = keras.layers.GlobalAveragePooling1D()(outputs)
        outputs = keras.layers.Dense(1, activation="sigmoid")(outputs)
        model = keras.Model(inputs, outputs)

        # Run a single batch of gradient descent.
        model.compile(loss="binary_crossentropy")
        loss = model.train_on_batch(x, y)

        # Make sure we have a valid loss.
        self.assertGreater(loss, 0)

    def test_quickstart(self):
        """This roughly matches the quick start example in our base README."""
        # Load a BERT model.
        classifier = keras_hub.models.TextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2,
            activation="softmax",
        )
        # Fine-tune.
        classifier.fit(x=["foo", "bar", "baz"], y=[1, 0, 1], batch_size=2)
        # Predict two new examples.
        classifier.predict(
            ["What an amazing movie!", "A total waste of my time."]
        )
