import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mpnet.mpnet_backbone import MPNetBackbone
from keras_hub.src.models.mpnet.mpnet_text_embedder import MPNetTextEmbedder
from keras_hub.src.models.text_embedder import TextEmbedder
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_preset(self):
        model = MPNetTextEmbedder.from_preset(
            "hf://sentence-transformers/all-mpnet-base-v2"
        )
        embeddings = model.encode_text(["That movie was terrible."])
        self.assertEqual(embeddings.shape, (1, 768))

    @pytest.mark.large
    def test_class_detection(self):
        model = TextEmbedder.from_preset(
            "hf://sentence-transformers/all-mpnet-base-v2",
            load_weights=False,
        )
        self.assertIsInstance(model, MPNetTextEmbedder)
        # Pipeline settings come from the sentence-transformer config files.
        self.assertEqual(model.pooling_mode, "mean")
        self.assertTrue(model.normalize)
        self.assertEqual(model.preprocessor.sequence_length, 384)
        self.assertTrue(model.preprocessor.tokenizer.strip_accents)
        model = Backbone.from_preset(
            "hf://sentence-transformers/all-mpnet-base-v2",
            load_weights=False,
        )
        self.assertIsInstance(model, MPNetBackbone)
        self.assertEqual(model.max_sequence_length, 512)
