import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.models.vit.vit_image_classifier import ViTImageClassifier
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="TensorFlow GPU CI OOM (ResourceExhaustedError)",
)
class TestViTConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_preset(self):
        model = ViTImageClassifier.from_preset(
            "hf://google/vit-base-patch16-224"
        )
        output = model.predict(
            {"images": np.ones((1, 224, 224, 3), dtype="float32")}
        )
        self.assertEqual(output.shape, (1, 1000))

    @pytest.mark.extra_large
    def test_class_detection(self):
        model = ImageClassifier.from_preset(
            "hf://google/vit-base-patch16-224",
            load_weights=False,
        )
        self.assertIsInstance(model, ViTImageClassifier)
        model = Backbone.from_preset(
            "hf://google/vit-base-patch16-224",
            load_weights=False,
        )
        self.assertIsInstance(model, ViTBackbone)


class TestViTClassificationHead(TestCase):
    def _save_hf_image_classifier(self, preset_dir, num_labels):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        config = transformers.ViTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=37,
            image_size=32,
            patch_size=16,
            num_labels=num_labels,
            architectures=["ViTForImageClassification"],
        )
        torch.manual_seed(0)
        hf_model = transformers.ViTForImageClassification(config).eval()
        # HF zero-initializes biases. Randomize the head so that a head that
        # was never ported cannot pass by accident.
        with torch.no_grad():
            hf_model.classifier.weight.normal_(std=0.2)
            hf_model.classifier.bias.normal_(std=0.2)
        hf_model.save_pretrained(preset_dir)
        return hf_model

    @pytest.mark.large
    def test_default_two_labels(self):
        with tempfile.TemporaryDirectory() as preset_dir:
            # `config.json` omits `id2label` for the default two-label setup.
            hf_model = self._save_hf_image_classifier(preset_dir, num_labels=2)
            keras_model = ImageClassifier.from_preset(
                preset_dir, preprocessor=None
            )

        self.assertIsInstance(keras_model, ViTImageClassifier)
        self.assertEqual(keras_model.num_classes, 2)
        self.assertAllClose(
            ops.convert_to_numpy(keras_model.output_dense.kernel),
            hf_model.classifier.weight.detach().numpy().T,
        )
        self.assertAllClose(
            ops.convert_to_numpy(keras_model.output_dense.bias),
            hf_model.classifier.bias.detach().numpy(),
        )

    @pytest.mark.large
    def test_classification_head_without_weights(self):
        with tempfile.TemporaryDirectory() as preset_dir:
            self._save_hf_image_classifier(preset_dir, num_labels=3)
            # `load_weights=False` must not read the weights file at all.
            os.remove(os.path.join(preset_dir, "model.safetensors"))
            keras_model = ImageClassifier.from_preset(
                preset_dir, preprocessor=None, load_weights=False
            )

        self.assertIsInstance(keras_model, ViTImageClassifier)
        self.assertEqual(keras_model.num_classes, 3)
