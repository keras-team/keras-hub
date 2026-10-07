import pytest

from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image_preprocessor import (  # noqa: E501
    StableDiffusion3TextToImagePreprocessor,
)
from keras_hub.src.models.text_to_image_preprocessor import (
    TextToImagePreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class TestTextToImagePreprocessor(TestCase):
    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            TextToImagePreprocessor.from_preset("stable_diffusion_3_medium"),
            StableDiffusion3TextToImagePreprocessor,
        )
        self.assertIsInstance(
            StableDiffusion3TextToImagePreprocessor.from_preset(
                "stable_diffusion_3_medium"
            ),
            StableDiffusion3TextToImagePreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            StableDiffusion3TextToImagePreprocessor.from_preset("gpt2_base_en")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            StableDiffusion3TextToImagePreprocessor.from_preset(
                "hf://spacy/en_core_web_sm"
            )
