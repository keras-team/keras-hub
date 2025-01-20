import numpy as np
import pytest

from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_ocr import PARSeqOCR
from keras_hub.src.models.parseq.parseq_preprocessor import PARSeqPreprocessor
from keras_hub.src.tests.test_case import TestCase


class PARSeqOCRTest(TestCase):
    def setUp(self):
        self.images = np.ones((2, 32, 128, 3))
        self.labels = np.ones((2, 4), int)

        self.backbone = PARSeqBackbone(alphabet_size=5, max_text_length=3)
        self.preprocessor = PARSeqPreprocessor()
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (self.images, self.labels)

    def test_basics(self):
        self.run_task_test(
            cls=PARSeqOCR,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 4, 3),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=PARSeqOCR,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_end_to_end_model_predict(self):
        model = PARSeqOCR(**self.init_kwargs)
        outputs = model.predict(self.images)
        self.assertAllEqual(outputs.shape, (2, 4, 3))

    @pytest.mark.skip(reason="disabled until preset's been uploaded to Kaggle")
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in PARSeqOCR.presets:
            self.run_preset_test(
                cls=PARSeqOCR,
                preset=preset,
                input_data=self.images,
                expected_output_shape=(2, 4, 3),
            )
