import numpy as np
import pytest

from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_causal_lm import PARSeqCausalLM
from keras_hub.src.models.parseq.parseq_causal_lm_preprocessor import (
    PARSeqCausalLMPreprocessor,
)
from keras_hub.src.models.parseq.parseq_image_converter import (
    PARSeqImageConverter,
)
from keras_hub.src.models.parseq.parseq_tokenizer import PARSeqTokenizer
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.tests.test_case import TestCase


class PARSeqCausalLMTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_height = 32
        self.image_width = 128
        self.num_channels = 3

        # Image Encoder parameters (as per your example)
        self.vit_patch_size = (4, 8)
        self.vit_num_layers = 2
        self.vit_num_heads = 2
        self.vit_hidden_dim = 64
        self.vit_mlp_dim = self.vit_hidden_dim * 4

        # PARSeq Backbone parameters
        self.vocabulary_size = 97
        self.max_label_length = 25
        self.decoder_hidden_dim = self.vit_hidden_dim
        self.num_decoder_layers = 1
        self.num_decoder_heads = 2
        self.decoder_mlp_dim = self.decoder_hidden_dim * 4

        image_converter = PARSeqImageConverter(
            image_size=[32, 128],
            offset=-1,
            scale=1.0 / 255.0 / 0.5,
            interpolation="bicubic",
        )
        tokenizer = PARSeqTokenizer()

        preprocessor = PARSeqCausalLMPreprocessor(
            image_converter=image_converter, tokenizer=tokenizer
        )

        image_encoder = ViTBackbone(
            image_shape=(
                self.image_height,
                self.image_width,
                self.num_channels,
            ),
            patch_size=self.vit_patch_size,
            num_layers=self.vit_num_layers,
            num_heads=self.vit_num_heads,
            hidden_dim=self.vit_hidden_dim,
            mlp_dim=self.vit_mlp_dim,
            use_class_token=False,
            name="image_encoder",
        )

        backbone = PARSeqBackbone(
            image_encoder=image_encoder,
            vocabulary_size=self.vocabulary_size,
            max_label_length=self.max_label_length,
            num_decoder_heads=self.num_decoder_heads,
            num_decoder_layers=self.num_decoder_layers,
            decoder_hidden_dim=self.decoder_hidden_dim,
            decoder_mlp_dim=self.decoder_mlp_dim,
        )

        self.init_kwargs = {"preprocessor": preprocessor, "backbone": backbone}

        # Dummy input data
        dummy_images = np.random.randn(
            self.batch_size,
            self.image_height,
            self.image_width,
            self.num_channels,
        )

        self.train_data = (
            {"images": dummy_images, "responses": ["abc", "xyz"]},
        )

    @pytest.mark.large
    def test_causal_lm_basics(self):
        expected_shape_full = (
            self.batch_size,
            self.max_label_length,
            self.vocabulary_size - 2,
        )

        self.run_task_test(
            cls=PARSeqCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=expected_shape_full,
        )

    def test_litert_export(self):
        # Create input data for export test
        input_data = {
            "images": np.random.randn(
                self.batch_size,
                self.image_height,
                self.image_width,
                self.num_channels,
            ),
            "token_ids": np.random.randint(
                0,
                self.vocabulary_size,
                (self.batch_size, self.max_label_length),
            ),
            "padding_mask": np.ones(
                (self.batch_size, self.max_label_length), dtype="int32"
            ),
        }
        self.run_litert_export_test(
            cls=PARSeqCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-4}},
        )
