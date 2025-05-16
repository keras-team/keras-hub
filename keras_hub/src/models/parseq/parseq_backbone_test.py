import keras
from keras import ops

from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.tests.test_case import TestCase


class PARSeqBackboneTest(TestCase):
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

        # Instantiate the actual ViTBackbone to be used as the image_encoder
        self.image_encoder = ViTBackbone(
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

        self.init_kwargs = {
            "image_encoder": self.image_encoder,
            "vocabulary_size": self.vocabulary_size,
            "max_label_length": self.max_label_length,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "num_decoder_layers": self.num_decoder_layers,
            "num_decoder_heads": self.num_decoder_heads,
            "decoder_mlp_dim": self.decoder_mlp_dim,
            "dropout_rate": 0.0,
            "attention_dropout": 0.0,
        }

        # Dummy input data
        dummy_images = keras.random.normal(
            shape=(
                self.batch_size,
                self.image_height,
                self.image_width,
                self.num_channels,
            ),
        )

        dummy_token_ids = keras.random.randint(
            minval=0,
            maxval=self.vocabulary_size,
            shape=(self.batch_size, self.max_label_length),
        )
        dummy_padding_mask = ops.ones(
            shape=(self.batch_size, self.max_label_length), dtype="int32"
        )

        self.input_data = {
            "images": dummy_images,
            "token_ids": dummy_token_ids,
            "padding_mask": dummy_padding_mask,
        }

    def test_backbone_basics(self):
        expected_shape_full = (
            self.batch_size,
            self.max_label_length,
            self.vocabulary_size - 2,
        )

        self.run_backbone_test(
            cls=PARSeqBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=expected_shape_full,
            # we have image_encoder as init_kwargs which is also a backbone
            run_quantization_check=False,
        )

    # @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=PARSeqBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
