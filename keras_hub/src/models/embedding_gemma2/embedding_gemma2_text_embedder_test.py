import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_text_embedder import (  # noqa: E501
    EmbeddingGemma2TextEmbedder,
)
from keras_hub.src.tests.test_case import TestCase


class EmbeddingGemma2TextEmbedderTest(TestCase):
    def setUp(self):
        from keras_hub.src.models.embedding_gemma2.embedding_gemma2_audio_converter import (  # noqa: E501
            EmbeddingGemma2AudioConverter,
        )
        from keras_hub.src.models.embedding_gemma2.embedding_gemma2_image_converter import (  # noqa: E501
            EmbeddingGemma2ImageConverter,
        )
        from keras_hub.src.models.embedding_gemma2.embedding_gemma2_text_embedder_preprocessor import (  # noqa: E501
            EmbeddingGemma2TextEmbedderPreprocessor,
        )
        from keras_hub.src.models.embedding_gemma2.embedding_gemma2_video_converter import (  # noqa: E501
            EmbeddingGemma2VideoConverter,
        )
        from keras_hub.src.models.gemma4.gemma4_audio_encoder import (
            Gemma4AudioEncoder,
        )
        from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
            Gemma4VisionEncoder,
        )
        from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import (
            MockGemma4Tokenizer,
        )

        vision_encoder = Gemma4VisionEncoder(
            image_size=16,
            patch_size=16,
            num_layers=1,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            head_dim=8,
            pooling_kernel_size=1,
            output_dim=16,
        )
        audio_encoder = Gemma4AudioEncoder(
            input_feat_size=8,
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            chunk_size=4,
            context_left=5,
            context_right=0,
            sscp_conv_channels=(4, 2),
            sscp_kernel_sizes=((3, 3), (3, 3)),
            sscp_stride_sizes=((2, 2), (2, 2)),
            output_proj_dims=48,
            output_dim=16,
        )

        # We need a small backbone WITH encoders attached to cover text-only
        # predictions passing through placeholders.
        self.backbone = EmbeddingGemma2Backbone(
            vocabulary_size=100,
            image_size=16,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=16,
            intermediate_dim=32,
            head_dim=8,
            embedding_dim=12,
            layer_types=["sliding_attention", "full_attention"],
            vision_encoder=vision_encoder,
            audio_encoder=audio_encoder,
            num_audio_tokens_per_clip=5,
        )

        self.tokenizer = MockGemma4Tokenizer(add_bos=False, add_eos=False)
        self.image_converter = EmbeddingGemma2ImageConverter(
            image_size=(16, 16),
            patch_size=16,
            pooling_kernel_size=1,
            max_soft_tokens=10,
        )
        self.audio_converter = EmbeddingGemma2AudioConverter(
            num_mels=8,
            num_fft_bins=16,
            stride=8,
            max_audio_length=1,
            sampling_rate=160,
        )
        self.video_converter = EmbeddingGemma2VideoConverter(
            image_size=(16, 16),
            patch_size=16,
            pooling_kernel_size=1,
            max_frames=32,
        )

        self.preprocessor = EmbeddingGemma2TextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            audio_converter=self.audio_converter,
            video_converter=self.video_converter,
            sequence_length=4,
        )

        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (
            ["hello", "world"],
            np.ones((2, 12), dtype="float32"),
        )
        self.input_data = self.train_data[0]

    def test_embedder_basics(self):
        self.run_task_test(
            cls=EmbeddingGemma2TextEmbedder,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 12),
            batch_size=2,
            compile_kwargs={"optimizer": "adam", "loss": "mse"},
            atol=1e-06,
            rtol=1e-06,
        )

    def test_encode_text_mocked(self):
        embedder = EmbeddingGemma2TextEmbedder(**self.init_kwargs)
        # Mock predict to just return ones
        from unittest import mock

        with mock.patch.object(
            embedder, "predict", return_value=np.ones((1, 12))
        ):
            output = embedder.encode_text("hello")
            self.assertEqual(output.shape, (1, 12))

    def test_output_is_normalized(self):
        embedder = EmbeddingGemma2TextEmbedder(**self.init_kwargs)
        output = embedder.predict(["hello"])
        norms = np.linalg.norm(output, axis=-1)
        self.assertAllClose(norms, np.ones_like(norms), atol=1e-5)

    def test_mean_pooling_respects_mask(self):
        """Mean pooling must ignore every padded position."""
        embedder = EmbeddingGemma2TextEmbedder(**self.init_kwargs)
        inputs = {
            k: ops.convert_to_numpy(v)
            for k, v in embedder.preprocessor(["hello", "hello"]).items()
        }
        token_ids = inputs["token_ids"].copy()
        padding_mask = inputs["padding_mask"]
        pad_positions = np.where(~padding_mask[1])[0]
        self.assertGreater(len(pad_positions), 0)
        # Row 1 differs from row 0 only at masked positions.
        token_ids[1, pad_positions] = 99
        inputs["token_ids"] = token_ids

        out = ops.convert_to_numpy(embedder(inputs))
        self.assertAllClose(out[0], out[1])

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=EmbeddingGemma2TextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in EmbeddingGemma2TextEmbedder.presets:
            self.run_preset_test(
                cls=EmbeddingGemma2TextEmbedder,
                preset=preset,
                input_data=self.input_data,
            )
