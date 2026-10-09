from keras import ops

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_encoder_block import (  # noqa: E501
    EmbeddingGemma2EncoderBlock,
)
from keras_hub.src.tests.test_case import TestCase


class EmbeddingGemma2EncoderBlockTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "num_query_heads": 2,
            "head_dim": 8,
            "num_key_value_heads": 1,
            "use_sliding_window_attention": True,
            "sliding_window_size": 2,
            "is_global_attention": False,
            "use_bidirectional_attention": True,
        }
        self.input_data = ops.ones((2, 6, 16))
        self.padding_mask = ops.array(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype="int32",
        )

    def test_encoder_block_basics(self):
        self.run_layer_test(
            cls=EmbeddingGemma2EncoderBlock,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=((2, 6, 16), (2, 2, 6, 1, 8)),
            expected_num_trainable_weights=13,
            expected_num_non_trainable_weights=1,
            expected_num_non_trainable_variables=1,
            run_training_check=False,
            run_precision_checks=False,
        )

    def test_bidirectional_sliding_mask(self):
        block = EmbeddingGemma2EncoderBlock(**self.init_kwargs)
        mask = block._compute_attention_mask(
            x=self.input_data,
            padding_mask=self.padding_mask,
            vision_mask=None,
            cache=None,
            cache_update_index=0,
        )
        # Expected mask for sequence 0 (length 4 valid, window size 2):
        # i=0: attends to j=0, 1, 2
        # i=1: attends to j=0, 1, 2, 3
        # i=2: attends to j=0, 1, 2, 3
        # i=3: attends to j=1, 2, 3
        expected_mask_seq0 = [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        self.assertAllClose(mask[0], expected_mask_seq0)

    def test_full_attention_mask(self):
        kwargs = self.init_kwargs.copy()
        kwargs["is_global_attention"] = True
        block = EmbeddingGemma2EncoderBlock(**kwargs)
        mask = block._compute_attention_mask(
            x=self.input_data,
            padding_mask=self.padding_mask,
            vision_mask=None,
            cache=None,
            cache_update_index=0,
        )
        # Full bidirectional attention within padding limits
        expected_mask_seq0 = [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        self.assertAllClose(mask[0], expected_mask_seq0)
