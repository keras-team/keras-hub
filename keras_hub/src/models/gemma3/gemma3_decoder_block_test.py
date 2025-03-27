import numpy as np

from keras_hub.src.models.gemma3.gemma3_decoder_block import Gemma3DecoderBlock
from keras_hub.src.tests.test_case import TestCase


class Gemma3DecoderBlockTest(TestCase):
    def setUp(self):
        self.batch_size = 3

        self.total_sequence_length = 10

        self.hidden_dim = 64

        self.decoder_block = Gemma3DecoderBlock(self.hidden_dim, 64, 64, 8, 8)

        self.dummy_input = np.random.rand(
            self.batch_size, self.total_sequence_length, self.hidden_dim
        )

    def _build_causal_mask(self):
        mask = np.zeros(
            (
                self.batch_size,
                self.total_sequence_length,
                self.total_sequence_length,
            )
        )
        for i in range(self.total_sequence_length):
            mask[:, i, : i + 1] = True
        return mask

    def test_gemma3_attention_mask_computation(self):
        attn_mask = self.decoder_block._compute_attention_mask(
            x=self.dummy_input,
            padding_mask=None,
            vision_mask=None,
            cache=None,
            cache_update_index=0,
        )
        expected_mask = self._build_causal_mask()
        self.assertAllEqual(
            expected_mask,
            attn_mask,
        )

    def test_gemma3_attention_mask_computation_with_vision_mask(self):
        vision_mask = np.full(
            (self.batch_size, self.total_sequence_length), False
        )
        vision_mask[0, 3:6] = True
        vision_mask[1, 2:5] = True
        vision_mask[1, 6:9] = True

        attn_mask = self.decoder_block._compute_attention_mask(
            x=self.dummy_input,
            padding_mask=None,
            vision_mask=vision_mask,
            cache=None,
            cache_update_index=0,
        )

        expected_mask = self._build_causal_mask()
        expected_mask[0, 3, 3:6] = True
        expected_mask[0, 4, 4:6] = True
        expected_mask[1, 2, 2:5] = True
        expected_mask[1, 3, 3:5] = True
        expected_mask[1, 4, 4:5] = True
        expected_mask[1, 6, 6:9] = True
        expected_mask[1, 7, 7:9] = True
        expected_mask[1, 8, 8:9] = True
        self.assertAllEqual(
            expected_mask,
            attn_mask,
        )

    def test_gemma3_attention_mask_computation_with_dual_masks(self):
        padding_mask = np.full(
            (self.batch_size, self.total_sequence_length), True
        )
        padding_mask[1, 6:] = False

        vision_mask = np.full(
            (self.batch_size, self.total_sequence_length), False
        )
        vision_mask[0, 3:6] = True
        vision_mask[1, 2:5] = True
        vision_mask[1, 6:9] = True

        attn_mask = self.decoder_block._compute_attention_mask(
            x=self.dummy_input,
            padding_mask=padding_mask,
            vision_mask=vision_mask,
            cache=None,
            cache_update_index=0,
        )
        expected_mask = self._build_causal_mask()
        expected_mask[0, 3, 3:6] = True
        expected_mask[0, 4, 4:6] = True
        expected_mask[1, 2, 2:5] = True
        expected_mask[1, 3, 3:5] = True
        expected_mask[1, 4, 4:5] = True

        for i in range(6, 10):
            expected_mask[1, i, 6:] = False

        print(expected_mask[1])
        print(attn_mask[1])

        self.assertAllEqual(
            expected_mask,
            attn_mask,
        )
