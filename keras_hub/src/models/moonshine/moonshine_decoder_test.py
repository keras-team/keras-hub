import keras

from keras_hub.src.models.moonshine.moonshine_decoder import (
    MoonshineDecoderBlock,
)
from keras_hub.src.tests.test_case import TestCase


# TODO: Enable test case and remove debugging code.
class MoonshineDecoderTest(TestCase):
    def setUp(self):
        super().setUp()
        self.hidden_dim = 64
        self.intermediate_dim = 256
        self.num_heads = 4
        self.decoder_block = MoonshineDecoderBlock(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_heads=self.num_heads,
            feedforward_expansion_factor=4,
            use_swiglu_activation=True,
            pad_head_dim_to_multiple_of=None,
            initializer_range=0.02,
            attention_bias=False,
            attention_dropout=0.0,
        )
        self.batch_size = 2
        self.seq_len = 10
        self.encoder_seq_len = 16
        self.head_dim = self.hidden_dim // self.num_heads  # 16
        self.rotary_dim = int(
            self.head_dim * 0.62
        )  # Default partial_rotary_factor = 0.62
        self.rotary_dim = (self.rotary_dim // 2) * 2  # Ensure even
        self.rotary_dim = self.rotary_dim // 2  # Half for freqs, e.g., 4
        self.x = keras.random.normal(
            (self.batch_size, self.seq_len, self.hidden_dim)
        )
        self.context = keras.random.normal(
            (self.batch_size, self.encoder_seq_len, self.hidden_dim)
        )
        self.rotary_embedding = keras.random.normal(
            (self.seq_len, self.rotary_dim)
        )
        self.decoder_attention_mask = keras.ops.ones(
            (self.batch_size, self.seq_len), dtype="bool"
        )
        self.encoder_attention_mask = keras.ops.ones(
            (self.batch_size, self.encoder_seq_len), dtype="bool"
        )

    def test_initialization(self):
        self.assertEqual(self.decoder_block.hidden_dim, self.hidden_dim)
        self.assertEqual(
            self.decoder_block.intermediate_dim, self.intermediate_dim
        )
        self.assertEqual(self.decoder_block.num_heads, self.num_heads)
        self.assertTrue(self.decoder_block.use_swiglu_activation)

    def test_forward_pass_without_caching(self):
        outputs = self.decoder_block(
            [self.x, self.context, self.rotary_embedding],
            decoder_attention_mask=self.decoder_attention_mask,
            encoder_attention_mask=self.encoder_attention_mask,
        )
        x, cache_k, cache_v, x_attn_cache_k, x_attn_cache_v = outputs
        self.assertEqual(
            x.shape, (self.batch_size, self.seq_len, self.hidden_dim)
        )
        self.assertEqual(
            cache_k.shape,
            (self.batch_size, self.seq_len, self.num_heads, self.head_dim),
        )
        self.assertEqual(
            cache_v.shape,
            (self.batch_size, self.seq_len, self.num_heads, self.head_dim),
        )
        self.assertEqual(
            x_attn_cache_k.shape,
            (
                self.batch_size,
                self.encoder_seq_len,
                self.num_heads,
                self.head_dim,
            ),
        )
        self.assertEqual(
            x_attn_cache_v.shape,
            (
                self.batch_size,
                self.encoder_seq_len,
                self.num_heads,
                self.head_dim,
            ),
        )

    def test_forward_pass_with_padding(self):
        # Padding in decoder sequence.
        padded_mask = keras.ops.concatenate(
            [
                keras.ops.ones((self.batch_size, 5), dtype="bool"),
                keras.ops.zeros(
                    (self.batch_size, self.seq_len - 5), dtype="bool"
                ),
            ],
            axis=1,
        )
        outputs = self.decoder_block(
            [self.x, self.context, self.rotary_embedding],
            decoder_attention_mask=padded_mask,
            encoder_attention_mask=self.encoder_attention_mask,
        )
        x, _, _, _, _ = outputs
        self.assertEqual(
            x.shape, (self.batch_size, self.seq_len, self.hidden_dim)
        )

    def test_autoregressive_caching(self):
        # First pass to get initial caches.
        outputs_full = self.decoder_block(
            [self.x, self.context, self.rotary_embedding],
            decoder_attention_mask=self.decoder_attention_mask,
            encoder_attention_mask=self.encoder_attention_mask,
        )
        _, cache_k_full, cache_v_full, x_attn_cache_k, x_attn_cache_v = (
            outputs_full
        )

        # Autoregressive decoding.
        for i in range(self.seq_len):
            x_i = self.x[:, i : i + 1, :]
            rotary_i = self.rotary_embedding[i : i + 1, :]
            mask_i = self.decoder_attention_mask[:, i : i + 1]
            cache_k = None if i == 0 else cache_k_full[:, :i, :, :]
            cache_v = None if i == 0 else cache_v_full[:, :i, :, :]
            outputs_i = self.decoder_block(
                [
                    x_i,
                    self.context,
                    cache_k,
                    cache_v,
                    x_attn_cache_k,
                    x_attn_cache_v,
                    rotary_i,
                ],
                use_cache=True,
                decoder_attention_mask=mask_i,
                encoder_attention_mask=self.encoder_attention_mask,
            )
            x_i_out, new_cache_k, new_cache_v = outputs_i
            self.assertEqual(
                x_i_out.shape, (self.batch_size, 1, self.hidden_dim)
            )
            self.assertEqual(
                new_cache_k.shape,
                (self.batch_size, i + 1, self.num_heads, self.head_dim),
            )
            self.assertEqual(
                new_cache_v.shape,
                (self.batch_size, i + 1, self.num_heads, self.head_dim),
            )

    def test_caching_consistency(self):
        # Full sequence without caching.
        outputs_full = self.decoder_block(
            [self.x, self.context, self.rotary_embedding],
            decoder_attention_mask=self.decoder_attention_mask,
            encoder_attention_mask=self.encoder_attention_mask,
        )
        x_full, _, _, _, _ = outputs_full

        # Autoregressive with caching.
        x_auto = []
        cache_k, cache_v = None, None
        x_attn_cache_k, x_attn_cache_v = (
            outputs_full[3],
            outputs_full[4],
        )  # Precomputed cross-attention caches
        for i in range(self.seq_len):
            x_i = self.x[:, i : i + 1, :]
            rotary_i = self.rotary_embedding[i : i + 1, :]
            mask_i = self.decoder_attention_mask[:, i : i + 1]
            outputs_i = self.decoder_block(
                [
                    x_i,
                    self.context,
                    cache_k,
                    cache_v,
                    x_attn_cache_k,
                    x_attn_cache_v,
                    rotary_i,
                ],
                use_cache=True,
                decoder_attention_mask=mask_i,
                encoder_attention_mask=self.encoder_attention_mask,
            )
            x_i_out, cache_k, cache_v = outputs_i
            x_auto.append(x_i_out)
        x_auto = keras.ops.concatenate(x_auto, axis=1)
        self.assertAllClose(x_full, x_auto, atol=1e-5)

    def test_serialization(self):
        config = self.decoder_block.get_config()
        new_decoder_block = MoonshineDecoderBlock.from_config(config)
        self.assertEqual(new_decoder_block.hidden_dim, self.hidden_dim)
        self.assertEqual(
            new_decoder_block.intermediate_dim, self.intermediate_dim
        )
        self.assertEqual(new_decoder_block.num_heads, self.num_heads)
        self.assertEqual(new_decoder_block.use_swiglu_activation, True)
