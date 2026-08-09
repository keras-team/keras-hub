from absl.testing import parameterized
from keras import ops
from keras import random
from keras.src.backend import get_keras_mask
from keras.src.backend import set_keras_mask

from keras_hub.src.layers.modeling.transformer_decoder import TransformerDecoder
from keras_hub.src.tests.test_case import TestCase


class TransformerDecoderTest(TestCase):
    @parameterized.named_parameters(
        ("without_norm_first", False),
        ("with_norm_first", True),
    )
    def test_layer_behaviors(self, normalize_first):
        self.run_layer_test(
            cls=TransformerDecoder,
            init_kwargs={
                "intermediate_dim": 4,
                "num_heads": 2,
                "normalize_first": normalize_first,
                "activation": "relu",
                "layer_norm_epsilon": 1e-05,
                "kernel_initializer": "HeNormal",
                "bias_initializer": "Zeros",
                "dropout": 0.1,
            },
            input_data=random.uniform(shape=(2, 4, 6)),
            expected_output_shape=(2, 4, 6),
            expected_num_trainable_weights=16,
            expected_num_non_trainable_variables=3,  # dropout rng seeds
        )

    @parameterized.named_parameters(
        ("without_norm_first", False),
        ("with_norm_first", True),
    )
    def test_layer_behaviors_with_cross_attention(self, normalize_first):
        self.run_layer_test(
            cls=TransformerDecoder,
            init_kwargs={
                "intermediate_dim": 4,
                "num_heads": 2,
                "normalize_first": normalize_first,
                "activation": "relu",
                "layer_norm_epsilon": 1e-05,
                "kernel_initializer": "HeNormal",
                "bias_initializer": "Zeros",
                "dropout": 0.1,
            },
            input_data={
                "decoder_sequence": random.uniform(shape=(2, 4, 6)),
                "encoder_sequence": random.uniform(shape=(2, 4, 6)),
            },
            expected_output_shape=(2, 4, 6),
            expected_num_trainable_weights=26,
            expected_num_non_trainable_variables=5,  # dropout rng seeds
        )

    def test_invalid_calls(self):
        encoder_input = ops.zeros((2, 4, 6))
        decoder_input = ops.zeros((2, 4, 6))

        # with cross-attention.
        decoder = TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        decoder(decoder_input, encoder_input)
        # should raise ValueError if encoder_input is not provided
        with self.assertRaises(ValueError):
            decoder(decoder_input)

        # without cross-attention.
        decoder = TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        decoder(decoder_input)
        # should raise ValueError if encoder_input is provided
        with self.assertRaises(ValueError):
            decoder(decoder_input, encoder_input)

    def test_value_error_when_invalid_kernel_inititalizer(self):
        with self.assertRaises(ValueError):
            TransformerDecoder(
                intermediate_dim=4,
                num_heads=2,
                dropout=0.5,
                kernel_initializer="Invalid",
            )

    def test_training_propagation(self):
        decoder = TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
            dropout=0.99999,  # Zeros out the outputs after the dropout layer
        )
        decoder_sequence = random.uniform(shape=[1, 4, 6])
        encoder_sequence = random.uniform(shape=[1, 4, 6])
        outputs = decoder(decoder_sequence, encoder_sequence, training=True)

        # Custom computation with dropout rates set to about 1.0
        x = decoder_sequence
        x = decoder._self_attention_layer_norm(x)
        x = decoder._feedforward_layer_norm(x)

        self.assertAllClose(outputs, x, atol=1e-5)

    def test_mask_propagation(self):
        decoder = TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        decoder_sequence = random.uniform(shape=[1, 4, 6])
        encoder_sequence = random.uniform(shape=[1, 4, 6])
        mask = ops.array([[True, True, False, False]])
        set_keras_mask(decoder_sequence, mask)
        outputs = decoder(decoder_sequence, encoder_sequence)
        self.assertAllEqual(get_keras_mask(outputs), mask)

    def test_mask_propagation_without_cross_attention(self):
        decoder = TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        decoder_sequence = random.uniform(shape=[1, 4, 6])
        mask = ops.array([[True, True, False, False]])
        set_keras_mask(decoder_sequence, mask)
        outputs = decoder(decoder_sequence)
        self.assertAllEqual(get_keras_mask(outputs), mask)

    def test_cache_call_is_correct(self):
        batch_size, seq_len, num_heads, key_dim = 2, 5, 2, 4
        hidden_dim = num_heads * key_dim

        input_shape = (batch_size, seq_len, hidden_dim)
        x = random.uniform(shape=input_shape)
        input_cache = ops.zeros((batch_size, 2, seq_len, num_heads, key_dim))
        outputs = ops.zeros_like(x)

        layer = TransformerDecoder(
            intermediate_dim=4,
            num_heads=num_heads,
        )
        no_loop_outputs, no_loop_cache = layer(
            x,
            self_attention_cache=input_cache,
            self_attention_cache_update_index=0,
        )

        def loop_body(i, outputs, cache):
            # Compute the rest tokens.
            next_input = ops.slice(x, (0, i, 0), (batch_size, 1, hidden_dim))
            next_output, cache = layer(
                decoder_sequence=next_input,
                self_attention_cache=cache,
                self_attention_cache_update_index=i,
            )
            outputs = ops.slice_update(outputs, [0, i, 0], next_output)
            return i + 1, outputs, cache

        def call(outputs, cache):
            _, outputs, cache = ops.while_loop(
                cond=lambda i, outputs, cache: i < seq_len,
                body=loop_body,
                loop_vars=[0, outputs, cache],
            )
            return outputs, cache

        output, output_cache = call(outputs, input_cache)
        self.assertAllClose(output, no_loop_outputs)
        self.assertAllClose(output_cache, no_loop_cache)

    def test_cache_call_is_correct_with_cross_attention(self):
        batch_size, seq_len, num_heads, key_dim = 2, 5, 2, 4
        hidden_dim = num_heads * key_dim

        input_shape = (batch_size, seq_len, hidden_dim)
        cache_shape = (batch_size, 2, seq_len, num_heads, key_dim)
        decoder_sequence = random.uniform(shape=input_shape)
        encoder_sequence = random.uniform(shape=input_shape)
        empty_cache = ops.zeros(cache_shape)
        outputs = ops.zeros_like(decoder_sequence)

        layer = TransformerDecoder(
            intermediate_dim=4,
            num_heads=num_heads,
        )
        no_loop_outputs, no_loop_self_cache, no_loop_cross_cache = layer(
            decoder_sequence,
            encoder_sequence,
            self_attention_cache=empty_cache,
            self_attention_cache_update_index=0,
            cross_attention_cache=empty_cache,
            cross_attention_cache_update_index=0,
        )

        def loop_body(i, outputs, self_cache, cross_cache):
            # Compute the rest tokens.
            start, size = (0, i, 0), (batch_size, 1, hidden_dim)
            next_input = ops.slice(decoder_sequence, start, size)
            next_output, self_cache, cross_cache = layer(
                decoder_sequence=next_input,
                encoder_sequence=encoder_sequence,
                self_attention_cache=self_cache,
                self_attention_cache_update_index=i,
                cross_attention_cache=cross_cache,
            )
            outputs = ops.slice_update(outputs, start, next_output)
            return i + 1, outputs, self_cache, cross_cache

        def call(outputs, self_cache, cross_cache):
            _, outputs, self_cache, cross_cache = ops.while_loop(
                cond=lambda i, outputs, self_cache, cross_cache: i < seq_len,
                body=loop_body,
                loop_vars=[0, outputs, self_cache, cross_cache],
            )
            return outputs, self_cache, cross_cache

        output, self_cache, cross_cache = call(
            outputs, empty_cache, no_loop_cross_cache
        )
        self.assertAllClose(output, no_loop_outputs)
        self.assertAllClose(self_cache, no_loop_self_cache)
        self.assertAllClose(cross_cache, no_loop_cross_cache)
