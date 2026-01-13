import keras
import pytest
from keras import ops

from keras_hub.src.models.edrec.edrec_backbone import EdRecBackbone
from keras_hub.src.models.edrec.edrec_seq2seq_lm import EdRecSeq2SeqLM
from keras_hub.src.tests.test_case import TestCase


class EdRecSeq2SeqLMTest(TestCase):
    def setUp(self):
        super().setUp()
        self.backbone = EdRecBackbone(
            vocab_size=10,
            num_layers_enc=2,
            num_layers_dec=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            dropout=0.0,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
        }
        self.input_data = {
            "encoder_token_ids": ops.ones((2, 5), dtype="int32"),
            "encoder_padding_mask": ops.zeros((2, 5), dtype="int32"),
            "decoder_token_ids": ops.ones((2, 5), dtype="int32"),
            "decoder_padding_mask": ops.zeros((2, 5), dtype="int32"),
        }

    def test_lm_basics(self):
        lm = EdRecSeq2SeqLM(**self.init_kwargs)
        lm.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        lm.fit(self.input_data, self.input_data["decoder_token_ids"])

    def test_generate(self):
        seq_2_seq_lm = EdRecSeq2SeqLM(**self.init_kwargs)
        seq_2_seq_lm.compile(sampler="greedy")

        # Test generation with dictionary input
        decoder_padding_mask = ops.zeros((2, 10), dtype="bool")
        decoder_padding_mask = ops.slice_update(
            decoder_padding_mask, [0, 0], ops.ones((2, 1), dtype="bool")
        )
        inputs = {
            "encoder_token_ids": ops.ones((2, 5), dtype="int32"),
            "encoder_padding_mask": ops.ones((2, 5), dtype="int32"),
            "decoder_token_ids": ops.zeros((2, 10), dtype="int32"),
            "decoder_padding_mask": decoder_padding_mask,
        }
        output = seq_2_seq_lm.generate(
            inputs, max_length=10, stop_token_ids=None
        )
        if isinstance(output, dict):
            output = output["decoder_token_ids"]
        self.assertEqual(output.shape, (2, 10))

    def test_beam_search(self):
        seq_2_seq_lm = EdRecSeq2SeqLM(**self.init_kwargs)
        seq_2_seq_lm.compile(sampler="beam")
        inputs = {
            "encoder_token_ids": ops.ones((2, 5), dtype="int32"),
            "encoder_padding_mask": ops.ones((2, 5), dtype="int32"),
        }
        seq_2_seq_lm.generate(inputs, max_length=10, stop_token_ids=None)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=EdRecSeq2SeqLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
