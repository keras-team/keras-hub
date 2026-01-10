from keras import ops
from keras_hub.src.models.edrec.edrec_backbone import EdRecBackbone
from keras_hub.src.models.edrec.edrec_seq2seq_lm import EdRecSeq2SeqLM
from keras_hub.src.tests.test_case import TestCase
import pytest


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
    self.run_task_test(
        cls=EdRecSeq2SeqLM,
        init_kwargs=self.init_kwargs,
        train_data=(self.input_data,),
        expected_output_shape=(2, 5, 10),  # B, L, Vocab
    )

  def test_generate(self):
    seq_2_seq_lm = EdRecSeq2SeqLM(**self.init_kwargs)

    # Test generation with dictionary input
    inputs = {
        "encoder_token_ids": ops.ones((2, 5), dtype="int32"),
        "encoder_padding_mask": ops.ones((2, 5), dtype="int32"),
    }
    output = seq_2_seq_lm.generate(inputs, max_length=10, stop_token_ids=None)
    # Check shape (B, 10) likely if just IDs, or maybe dict
    # Default generate returns just token IDs if preprocessor is None?
    # BartSeq2SeqLMTest passed string.
    # Here we pass dict.
    # It should return dict or just IDs?
    # Without preprocessor, it usually returns token IDs.
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
