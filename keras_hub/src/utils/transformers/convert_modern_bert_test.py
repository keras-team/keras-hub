import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.masked_lm import MaskedLM
from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modern_bert.modern_bert_masked_lm import (
    ModernBertMaskedLM,
)
from keras_hub.src.tests.test_case import TestCase

HF_MODEL_ID = "answerdotai/ModernBERT-base"
HF_PRESET = f"hf://{HF_MODEL_ID}"

# `local_attention=128` gives a sliding-window radius of 64. A sequence
# shorter than that makes the window mask all-ones, so the local layers
# become indistinguishable from the global ones and the comparison silently
# stops covering the alternating-attention logic. 256 is the shortest length
# that exercises it.
SEQ_LEN = 256


class TestConvertModernBert(TestCase):
    @pytest.mark.extra_large
    def test_convert_modern_bert_base(self):
        """Compare backbone hidden states against HF.

        Loads through `from_preset` rather than hand-driving
        `SafetensorLoader`, so `preset_loader.py` dispatch and
        `convert_backbone_config` are both in the path.
        """
        import torch
        from transformers import AutoModel

        hf_model = AutoModel.from_pretrained(HF_MODEL_ID)
        hf_model.eval()

        backbone = ModernBertBackbone.from_preset(HF_PRESET)

        np.random.seed(42)
        batch_size = 2
        vocabulary_size = hf_model.config.vocab_size
        token_ids = np.random.randint(
            100, vocabulary_size - 100, size=(batch_size, SEQ_LEN)
        )
        padding_mask = np.ones((batch_size, SEQ_LEN), dtype=np.int64)

        with torch.no_grad():
            hf_outputs = (
                hf_model(
                    input_ids=torch.from_numpy(token_ids),
                    attention_mask=torch.from_numpy(padding_mask),
                )
                .last_hidden_state.cpu()
                .numpy()
            )

        keras_outputs = backbone(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask.astype(bool),
            }
        )

        if hasattr(keras_outputs, "numpy"):
            keras_outputs = keras_outputs.numpy()

        # Divergence grows with sequence length; ~5e-4 is expected at 256, so
        # this leaves headroom without being vacuous.
        self.assertAllClose(hf_outputs, keras_outputs, atol=2e-3, rtol=2e-3)

    @pytest.mark.extra_large
    def test_convert_masked_lm_end_to_end(self):
        """Fill-mask parity with the tokenizer in the loop.

        The backbone comparison above feeds identical ids to both models,
        which bypasses `ModernBertTokenizer` entirely. This test starts from
        raw text on both sides, so a tokenization difference (such as the
        `[MASK]` `lstrip` handling) shows up as a prediction mismatch.
        """
        import torch
        from transformers import AutoModelForMaskedLM
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        hf_model = AutoModelForMaskedLM.from_pretrained(HF_MODEL_ID)
        hf_model.eval()

        keras_model = ModernBertMaskedLM.from_preset(HF_PRESET)
        keras_tokenizer = keras_model.preprocessor.tokenizer

        prompts = [
            "The capital of France is [MASK].",
            "She went to the [MASK] to buy milk.",
            "[MASK] is the largest planet in our solar system.",
        ]

        for prompt in prompts:
            hf_inputs = hf_tokenizer(prompt, return_tensors="pt")
            keras_ids = [int(i) for i in keras_tokenizer([prompt])[0]]

            hf_body_ids = [
                i
                for i in hf_inputs["input_ids"][0].tolist()
                if i
                not in (
                    hf_tokenizer.cls_token_id,
                    hf_tokenizer.sep_token_id,
                )
            ]
            self.assertAllEqual(
                keras_ids,
                hf_body_ids,
                msg=f"Tokenization diverged for {prompt!r}",
            )

            with torch.no_grad():
                hf_logits = hf_model(**hf_inputs).logits

            hf_mask_position = (
                (hf_inputs["input_ids"][0] == hf_tokenizer.mask_token_id)
                .nonzero()[0]
                .item()
            )
            hf_top1 = int(hf_logits[0, hf_mask_position].argmax())

            # Drive the KerasHub forward pass from KerasHub's own
            # tokenization, so this stays an end-to-end check.
            keras_input_ids = np.array(
                [
                    [keras_tokenizer.cls_token_id]
                    + keras_ids
                    + [keras_tokenizer.sep_token_id]
                ]
            )
            keras_mask_position = int(
                np.argmax(keras_input_ids[0] == keras_tokenizer.mask_token_id)
            )

            keras_logits = keras_model(
                {
                    "token_ids": keras_input_ids,
                    "padding_mask": np.ones_like(keras_input_ids, dtype=bool),
                    "mask_positions": np.array([[keras_mask_position]]),
                }
            )
            keras_logits = np.asarray(keras_logits)
            keras_top1 = int(keras_logits[0, 0].argmax())

            self.assertEqual(
                keras_top1,
                hf_top1,
                msg=(
                    f"Fill-mask prediction diverged for {prompt!r}: "
                    f"KerasHub predicted "
                    f"{hf_tokenizer.decode(keras_top1)!r}, HF predicted "
                    f"{hf_tokenizer.decode(hf_top1)!r}"
                ),
            )

    @pytest.mark.large
    def test_class_detection(self):
        model = MaskedLM.from_preset(HF_PRESET, load_weights=False)
        self.assertIsInstance(model, ModernBertMaskedLM)

        model = Backbone.from_preset(HF_PRESET, load_weights=False)
        self.assertIsInstance(model, ModernBertBackbone)
