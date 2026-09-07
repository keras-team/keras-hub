import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_causal_lm_preprocessor import (  # noqa: E501
    MuseGlimmerCausalLMPreprocessor,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    MuseGlimmerImageConverter,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_tokenizer import (
    MuseGlimmerTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|end_of_text|>", "<|begin_of_text|>"]
        self.vocab += ["<|finetune_right_pad|>"]
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = MuseGlimmerTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.init_kwargs = {"tokenizer": self.tokenizer, "sequence_length": 8}

    def test_text_only_generate_preprocess(self):
        preprocessor = MuseGlimmerCausalLMPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(" airplane at airport")
        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)

    def test_image_generate_preprocess_expands_placeholder(self):
        image_converter = MuseGlimmerImageConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            max_image_tokens=64,
            scale=1 / 255.0,
        )
        preprocessor = MuseGlimmerCausalLMPreprocessor(
            **self.init_kwargs, image_converter=image_converter
        )
        # Build a prompt with a single image placeholder token id directly.
        image = np.random.randint(0, 255, (8, 8, 3)).astype("float32")
        result = image_converter(image)
        t, h, w = (int(v) for v in result["grid_thw"])
        num_merged_tokens = t * (h // 2) * (w // 2)

        ids = preprocessor._expand_vision_placeholders(
            [preprocessor.tokenizer.image_token_id, 5, 6],
            [num_merged_tokens],
            [],
        )
        self.assertEqual(
            ids,
            [preprocessor.tokenizer.image_token_id] * num_merged_tokens
            + [5, 6],
        )
