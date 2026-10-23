import pytest

from keras_hub.src.models.bloom.bloom_backbone import BloomBackbone
from keras_hub.src.models.bloom.bloom_causal_lm import BloomCausalLM
from keras_hub.src.models.bloom.bloom_causal_lm_preprocessor import (
    BloomCausalLMPreprocessor,
)
from keras_hub.src.models.bloom.bloom_tokenizer import BloomTokenizer
from keras_hub.src.tests.test_case import TestCase


class BloomCausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<unk>", "<s>", "</s>", "<pad>", "!"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = BloomTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.preprocessor = BloomCausalLMPreprocessor(
            self.tokenizer,
            sequence_length=8,
        )
        self.vocabulary_size = self.preprocessor.tokenizer.vocabulary_size()
        self.backbone = BloomBackbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=16,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            [
                " airplane at airport",
                " airplane airport",
            ],
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=BloomCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, self.vocabulary_size),
        )

    def test_generate(self):
        self.run_generate_test(
            cls=BloomCausalLM,
            init_kwargs=self.init_kwargs,
            prompt="airplane at airport",
            num_prompt_tokens=4,
        )

    def test_generate_with_bfloat16(self):
        backbone = BloomBackbone.from_config(
            {**self.backbone.get_config(), "dtype": "bfloat16"}
        )
        causal_lm = BloomCausalLM(
            backbone=backbone, preprocessor=self.preprocessor
        )
        # String input.
        prompt = "airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["token_ids"][:, :4],
            prompt_ids["token_ids"][:, :4],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :4],
            prompt_ids["padding_mask"][:, :4],
        )

    def test_generate_with_mixed_float16(self):
        backbone = BloomBackbone.from_config(
            {**self.backbone.get_config(), "dtype": "mixed_float16"}
        )
        causal_lm = BloomCausalLM(
            backbone=backbone, preprocessor=self.preprocessor
        )
        # String input.
        prompt = "airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["token_ids"][:, :4],
            prompt_ids["token_ids"][:, :4],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :4],
            prompt_ids["padding_mask"][:, :4],
        )

    def test_early_stopping(self):
        self.run_early_stopping_test(
            cls=BloomCausalLM,
            init_kwargs=self.init_kwargs,
            prompt=["airplane at", "airplane"],
        )

    def test_generate_compilation(self):
        self.run_generate_compilation_test(
            cls=BloomCausalLM,
            init_kwargs=self.init_kwargs,
            prompt="airplane at airport",
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BloomCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=BloomCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BloomCausalLM.presets:
            self.run_preset_test(
                cls=BloomCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
