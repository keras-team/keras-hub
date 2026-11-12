import pytest

from keras_hub.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_causal_lm import GPTNeoXCausalLM
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_causal_lm_preprocessor import (
    GPTNeoXCausalLMPreprocessor,
)
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_tokenizer import GPTNeoXTokenizer
from keras_hub.src.tests.test_case import TestCase


class GPTNeoXCausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|endoftext|>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.preprocessor = GPTNeoXCausalLMPreprocessor(
            GPTNeoXTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=8,
        )
        self.vocabulary_size = self.preprocessor.tokenizer.vocabulary_size()
        self.backbone = GPTNeoXBackbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=self.preprocessor.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=GPTNeoXCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, self.vocabulary_size),
        )

    def test_generate(self):
        self.run_generate_test(
            cls=GPTNeoXCausalLM,
            init_kwargs=self.init_kwargs,
            prompt=" airplane at airport",
        )

    def test_early_stopping(self):
        self.run_early_stopping_test(
            cls=GPTNeoXCausalLM,
            init_kwargs=self.init_kwargs,
            prompt=[" airplane at airport", " airplane"],
        )

    def test_generate_compilation(self):
        self.run_generate_compilation_test(
            cls=GPTNeoXCausalLM,
            init_kwargs=self.init_kwargs,
            prompt=" airplane at airport",
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GPTNeoXCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=GPTNeoXCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-4}},
        )
