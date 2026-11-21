from keras import ops

from keras_hub.src.layers.preprocessing.masked_lm_mask_generator import (
    MaskedLMMaskGenerator,
)
from keras_hub.src.tests.test_case import TestCase


class MaskedLMMaskGeneratorTest(TestCase):
    def setUp(self):
        super().setUp()
        self.VOCAB = [
            "[UNK]",
            "[MASK]",
            "[RANDOM]",
            "[CLS]",
            "[SEP]",
            "do",
            "you",
            "like",
            "machine",
            "learning",
            "welcome",
            "to",
            "keras",
        ]
        self.mask_token_id = self.VOCAB.index("[MASK]")
        self.vocabulary_size = len(self.VOCAB)

    def test_layer_basics(self):
        # `mask_selection_rate=1`, `mask_token_rate=1`, `random_token_rate=0`
        # force every random draw to the same outcome, so the eager and
        # dataset-mapped outputs are deterministically identical.
        self.run_preprocessing_layer_test(
            cls=MaskedLMMaskGenerator,
            init_kwargs={
                "vocabulary_size": self.vocabulary_size,
                "mask_selection_rate": 1,
                "mask_selection_length": 4,
                "mask_token_id": self.mask_token_id,
                "mask_token_rate": 1,
                "random_token_rate": 0,
            },
            input_data=[[5, 3, 2, 4], [1, 2, 3, 4]],
            expected_output={
                "token_ids": [[1, 1, 1, 1], [1, 1, 1, 1]],
                "mask_positions": [[0, 1, 2, 3], [0, 1, 2, 3]],
                "mask_ids": [[5, 3, 2, 4], [1, 2, 3, 4]],
                "mask_weights": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            },
        )

    def test_mask_ragged(self):
        masked_lm_masker = MaskedLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=1,
            mask_selection_length=4,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = [[5, 3, 2], [1, 2, 3, 4]]
        x = masked_lm_masker(inputs)
        self.assertAllEqual(x["token_ids"], [[1, 1, 1], [1, 1, 1, 1]])
        self.assertAllEqual(x["mask_positions"], [[0, 1, 2, 0], [0, 1, 2, 3]])
        self.assertAllEqual(x["mask_ids"], [[5, 3, 2, 0], [1, 2, 3, 4]])

    def test_unbatched(self):
        masked_lm_masker = MaskedLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=1,
            mask_selection_length=4,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = [5, 3, 2, 4]
        x = masked_lm_masker(inputs)
        self.assertAllEqual(x["token_ids"], [1, 1, 1, 1])
        self.assertAllEqual(x["mask_positions"], [0, 1, 2, 3])
        self.assertAllEqual(x["mask_ids"], [5, 3, 2, 4])

    def test_random_replacement(self):
        masked_lm_masker = MaskedLMMaskGenerator(
            vocabulary_size=10_000,
            mask_selection_rate=1,
            mask_selection_length=4,
            mask_token_id=self.mask_token_id,
            mask_token_rate=0,
            random_token_rate=1,
        )
        inputs = [5, 3, 2, 4]
        x = masked_lm_masker(inputs)
        self.assertNotAllEqual(x["token_ids"], [1, 1, 1, 1])
        self.assertAllEqual(x["mask_positions"], [0, 1, 2, 3])
        self.assertAllEqual(x["mask_ids"], [5, 3, 2, 4])

    def test_number_of_masked_position_as_expected(self):
        mask_selection_rate = 0.5
        mask_selection_length = 5
        inputs = [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]
        # Cap the number of masked tokens at 0, so we can test if
        # mask_selection_length takes effect.
        mask_selection_length = 0
        masked_lm_masker = MaskedLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=mask_selection_rate,
            mask_token_id=self.mask_token_id,
            mask_selection_length=mask_selection_length,
        )
        outputs = masked_lm_masker(inputs)
        self.assertEqual(ops.sum(outputs["mask_positions"]), 0)

    def test_invalid_mask_token(self):
        with self.assertRaisesRegex(ValueError, "Mask token id should be*"):
            _ = MaskedLMMaskGenerator(
                vocabulary_size=self.vocabulary_size,
                mask_selection_rate=0.5,
                mask_token_id=self.vocabulary_size,
                mask_selection_length=5,
            )

    def test_unselectable_tokens(self):
        unselectable_token_ids = [
            self.vocabulary_size - 1,
            self.vocabulary_size - 2,
        ]
        masked_lm_masker = MaskedLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=1,
            mask_token_id=self.mask_token_id,
            mask_selection_length=5,
            unselectable_token_ids=unselectable_token_ids,
            mask_token_rate=1,
            random_token_rate=0,
        )
        outputs = masked_lm_masker([unselectable_token_ids])
        # Verify that no token is masked out.
        self.assertEqual(ops.sum(outputs["mask_weights"]), 0)
