import tensorflow as tf
from absl.testing import parameterized

from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.tests.test_case import TestCase


class StartEndPackerTest(TestCase):
    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_dense_input(self, allow_python_workflow):
        # right padding
        input_data = [5, 6, 7]
        start_end_packer = StartEndPacker(
            sequence_length=5, _allow_python_workflow=allow_python_workflow
        )
        output = start_end_packer(input_data)
        expected_output = [5, 6, 7, 0, 0]
        self.assertAllEqual(output, expected_output)
        # left padding
        start_end_packer = StartEndPacker(
            sequence_length=5,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [0, 0, 5, 6, 7]
        self.assertAllEqual(output, expected_output)

    def test_bfloat16_dtype(self):
        # Core Keras has a strange bug where it converts int to floats in
        # ops.convert_to_tensor only with jax and bfloat16.
        input_data = [5, 6, 7]
        start_end_packer = StartEndPacker(sequence_length=5, dtype="bfloat16")
        output = start_end_packer(input_data)
        self.assertDTypeEqual(output, "int32")

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_dense_2D_input(self, allow_python_workflow):
        # right padding
        input_data = [[5, 6, 7]]
        start_end_packer = StartEndPacker(
            sequence_length=5, _allow_python_workflow=allow_python_workflow
        )
        output = start_end_packer(input_data)
        expected_output = [[5, 6, 7, 0, 0]]
        self.assertAllEqual(output, expected_output)
        # left padding
        start_end_packer = StartEndPacker(
            sequence_length=5,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[0, 0, 5, 6, 7]]
        self.assertAllEqual(output, expected_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_ragged_input(self, allow_python_workflow):
        # right padding
        input_data = [[5, 6, 7], [8, 9, 10, 11]]
        start_end_packer = StartEndPacker(
            sequence_length=5, _allow_python_workflow=allow_python_workflow
        )
        output = start_end_packer(input_data)
        expected_output = [[5, 6, 7, 0, 0], [8, 9, 10, 11, 0]]
        self.assertAllEqual(output, expected_output)
        # left padding
        start_end_packer = StartEndPacker(
            sequence_length=5,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]]
        self.assertAllEqual(output, expected_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_start_end_token(self, allow_python_workflow):
        # right padding
        input_data = [[5, 6, 7], [8, 9, 10, 11]]
        start_end_packer = StartEndPacker(
            sequence_length=6,
            start_value=1,
            end_value=2,
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[1, 5, 6, 7, 2, 0], [1, 8, 9, 10, 11, 2]]
        self.assertAllEqual(output, expected_output)
        # left padding
        start_end_packer = StartEndPacker(
            sequence_length=6,
            start_value=1,
            end_value=2,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[0, 1, 5, 6, 7, 2], [1, 8, 9, 10, 11, 2]]
        self.assertAllEqual(output, expected_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_multiple_start_end_tokens(self, allow_python_workflow):
        # right padding
        input_data = [[5, 6, 7], [8, 9, 10, 11, 12, 13]]
        start_end_packer = StartEndPacker(
            sequence_length=8,
            start_value=[1, 2],
            end_value=[3, 4],
            pad_value=0,
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[1, 2, 5, 6, 7, 3, 4, 0], [1, 2, 8, 9, 10, 11, 3, 4]]
        self.assertAllEqual(output, expected_output)

        # left padding
        start_end_packer = StartEndPacker(
            sequence_length=8,
            start_value=[1, 2],
            end_value=[3, 4],
            pad_value=0,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[0, 1, 2, 5, 6, 7, 3, 4], [1, 2, 8, 9, 10, 11, 3, 4]]
        self.assertAllEqual(output, expected_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_start_end_padding_value(self, allow_python_workflow):
        # right padding
        input_data = [[5, 6, 7], [8, 9, 10, 11]]
        start_end_packer = StartEndPacker(
            sequence_length=7,
            start_value=1,
            end_value=2,
            pad_value=3,
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[1, 5, 6, 7, 2, 3, 3], [1, 8, 9, 10, 11, 2, 3]]
        self.assertAllEqual(output, expected_output)

        # left padding
        start_end_packer = StartEndPacker(
            sequence_length=7,
            start_value=1,
            end_value=2,
            pad_value=3,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[3, 3, 1, 5, 6, 7, 2], [3, 1, 8, 9, 10, 11, 2]]
        self.assertAllEqual(output, expected_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_truncation(self, allow_python_workflow):
        # right padding
        input_data = list(range(10))
        packer = StartEndPacker(
            sequence_length=7,
            start_value=98,
            end_value=99,
            _allow_python_workflow=allow_python_workflow,
        )
        expected_output = [98, 0, 1, 2, 3, 4, 99]
        self.assertAllEqual(packer(input_data), expected_output)

        # left padding
        packer = StartEndPacker(
            sequence_length=7,
            start_value=98,
            end_value=99,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        self.assertAllEqual(packer(input_data), expected_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_truncation_wo_endvalue(self, allow_python_workflow):
        # right padding
        input_data = list(range(10))
        packer = StartEndPacker(
            sequence_length=7,
            start_value=98,
            _allow_python_workflow=allow_python_workflow,
        )
        expected_output = [98, 0, 1, 2, 3, 4, 5]
        self.assertAllEqual(packer(input_data), expected_output)

        # left padding
        packer = StartEndPacker(
            sequence_length=7,
            start_value=98,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        self.assertAllEqual(packer(input_data), expected_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_end_token_value_during_truncation(self, allow_python_workflow):
        # right padding
        input_data = [[5, 6], [8, 9, 10, 11, 12, 13]]
        start_end_packer = StartEndPacker(
            sequence_length=5,
            start_value=1,
            end_value=2,
            pad_value=0,
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[1, 5, 6, 2, 0], [1, 8, 9, 10, 2]]
        self.assertAllEqual(output, expected_output)

        # left padding
        start_end_packer = StartEndPacker(
            sequence_length=5,
            start_value=1,
            end_value=2,
            pad_value=0,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [[0, 1, 5, 6, 2], [1, 8, 9, 10, 2]]
        self.assertAllEqual(output, expected_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_string_input(self, allow_python_workflow):
        # right padding
        input_data = [["KerasHub", "is", "awesome"], ["amazing"]]
        start_end_packer = StartEndPacker(
            sequence_length=5,
            start_value="[START]",
            end_value="[END]",
            pad_value="[PAD]",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [
            ["[START]", "KerasHub", "is", "awesome", "[END]"],
            ["[START]", "amazing", "[END]", "[PAD]", "[PAD]"],
        ]
        self.assertAllEqual(output, expected_output)

        # left padding
        start_end_packer = StartEndPacker(
            sequence_length=5,
            start_value="[START]",
            end_value="[END]",
            pad_value="[PAD]",
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [
            ["[START]", "KerasHub", "is", "awesome", "[END]"],
            ["[PAD]", "[PAD]", "[START]", "amazing", "[END]"],
        ]
        self.assertAllEqual(output, expected_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_string_input_with_multiple_special_values(
        self, allow_python_workflow
    ):
        # right padding
        input_data = [["KerasHub", "is", "awesome"], ["amazing"]]
        start_end_packer = StartEndPacker(
            sequence_length=6,
            start_value=["[END]", "[START]"],
            end_value="[END]",
            pad_value="[PAD]",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [
            ["[END]", "[START]", "KerasHub", "is", "awesome", "[END]"],
            ["[END]", "[START]", "amazing", "[END]", "[PAD]", "[PAD]"],
        ]
        self.assertAllEqual(output, expected_output)

        # left padding
        start_end_packer = StartEndPacker(
            sequence_length=6,
            start_value=["[END]", "[START]"],
            end_value="[END]",
            pad_value="[PAD]",
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output = start_end_packer(input_data)
        expected_output = [
            ["[END]", "[START]", "KerasHub", "is", "awesome", "[END]"],
            ["[PAD]", "[PAD]", "[END]", "[START]", "amazing", "[END]"],
        ]
        self.assertAllEqual(output, expected_output)

    def test_special_token_dtype_error(self):
        with self.assertRaises(ValueError):
            StartEndPacker(sequence_length=5, start_value=1.0)

    def test_batch(self):
        start_end_packer = StartEndPacker(
            sequence_length=7, start_value=1, end_value=2, pad_value=3
        )

        ds = tf.data.Dataset.from_tensor_slices(
            tf.ragged.constant([[5, 6, 7], [8, 9, 10, 11]])
        )
        ds = ds.batch(2).map(start_end_packer)
        output = ds.take(1).get_single_element()

        exp_output = [[1, 5, 6, 7, 2, 3, 3], [1, 8, 9, 10, 11, 2, 3]]
        self.assertAllEqual(output, exp_output)

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_call_overrides(self, allow_python_workflow):
        x = [5, 6, 7]
        packer = StartEndPacker(
            start_value=1,
            end_value=2,
            sequence_length=4,
            _allow_python_workflow=allow_python_workflow,
        )
        self.assertAllEqual(packer(x), [1, 5, 6, 2])
        self.assertAllEqual(packer(x, add_start_value=False), [5, 6, 7, 2])
        self.assertAllEqual(packer(x, add_end_value=False), [1, 5, 6, 7])
        self.assertAllEqual(packer(x, sequence_length=2), [1, 2])

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_get_config(self, allow_python_workflow):
        start_end_packer = StartEndPacker(
            sequence_length=512,
            start_value=10,
            end_value=20,
            pad_value=100,
            name="start_end_packer_test",
            _allow_python_workflow=allow_python_workflow,
        )

        config = start_end_packer.get_config()
        expected_config_subset = {
            "sequence_length": 512,
            "start_value": 10,
            "end_value": 20,
            "pad_value": 100,
        }

        self.assertEqual(config, {**config, **expected_config_subset})

    @parameterized.named_parameters(
        ("allow_python_workflow", True),
        ("disallow_python_workflow", False),
    )
    def test_return_padding_mask(self, allow_python_workflow):
        # right_padding
        input_data = [[5, 6, 7], [8, 9, 10, 11]]
        start_end_packer = StartEndPacker(
            sequence_length=6,
            start_value=1,
            end_value=2,
            return_padding_mask=True,
            _allow_python_workflow=allow_python_workflow,
        )
        output, padding_mask = start_end_packer(input_data)
        expected_output = [[1, 5, 6, 7, 2, 0], [1, 8, 9, 10, 11, 2]]
        expected_padding_mask = [
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
        self.assertAllEqual(output, expected_output)
        self.assertAllEqual(padding_mask, expected_padding_mask)

        # left_padding
        start_end_packer = StartEndPacker(
            sequence_length=6,
            start_value=1,
            end_value=2,
            return_padding_mask=True,
            padding_side="left",
            _allow_python_workflow=allow_python_workflow,
        )
        output, padding_mask = start_end_packer(input_data)
        expected_output = [[0, 1, 5, 6, 7, 2], [1, 8, 9, 10, 11, 2]]
        expected_padding_mask = [
            [False, True, True, True, True, True],
            [True, True, True, True, True, True],
        ]
        self.assertAllEqual(output, expected_output)
        self.assertAllEqual(padding_mask, expected_padding_mask)
