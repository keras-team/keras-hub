import keras
import numpy as np
from absl.testing import parameterized
from scipy.optimize import linear_sum_assignment

from keras_hub.src.models.d_fine.d_fine_utils import hungarian_assignment
from keras_hub.src.tests.test_case import TestCase


class DFineUtilsTest(TestCase):
    @parameterized.named_parameters(
        (
            "square_matrix",
            np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype="float32"),
        ),
        (
            "rectangular_more_rows",
            np.array(
                [[10, 20, 30], [40, 50, 5], [15, 25, 35], [5, 10, 15]],
                dtype="float32",
            ),
        ),
        (
            "rectangular_more_cols",
            np.array(
                [[10, 20, 30, 40], [50, 5, 15, 25], [35, 45, 55, 65]],
                dtype="float32",
            ),
        ),
        (
            "duplicate_min_costs",
            np.array([[1, 1, 2], [2, 3, 1], [3, 1, 4]], dtype="float32"),
        ),
        (
            "larger_matrix",
            np.array(
                [
                    [9, 2, 7, 8, 4],
                    [6, 4, 3, 7, 5],
                    [5, 8, 1, 8, 2],
                    [7, 6, 9, 4, 1],
                    [3, 5, 8, 5, 4],
                ],
                dtype="float32",
            ),
        ),
    )
    def test_hungarian_assignment_equivalence(self, cost_matrix):
        # Test if the Keras version is equivalent to SciPy's
        # `optimize.linear_sum_assignment`.
        num_queries = max(cost_matrix.shape)
        keras_row_ind, keras_col_ind, keras_valid_mask = hungarian_assignment(
            keras.ops.convert_to_tensor(cost_matrix),
            num_queries,
        )
        scipy_row_ind, scipy_col_ind = linear_sum_assignment(cost_matrix)
        scipy_cost = cost_matrix[scipy_row_ind, scipy_col_ind].sum()
        valid_row_ind = keras.ops.convert_to_numpy(
            keras_row_ind[keras_valid_mask]
        )
        valid_col_ind = keras.ops.convert_to_numpy(
            keras_col_ind[keras_valid_mask]
        )
        keras_cost = cost_matrix[valid_row_ind, valid_col_ind].sum()
        self.assertAllClose(keras_cost, scipy_cost)
