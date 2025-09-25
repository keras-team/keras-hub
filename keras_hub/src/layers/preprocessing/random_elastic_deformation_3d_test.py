# Add keras.utils for the random seed
from keras import utils
import os
import numpy as np
import keras
from keras import Model
from keras import Input
from keras import ops
from keras_hub.src.layers.preprocessing.random_elastic_deformation_3d import RandomElasticDeformation3D
from keras_hub.src.tests.test_case import TestCase


class RandomElasticDeformation3DTest(TestCase):
    def test_layer_basics(self):
        # --- BEST PRACTICE: Add a seed for reproducibility ---
        utils.set_random_seed(0)
        layer = RandomElasticDeformation3D(
            grid_size=(4, 4, 4),
            alpha=10.0,
            sigma=2.0,
        )
        image = ops.ones((2, 32, 32, 32, 3), dtype="float32")
        label = ops.ones((2, 32, 32, 32, 1), dtype="int32")
        output_image, output_label = layer((image, label))
        self.assertEqual(ops.shape(image), ops.shape(output_image))
        self.assertEqual(ops.shape(label), ops.shape(output_label))
        self.assertEqual(image.dtype, output_image.dtype)
        self.assertEqual(label.dtype, output_label.dtype)

    def test_serialization(self):
        # --- BEST PRACTICE: Add a seed for reproducibility ---
        utils.set_random_seed(0)
        layer = RandomElasticDeformation3D(
            grid_size=(3, 3, 3),
            alpha=50.0,
            sigma=5.0,
        )
        image_data = ops.ones((2, 16, 16, 16, 3), dtype="float32")
        label_data = ops.ones((2, 16, 16, 16, 1), dtype="int32")
        input_data = (image_data, label_data)
        image_input = Input(shape=(16, 16, 16, 3), dtype="float32")
        label_input = Input(shape=(16, 16, 16, 1), dtype="int32")
        outputs = layer((image_input, label_input))
        model = Model(inputs=[image_input, label_input], outputs=outputs)
        original_output_image, original_output_label = model(input_data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        
        # --- FIX: Remove the deprecated save_format argument ---
        model.save(path)
        
        loaded_model = keras.models.load_model(
            path, custom_objects={"RandomElasticDeformation3D": RandomElasticDeformation3D}
        )
        loaded_output_image, loaded_output_label = loaded_model(input_data)
        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output_image),
            ops.convert_to_numpy(loaded_output_image),
        )
        np.testing.assert_array_equal(
            ops.convert_to_numpy(original_output_label),
            ops.convert_to_numpy(loaded_output_label),
        )

    def test_label_values_are_preserved(self):
        # --- BEST PRACTICE: Add a seed for reproducibility ---
        utils.set_random_seed(0)
        image = ops.zeros(shape=(1, 16, 16, 16, 1), dtype="float32")
        label_arange = ops.arange(16**3, dtype="int32")
        label = ops.reshape(label_arange, (1, 16, 16, 16, 1)) % 4
        layer = RandomElasticDeformation3D(alpha=80.0, sigma=8.0)
        _, output_label = layer((image, label))
        output_values = set(np.unique(ops.convert_to_numpy(output_label)))
        expected_values = {0, 1, 2, 3}
        self.assertLessEqual(output_values, expected_values)