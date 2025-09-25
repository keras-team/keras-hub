import tensorflow as tf
from tensorflow import keras
from keras_hub.src.layers.preprocessing.random_elastic_deformation_3d import RandomElasticDeformation3D

class RandomElasticDeformation3DTest(tf.test.TestCase):

    def test_output_shape_is_same_as_input_dhwc(self):
        input_image = tf.random.uniform(shape=(2, 32, 64, 64, 3), dtype=tf.float32)
        input_label = tf.random.uniform(shape=(2, 32, 64, 64, 1), maxval=4, dtype=tf.int32)
        layer = RandomElasticDeformation3D(data_format="DHWC")
        output_image, output_label = layer((input_image, tf.cast(input_label, tf.float32)))
        self.assertAllEqual(tf.shape(input_image), tf.shape(output_image))
        self.assertAllEqual(tf.shape(input_label), tf.shape(output_label))

    def test_output_shape_is_same_as_input_hwdc(self):
        input_image = tf.random.uniform(shape=(2, 64, 64, 32, 3), dtype=tf.float32)
        input_label = tf.random.uniform(shape=(2, 64, 64, 32, 1), maxval=4, dtype=tf.int32)
        layer = RandomElasticDeformation3D(data_format="HWDC")
        output_image, output_label = layer((input_image, tf.cast(input_label, tf.float32)))
        self.assertAllEqual(tf.shape(input_image), tf.shape(output_image))
        self.assertAllEqual(tf.shape(input_label), tf.shape(output_label))

    def test_unbatched_input(self):
        input_image = tf.random.uniform(shape=(32, 64, 64, 3), dtype=tf.float32)
        input_label = tf.random.uniform(shape=(32, 64, 64, 1), maxval=4, dtype=tf.int32)
        layer = RandomElasticDeformation3D(data_format="DHWC")
        output_image, output_label = layer((input_image, tf.cast(input_label, tf.float32)))
        self.assertAllEqual(tf.shape(input_image), tf.shape(output_image))
        self.assertEqual(tf.rank(output_image), 4)

    def test_dtype_preservation(self):
        input_image = tf.random.uniform(shape=(2, 16, 16, 16, 3), dtype=tf.float32)
        input_label = tf.random.uniform(shape=(2, 16, 16, 16, 1), maxval=4, dtype=tf.int32)
        layer = RandomElasticDeformation3D()
        output_image, output_label = layer((input_image, tf.cast(input_label, tf.float32)))
        self.assertEqual(output_image.dtype, tf.float32)
        self.assertEqual(output_label.dtype, tf.float32)

    def test_label_values_are_preserved(self):
        input_image = tf.zeros(shape=(1, 16, 16, 16, 1), dtype=tf.float32)
        label_arange = tf.experimental.numpy.arange(16**3)
        input_label = tf.reshape(label_arange, (1, 16, 16, 16, 1))
        input_label = tf.cast(input_label, dtype=tf.float32) % 4
        
        layer = RandomElasticDeformation3D(alpha=80.0, sigma=8.0)
        _, output_label = layer((input_image, input_label))
        
        unique_values_tensor = tf.unique(tf.reshape(output_label, [-1]))[0]
        

        expected_values = [0., 1., 2., 3.]
        actual_values = unique_values_tensor.numpy().tolist()
        self.assertContainsSubset(expected_values, actual_values)
        
    def test_config_serialization(self):
        layer = RandomElasticDeformation3D(
            grid_size=(3, 3, 3),
            alpha=50.0,
            sigma=5.0,
            data_format="HWDC"
        )
        config = layer.get_config()
        new_layer = RandomElasticDeformation3D.from_config(config)
        self.assertEqual(new_layer.grid_size, (3, 3, 3))
        self.assertAllClose(new_layer.alpha, tf.constant(50.0, dtype=tf.bfloat16))
        self.assertAllClose(new_layer.sigma, tf.constant(5.0, dtype=tf.bfloat16))
        self.assertEqual(new_layer.data_format, "HWDC")

if __name__ == "__main__":
    tf.test.main()