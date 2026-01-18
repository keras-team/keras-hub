import keras

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.ReversibleEmbedding")
class ReversibleEmbedding(keras.layers.ReversibleEmbedding):
    pass
