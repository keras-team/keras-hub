import os

os.environ["KERAS_BACKEND"] = "torch"
import keras

print("Keras import OK")
print(keras.__version__)
