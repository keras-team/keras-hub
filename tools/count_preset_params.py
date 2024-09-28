"""
Small utility script to count parameters in our preset checkpoints.

Usage:
python tools/count_preset_params.py
python tools/count_preset_params.py --model BertBackbone
python tools/count_preset_params.py --preset bert_base_multi
"""

import inspect

from absl import app
from absl import flags
from keras.utils.layer_utils import count_params
from tensorflow import keras

import keras_hub

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "The name of a model, e.g. BertBackbone.")
flags.DEFINE_string(
    "preset", None, "The name of a preset, e.g. bert_base_multi."
)


def main(_):
    for name, symbol in keras_hub.models.__dict__.items():
        if inspect.isclass(symbol) and issubclass(symbol, keras.Model):
            if FLAGS.model and name != FLAGS.model:
                continue
            if not hasattr(symbol, "from_preset"):
                continue
            for preset in symbol.presets:
                if FLAGS.preset and preset != FLAGS.preset:
                    continue
                model = symbol.from_preset(preset)
                params = count_params(model.weights)
                print(f"{name} {preset} {params}")


if __name__ == "__main__":
    app.run(main)
