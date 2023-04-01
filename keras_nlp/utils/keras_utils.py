# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import platform
import sys

import tensorflow as tf
from absl import logging
from tensorflow import keras

from keras_nlp.utils.tf_utils import is_tensor_type


def clone_initializer(initializer):
    """Clones an initializer to ensure a new seed.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    # If we get a string or dict, just return as we cannot and should not clone.
    if not isinstance(initializer, keras.initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)


def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    """Packs user-provided data into a tuple.

    This is a temporary copy of `keras.utils.pack_x_y_sample_weight` while we
    wait for the a change to the upstream version to propagate to a stable
    release. See https://github.com/keras-team/keras-nlp/issues/492
    """
    if y is None:
        if not isinstance(x, (list, tuple)):
            return x
        else:
            return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)


def convert_inputs_to_list_of_tensor_segments(x):
    """Converts user inputs to a list of a tensor segments.

    For models and layers which accept lists of string tensors to pack together,
    this method converts user inputs to a uniform format in a way that can be
    considered canonical for the library.

    We handle the following:

    - A single string will be converted to a tensor and wrapped in a list.
    - A list of strings will be converted to a tensor and wrapped in a list.
    - A single tensor will be wrapped in a list.
    - A list of tensors will be passed through unaltered.

    All other inputs will result in an error. This effectively means that users
    who would like to pack multiple segments together should convert those
    segments to tensors before calling the layer. This removes any ambiguity
    in the input for those cases.
    """
    # Check the input type.
    is_string = isinstance(x, (str, bytes))
    is_tensor = is_tensor_type(x)
    is_string_list = (
        isinstance(x, (list, tuple)) and x and isinstance(x[0], (str, bytes))
    )
    is_tensor_list = isinstance(x, (list, tuple)) and x and is_tensor_type(x[0])

    if is_string or is_string_list:
        # Automatically convert raw strings or string lists to tensors.
        # Wrap this input as a single (possibly batched) segment.
        x = [tf.convert_to_tensor(x)]
    elif is_tensor:
        # Automatically wrap a single tensor as a single segment.
        x = [x]
    elif is_tensor_list:
        # Pass lists of tensors though unaltered.
        x = x
    else:
        # Error for all other input.
        raise ValueError(
            f"Unsupported input for `x`. `x` should be a string, a list of "
            "strings, or a list of tensors. If passing multiple segments "
            "which should packed together, please convert your inputs to a "
            f"list of tensors. Received `x={x}`"
        )
    return x


def is_xla_compatible(model):
    """Determine if model and platform xla-compatible."""
    return not (
        platform.system() == "Darwin" and "arm" in platform.processor().lower()
    ) and not isinstance(
        model.distribute_strategy,
        (
            tf.compat.v1.distribute.experimental.TPUStrategy,
            tf.distribute.TPUStrategy,
        ),
    )


def print_msg(message, line_break=True):
    """Print the message to absl logging or stdout."""
    # Copied from core Keras.
    if keras.utils.is_interactive_logging_enabled():
        if line_break:
            sys.stdout.write(message + "\n")
        else:
            sys.stdout.write(message)
        sys.stdout.flush()
    else:
        logging.info(message)


def print_row(fields, positions, print_fn, nested_level=0):
    """Print a row of a summary message."""
    # Copied from core Keras.
    left_to_print = [str(x) for x in fields]
    while any(left_to_print):
        line = ""
        for col in range(len(left_to_print)):
            if col > 0:
                start_pos = positions[col - 1]
            else:
                start_pos = 0
            end_pos = positions[col]
            # Leave room for 2 spaces to delineate columns
            # we don't need any if we are printing the last column
            space = 2 if col != len(positions) - 1 else 0
            cutoff = end_pos - start_pos - space
            fit_into_line = left_to_print[col][:cutoff]
            # For nicer formatting we line-break on seeing end of
            # tuple/dict etc.
            line_break_conditions = ("),", "},", "],", "',")
            candidate_cutoffs = [
                fit_into_line.find(x) + len(x)
                for x in line_break_conditions
                if fit_into_line.find(x) >= 0
            ]
            if candidate_cutoffs:
                cutoff = min(candidate_cutoffs)
                fit_into_line = fit_into_line[:cutoff]

            if col == 0:
                line += "|" * nested_level + " "
            line += fit_into_line
            line += " " * space if space else ""
            left_to_print[col] = left_to_print[col][cutoff:]

            # Pad out to the next position
            if nested_level:
                line += " " * (positions[col] - len(line) - nested_level)
            else:
                line += " " * (positions[col] - len(line))
        line += "|" * nested_level
        print_fn(line)
