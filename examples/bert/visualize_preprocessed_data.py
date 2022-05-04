# Copyright 2022 The KerasNLP Authors
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
"""A script to visualize the preprocessed data.

To run the script, use
```
python examples/bert/visualize_preprocessed_data.py \
    --filepath=path/to/your/tfrecord
```
"""

import tensorflow as tf
from absl import app
from absl import flags

from examples.bert.bert_utils import decode_record

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "filepath",
    None,
    "The file path to preprocessed data.",
)


def visualize_tfrecord(filepath):
    dataset = tf.data.TFRecordDataset(filepath)
    sample = next(iter(dataset))
    decoded = decode_record(sample)
    for key, val in decoded.items():
        print(key, ": ", val)


def main(_):
    visualize_tfrecord(FLAGS.filepath)


if __name__ == "__main__":
    flags.mark_flag_as_required("filepath")
    app.run(main)
