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

import os

import tensorflow as tf
from tensorflow import keras

ORIGIN = "https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert"


class BertTest(tf.test.TestCase):
    def test_end_to_end(self):
        """Runs an end-to-end test of all BERT modeling scripts."""

        # Download test data.
        temp_dir = self.get_temp_dir()
        vocab_file = keras.utils.get_file(
            origin=f"{ORIGIN}/bert_vocab_uncased.txt",
            cache_dir=temp_dir,
        )
        data_file = keras.utils.get_file(
            origin=f"{ORIGIN}/wiki_example_data.txt",
            cache_dir=temp_dir,
        )
        # Make the test vocab smaller for memory constraints.
        with open(vocab_file, "r+") as file:
            [file.readline() for x in range(5000)]
            file.truncate()

        # Split sentences.
        cmd = (
            "python3 examples/tools/split_sentences.py"
            f"    --input_files {data_file}"
            f"    --output_directory {temp_dir}/split"
        )
        print(cmd)
        self.assertEqual(os.system(cmd), 0)
        # Preprocess data.
        cmd = (
            "python3 examples/bert/bert_preprocess.py"
            f"    --input_files {temp_dir}/split"
            f"    --vocab_file {vocab_file}"
            f"    --output_file {temp_dir}/data.tfrecord"
        )
        print(cmd)
        self.assertEqual(os.system(cmd), 0)
        # Run pretraining.
        cmd = (
            "python3 examples/bert/bert_train.py"
            f"    --input_files {temp_dir}/data.tfrecord"
            f"    --vocab_file {vocab_file}"
            f"    --saved_model_output {temp_dir}/model/"
            f"    --num_train_steps 5"
        )
        print(cmd)
        self.assertEqual(os.system(cmd), 0)
        # Run fine-tuning.
        cmd = (
            "python3 examples/bert/bert_finetune_glue.py"
            f"    --saved_model_input {temp_dir}/model/"
            f"    --vocab_file {vocab_file}"
            f"    --num_train_steps 1"
        )
        print(cmd)
        self.assertEqual(os.system(cmd), 0)
