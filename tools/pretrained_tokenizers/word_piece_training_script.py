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
import os
import time

import keras_nlp

# List directories of parsed Wikipedia articles and vocab sizes
directories = [
    "eswiki_parsed",
    "frwiki_parsed",
    "hiwiki_parsed",
    "arwiki_parsed",
    "ruwiki_parsed",
    "bnwiki_parsed",
    "idwiki_parsed",
    "ptwiki_parsed",
]
vocab_sizes = [20000, 50000]
identifier = "v1"

# Runs the computation
for directory in directories:
    for vocab_size in vocab_sizes:
        print(f"Running directory {directory} with vocab size {vocab_size}")
        files = []
        for folder in os.listdir(directory):
            path = os.path.join(directory, folder)
            for file in os.listdir(path):
                if file[0] != ".":
                    files.append(os.path.join(path, file))

        if os.path.exists(f"{directory}_{vocab_size}_{identifier}.txt"):
            raise ValueError("already done.")

        start = time.time()
        keras_nlp.tokenizers.compute_word_piece_vocabulary(
            files,
            vocabulary_size=vocab_size,
            lowercase=False,
            strip_accents=False,
            vocabulary_output_file=f"{directory}_{vocab_size}_{identifier}.txt",
        )
        end = time.time()
        print("Time taken:", end - start)
