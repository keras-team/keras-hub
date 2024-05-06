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
import io
import pathlib

import sentencepiece


def train_sentencepiece(data, filename, *args, **kwargs):
    bytes_io = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(data), model_writer=bytes_io, *args, **kwargs
    )
    with open(
        pathlib.Path(__file__).parent.parent.parent
        / "keras_nlp"
        / "src"
        / "tests"
        / "test_data"
        / filename,
        mode="wb",
    ) as f:
        f.write(bytes_io.getbuffer())
