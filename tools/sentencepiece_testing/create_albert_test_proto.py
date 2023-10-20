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


def _train_sentencepiece(data, *args, **kwargs):
    bytes_io = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(data), model_writer=bytes_io, *args, **kwargs
    )
    return bytes_io


def main():
    bytes_io = _train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        vocab_size=12,
        model_type="WORD",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="[CLS]",
        eos_piece="[SEP]",
        user_defined_symbols="[MASK]",
    )
    with open(
        pathlib.Path(__file__).parent.parent.parent
        / "keras_nlp"
        / "tests"
        / "test_data"
        / "albert_sentencepiece.proto",
        mode="wb",
    ) as f:
        f.write(bytes_io.getbuffer())

    bytes_io = _train_sentencepiece(
        ["abc"],
        vocab_size=5,
        pad_id=-1,
        eos_id=-1,
        bos_id=-1,
    )
    with open(
        pathlib.Path(__file__).parent.parent.parent
        / "keras_nlp"
        / "tests"
        / "test_data"
        / "albert_sentencepiece_bad.proto",
        mode="wb",
    ) as f:
        f.write(bytes_io.getbuffer())


if __name__ == "__main__":
    main()
