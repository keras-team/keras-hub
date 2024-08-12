# Copyright 2024 The KerasNLP Authors
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

from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "deberta_v3_test_vocab.spm",
        vocab_size=12,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="[PAD]",
        bos_piece="[CLS]",
        eos_piece="[SEP]",
        unk_piece="[UNK]",
        user_defined_symbols="[MASK]",
    )


if __name__ == "__main__":
    main()
