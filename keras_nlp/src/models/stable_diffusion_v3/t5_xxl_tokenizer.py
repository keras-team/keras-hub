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
from keras_nlp.src.models.t5.t5_tokenizer import T5Tokenizer

try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


class T5XXLTokenizer(T5Tokenizer):
    def set_proto(self, proto):
        super().set_proto(proto)
        if proto is None:
            return

        # Re-instantiate `self._sentence_piece` with `add_eos=True`. This is
        # necessary to ensure consistent output with
        # `transformers.T5TokenizerFast` for StableDiffusionV3.
        self._sentence_piece = tf_text.SentencepieceTokenizer(
            model=self.proto,
            out_type=self.compute_dtype,
            add_eos=True,
        )
