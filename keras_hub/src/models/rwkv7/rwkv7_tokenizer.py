import os

import keras
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import tensor_to_list

VOCAB_FILENAME = "vocab.txt"


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr is not None:
            if fr.ch is not None:
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]

        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = idx, u, u.values
            if idx == len(key):
                break
            ch = key[idx]
        return ret


class RWKV_TOKENIZER:
    def __init__(self, vocabs):
        self.idx2token = {}
        sorted = []  # must be already sorted
        for l in vocabs:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx: int = 0
        tokens = []
        while idx < len(src):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        if isinstance(src, str):
            return self.encodeBytes(src.encode("utf-8"))
        else:
            return [self.encodeBytes(s.encode("utf-8")) for s in src]

    def decode(self, tokens):
        return [self.decodeBytes(batch).decode("utf-8") for batch in tokens]
        # try:
        #     return self.decodeBytes(tokens).decode('utf-8')
        # except:
        #     return '\ufffd' # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except BaseException:
                pass
            print(f"{repr(s)}{i}", end=" ")
        print()

@keras_hub_export("keras_hub.tokenizers.RWKVTokenizer")
class RWKVTokenizer(tokenizer.Tokenizer):
    def __init__(
        self,
        vocabulary=None,
        dtype="int32",
        **kwargs,
    ) -> None:
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        super().__init__(dtype=dtype, **kwargs)

        self.vocabulary = None
        if vocabulary is not None:
            self.set_vocabulary(vocabulary)
        self.file_assets = [VOCAB_FILENAME]

    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
        self._tokenizer = RWKV_TOKENIZER(vocabulary)
        self.pad_token_id = 0
        self.start_token_id = None
        self.end_token_id = self.tokenize(["\n\n"])[0][0]

    def save_assets(self, dir_path):
        path = os.path.join(dir_path, VOCAB_FILENAME)
        with open(path, "wb") as file:
            file.write("\n".join(self.vocabulary))

    def load_assets(self, dir_path=""):
        path = os.path.join(dir_path, VOCAB_FILENAME)
        with open(path, "r", encoding="utf-8") as f:
            vocabulary = f.readlines()
        self.set_vocabulary(vocabulary)

    def _check_vocabulary(self):
        if self.vocabulary is None:
            raise ValueError(
                "No vocabulary has been set for RWKVTokenizer. Make "
                "sure to pass a `vocabulary` argument when creating the layer."
            )

    def vocabulary_size(self):
        self._check_vocabulary()
        return int(len(self.vocabulary))

    def get_vocabulary(self):
        self._check_vocabulary()
        return tensor_to_list(self.vocabulary)

    def id_to_token(self, id):
        self._check_vocabulary()
        if id >= self.vocabulary_size() or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )
        return self._tokenizer.idx2token[id]

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        self._check_vocabulary()
        return int(self._tokenizer.token2idx[token])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary": None,  # Save vocabulary via an asset!
            }
        )
        return config

    def tokenize(self, inputs):
        self._check_vocabulary()
        tokens = self._tokenizer.encode(inputs)

        def tokens2ids(x):
            return [self.id_to_token(t) for t in x]

        if is_string_dtype(self.dtype):
            if isinstance(inputs, str):
                return tokens2ids(tokens)
            return [tokens2ids(t) for t in tokens]
        return tokens

    def detokenize(self, inputs):
        self._check_vocabulary()
        strip_zero_inputs = []
        for t in inputs:
            strip_zero_inputs.append([x for x in t if x != 0])

        return self._tokenizer.decode(strip_zero_inputs)

    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(
            input_spec.shape + (None,), dtype=self.compute_dtype
        )

    def call(self, inputs):
        return self.tokenize(inputs)
