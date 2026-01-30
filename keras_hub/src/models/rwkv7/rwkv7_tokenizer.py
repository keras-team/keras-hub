import os

import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tensor_to_list
from keras_hub.src.utils.tensor_utils import tf

# Vocabulary file name constant
VOCAB_FILENAME = "vocabulary.txt"


class TRIE:
    """Byte-level Trie structure for longest prefix matching.

    This class implements a trie data structure that stores byte
    sequences and allows efficient longest prefix matching.
    """

    __slots__ = tuple("ch,children,values,parent".split(","))

    def __init__(self, parent=None, ch=None):
        """Initialize a TRIE node.

        Args:
            parent: Parent node reference.
            ch: Byte value for this node.
        """
        self.ch = ch
        self.children = [None for _ in range(256)]
        self.values = set()
        self.parent = parent

    def __repr__(self):
        """String representation of the TRIE node."""
        current_node = self
        ret = []
        while current_node is not None:
            if current_node.ch is not None:
                ret.append(current_node.ch)
            current_node = current_node.parent
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key, idx=0, val=None):
        """Add a key-value pair to the trie.

        Args:
            key: Byte sequence to add.
            idx: Current index in key processing.
            val: Value to store (defaults to key).

        Returns:
            Final node where key was inserted.
        """
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self

        ch = key[idx]
        if self.children[ch] is None:
            self.children[ch] = TRIE(parent=self, ch=ch)

        return self.children[ch].add(key, idx + 1, val)

    def find_longest(self, key, idx=0):
        """Find longest match in trie for given key.

        Args:
            key: Byte sequence to search for.
            idx: Starting index for search.

        Returns:
            Tuple of (end_index, node, values) for match.
        """
        current_node = self
        ch = key[idx]
        ret = None

        while current_node.children[ch] is not None:
            current_node = current_node.children[ch]
            idx += 1
            if current_node.values:
                ret = idx, current_node, current_node.values
            if idx == len(key):
                break
            ch = key[idx]
        if ret is None:
            raise ValueError(f"No valid token found in trie for key: {key}")
        return ret


class RWKVTokenizerBase:
    """RWKV tokenizer implementation using byte-level trie.

    Implements tokenization using a fixed vocabulary and greedy
    longest-match algorithm on byte sequences.
    """

    def __init__(self, vocabs):
        """Initialize tokenizer with vocabulary.

        Args:
            vocabs: List of vocabulary entries in format
                   "<idx> <repr> <len>".
        """
        self.idx2token = {}
        for line in vocabs:
            idx = int(line[: line.index(" ")])
            x = eval(line[line.index(" ") : line.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(line[line.rindex(" ") :])
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for token, token_id in self.token2idx.items():
            _ = self.root.add(token, val=(token, token_id))

    def encodeBytes(self, src):
        """Encode byte sequence to token IDs.

        Args:
            src: Byte sequence to encode.

        Returns:
            List of token IDs.
        """
        idx = 0
        tokens = []
        while idx < len(src):
            prev_idx = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert idx != prev_idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        """Decode token IDs to byte sequence.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded byte sequence.
        """
        return b"".join(map(lambda i: self.idx2token[int(i)], tokens))

    def encode(self, src):
        """Encode text to token IDs.

        Args:
            src: Text string or list of strings.

        Returns:
            Token IDs or list of token ID lists.
        """
        if isinstance(src, str):
            return self.encodeBytes(src.encode("utf-8"))
        else:
            return [self.encodeBytes(s.encode("utf-8")) for s in src]

    def decode(self, tokens):
        """Decode token IDs to text.

        Args:
            tokens: Token IDs or list of token ID lists.

        Returns:
            List of decoded text strings.
        """
        return [self.decodeBytes(batch).decode("utf-8") for batch in tokens]
        # try:
        #     return self.decodeBytes(tokens).decode('utf-8')
        # except:
        #     return '\ufffd' # bad utf-8

    def printTokens(self, tokens):
        """Print tokens with their string representations.

        Args:
            tokens: List of token IDs to print.
        """
        for token_id in tokens:
            token = self.idx2token[token_id]
            try:
                token = token.decode("utf-8")
            except BaseException:
                pass
            print(f"{repr(token)}{token_id}", end=" ")
        print()


@keras_hub_export("keras_hub.tokenizers.RWKVTokenizer")
class RWKVTokenizer(tokenizer.Tokenizer):
    """RWKV byte-level tokenizer with longest-match trie search.

    This tokenizer maps raw text to a sequence of integer token ids
    using a fixed vocabulary and a greedy longest-match algorithm.

    Args:
        vocabulary: list of strings, each line formatted as
            "<idx> <repr> <len>".
        dtype: output dtype for tensor operations. Must be integer
            or string type.

    Examples:
    ```python
    vocab = ["0 ' ' 1", "1 '\\n' 1", "2 'the' 3", "3 'hello' 5"]
    tok = RWKVTokenizer(vocabulary=vocab)
    tok("hello the")
    ```

    Output:
    [3, 0, 2]
    """

    backbone_cls = RWKV7Backbone

    def __init__(
        self,
        vocabulary=None,
        dtype="int32",
        pad_token_id=0,
        start_token_id=None,
        end_token_id=None,
        **kwargs,
    ):
        """Initialize RWKV tokenizer.

        Args:
            vocabulary: Vocabulary list.
            dtype: Output data type.
            **kwargs: Additional keyword arguments.
        """
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        super().__init__(dtype=dtype, **kwargs)

        self.vocabulary = None
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id or self.pad_token_id
        if vocabulary is not None:
            self.set_vocabulary(vocabulary)
        self.file_assets = [VOCAB_FILENAME]

    def set_vocabulary(self, vocabulary):
        """Set the tokenizer vocabulary.

        Args:
            vocabulary: Vocabulary list to set.
        """
        self.vocabulary = vocabulary
        self._tokenizer = RWKVTokenizerBase(vocabulary)
        if self.end_token_id is None or self.end_token_id == self.pad_token_id:
            for line in vocabulary:
                idx = int(line[: line.index(" ")])
                repr_str = eval(line[line.index(" ") : line.rindex(" ")])
                if repr_str == "\n\n":
                    self.end_token_id = idx
                    break

    def save_assets(self, dir_path):
        """Save vocabulary to directory.

        Args:
            dir_path: Directory path to save to.
        """
        path = os.path.join(dir_path, VOCAB_FILENAME)
        with open(path, "w", encoding="utf-8") as file:
            file.write("".join(self.vocabulary))

    def load_assets(self, dir_path=""):
        """Load vocabulary from directory.

        Args:
            dir_path: Directory path to load from.
        """
        path = os.path.join(dir_path, VOCAB_FILENAME)
        with open(path, "r", encoding="utf-8") as f:
            vocabulary = f.readlines()
        self.set_vocabulary(vocabulary)

    def _check_vocabulary(self):
        """Check if vocabulary is set, raise error if not."""
        if self.vocabulary is None:
            raise ValueError(
                "No vocabulary has been set for RWKVTokenizer. Make "
                "sure to pass a `vocabulary` argument when creating the layer."
            )

    def vocabulary_size(self):
        """Get the size of the vocabulary.

        Returns:
            Number of tokens in vocabulary.
        """
        self._check_vocabulary()
        return int(len(self.vocabulary))

    def get_vocabulary(self):
        """Get the current vocabulary.

        Returns:
            Current vocabulary list.
        """
        self._check_vocabulary()
        return tensor_to_list(self.vocabulary)

    def id_to_token(self, id):
        """Convert token ID to string representation.

        Args:
            id: Token ID to convert.

        Returns:
            String representation of token.
        """
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
        """Get tokenizer configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "vocabulary": self.vocabulary,
                "end_token_id": self.end_token_id,
                "pad_token_id": self.pad_token_id,
                "start_token_id": self.start_token_id,
            }
        )
        return config

    @preprocessing_function
    def tokenize(self, inputs):
        self._check_vocabulary()

        if not tf.executing_eagerly() and tf.is_tensor(inputs):

            def tokenize_wrapper(text_tensor):
                text_list = (
                    text_tensor.numpy()
                    if hasattr(text_tensor, "numpy")
                    else text_tensor
                )
                if isinstance(text_list, bytes):
                    text_list = [text_list.decode("utf-8")]
                elif isinstance(text_list, np.ndarray):
                    text_list = [x.decode("utf-8") for x in text_list.flatten()]

                tokens = self._tokenizer.encode(text_list)

                if is_string_dtype(self.dtype):
                    result = [
                        self.id_to_token(i).decode("utf-8", errors="replace")
                        for i in tokens[0]
                    ]
                    return tf.constant(result, dtype=tf.string)
                else:
                    return tf.constant(tokens[0], dtype=self.compute_dtype)

            if inputs.shape.rank == 0:
                output = tf.py_function(
                    tokenize_wrapper,
                    [inputs],
                    Tout=tf.string
                    if is_string_dtype(self.dtype)
                    else self.compute_dtype,
                )
                output.set_shape([None])
                return output
            else:
                output = tf.map_fn(
                    lambda x: tf.py_function(
                        tokenize_wrapper,
                        [x],
                        Tout=tf.string
                        if is_string_dtype(self.dtype)
                        else self.compute_dtype,
                    ),
                    inputs,
                    fn_output_signature=tf.TensorSpec(
                        [None],
                        dtype=tf.string
                        if is_string_dtype(self.dtype)
                        else self.compute_dtype,
                    ),
                )
                return output

        if tf.is_tensor(inputs):
            inputs = tensor_to_list(inputs)

        tokens = self._tokenizer.encode(inputs)

        if is_string_dtype(self.dtype):

            def ids_to_str(ids):
                return [
                    self.id_to_token(i).decode("utf-8", errors="replace")
                    for i in ids
                ]

            if isinstance(inputs, str):
                return ids_to_str(tokens)
            return [ids_to_str(ts) for ts in tokens]

        if isinstance(inputs, str):
            return tf.convert_to_tensor(tokens, dtype=self.compute_dtype)
        else:
            return tf.ragged.constant(tokens, dtype=self.compute_dtype)

    @preprocessing_function
    def detokenize(self, inputs):
        """Convert tokens back to text.

        Args:
            inputs: Tokens to convert.

        Returns:
            Detokenized text.
        """
        self._check_vocabulary()

        if tf.is_tensor(inputs):
            inputs = tensor_to_list(inputs)

        if len(inputs) > 0 and isinstance(inputs[0], (int, np.integer)):
            inputs = [inputs]

        strip_zero_inputs = []
        for seq in inputs:
            if tf.is_tensor(seq):
                seq = tensor_to_list(seq)
            strip_zero_inputs.append([x for x in seq if x != 0])

        result = self._tokenizer.decode(strip_zero_inputs)

        return tf.convert_to_tensor(result, dtype=tf.string)

    def compute_output_spec(self, input_spec):
        """Compute output specification.

        Args:
            input_spec: Input specification.

        Returns:
            Output tensor specification.
        """
        return keras.KerasTensor(
            input_spec.shape + (None,), dtype=self.compute_dtype
        )

    def call(self, inputs):
        """Call the tokenizer on inputs.

        Args:
            inputs: Input text.

        Returns:
            Tokenized output.
        """
        return self.tokenize(inputs)
