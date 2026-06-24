from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.albert.albert_backbone import AlbertBackbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.AlbertTokenizer",
        "keras_hub.models.AlbertTokenizer",
    ]
)
class AlbertTokenizer(SentencePieceTokenizer):
    """ALBERT tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    ALBERT models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a ALBERT preset.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.AlbertTokenizer.from_preset(
        "albert_base_en_uncased",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=10,
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
    tokenizer = keras_hub.models.AlbertTokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = AlbertBackbone

    def __init__(self, proto, **kwargs):
        self._add_special_token("[CLS]", "cls_token")
        self._add_special_token("[SEP]", "sep_token")
        self._add_special_token("<pad>", "pad_token")
        self._add_special_token("[MASK]", "mask_token")
        # Also add `tokenizer.start_token` and `tokenizer.end_token` for
        # compatibility with other tokenizers.
        self._add_special_token("[CLS]", "start_token")
        self._add_special_token("[SEP]", "end_token")
        super().__init__(proto=proto, **kwargs)

    def detokenize(self, inputs):
        if not hasattr(self, "special_tokens_map"):
            self.special_tokens_map = {
                self.cls_token_id: "[CLS]",
                self.sep_token_id: "[SEP]",
                self.pad_token_id: "<pad>",
                self.mask_token_id: "[MASK]",
            }
            self.special_tokens_map = {
                k: v for k, v in self.special_tokens_map.items() if k is not None
            }

        import tensorflow as tf
        
        if not tf.executing_eagerly():
            return super().detokenize(inputs)
            
        inputs_tensor = tf.convert_to_tensor(inputs)
        inputs_list = inputs_tensor.numpy().tolist()
        
        is_scalar = False
        if isinstance(inputs_list, int):
            inputs_list = [[inputs_list]]
            is_scalar = True
        elif len(inputs_list) > 0 and isinstance(inputs_list[0], int):
            inputs_list = [inputs_list]
            is_scalar = True

        decoded_outputs = []
        for seq in inputs_list:
            words = []
            current_chunk = []
            for token_id in seq:
                if token_id in self.special_tokens_map:
                    if current_chunk:
                        decoded_chunk = super(AlbertTokenizer, self).detokenize(current_chunk)
                        if hasattr(decoded_chunk, "numpy"):
                            decoded_chunk = decoded_chunk.numpy()
                        if isinstance(decoded_chunk, bytes):
                            words.append(decoded_chunk.decode('utf-8'))
                        elif isinstance(decoded_chunk, list) and len(decoded_chunk) > 0 and isinstance(decoded_chunk[0], bytes):
                            words.append(decoded_chunk[0].decode('utf-8'))
                        elif isinstance(decoded_chunk, list):
                            words.append(str(decoded_chunk[0]))
                        elif isinstance(decoded_chunk, str):
                            words.append(decoded_chunk)
                        else:
                            words.append(decoded_chunk[0].decode('utf-8') if hasattr(decoded_chunk[0], 'decode') else str(decoded_chunk[0]))
                        current_chunk = []
                    words.append(self.special_tokens_map[token_id])
                else:
                    current_chunk.append(token_id)
            if current_chunk:
                decoded_chunk = super(AlbertTokenizer, self).detokenize(current_chunk)
                if hasattr(decoded_chunk, "numpy"):
                    decoded_chunk = decoded_chunk.numpy()
                if isinstance(decoded_chunk, bytes):
                    words.append(decoded_chunk.decode('utf-8'))
                elif isinstance(decoded_chunk, list) and len(decoded_chunk) > 0 and isinstance(decoded_chunk[0], bytes):
                    words.append(decoded_chunk[0].decode('utf-8'))
                elif isinstance(decoded_chunk, list):
                    words.append(str(decoded_chunk[0]))
                elif isinstance(decoded_chunk, str):
                    words.append(decoded_chunk)
                else:
                    words.append(decoded_chunk[0].decode('utf-8') if hasattr(decoded_chunk[0], 'decode') else str(decoded_chunk[0]))
            decoded_outputs.append(" ".join(words).strip())

        if is_scalar:
            return tf.convert_to_tensor(decoded_outputs[0])
        return tf.convert_to_tensor(decoded_outputs)
