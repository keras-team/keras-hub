from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.t5.t5_backbone import T5Backbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.T5Tokenizer",
        "keras_hub.models.T5Tokenizer",
    ]
)
class T5Tokenizer(SentencePieceTokenizer):
    """T5 tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    T5 models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a T5 preset.

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
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=8,
        model_type="WORD",
        bos_id=-1,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        pad_piece="<pad>",
        eos_piece="</s>",
        unk_piece="<unk>",
    )
    tokenizer = keras_hub.models.T5Tokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")

    # Batched inputs.
    tokenizer(["the quick brown fox", "the earth is round"])

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = T5Backbone

    def __init__(self, proto, **kwargs):
        # T5 uses the same start token as end token, i.e., "<\s>".
        self._add_special_token("</s>", "end_token")
        self._add_special_token("</s>", "start_token")
        self._add_special_token("<pad>", "pad_token")
        super().__init__(proto=proto, **kwargs)

    def detokenize(self, inputs):
        if not hasattr(self, "special_tokens_map"):
            self.special_tokens_map = {
                self.end_token_id: "</s>",
                self.pad_token_id: "<pad>",
            }
            self.special_tokens_map = {k: v for k, v in self.special_tokens_map.items() if k is not None}

        import tensorflow as tf
        if not tf.executing_eagerly():
            return super().detokenize(inputs)
            
        inputs_list = tf.convert_to_tensor(inputs).numpy().tolist()
        is_scalar = isinstance(inputs_list, int) or (len(inputs_list) > 0 and isinstance(inputs_list[0], int))
        if is_scalar: inputs_list = [inputs_list] if isinstance(inputs_list, list) else [[inputs_list]]

        decoded_outputs = []
        for seq in inputs_list:
            words, current_chunk = [], []
            def decode_and_append():
                if current_chunk:
                    decoded = super(self.__class__, self).detokenize(current_chunk)
                    if hasattr(decoded, "numpy"): decoded = decoded.numpy()
                    if isinstance(decoded, list) and len(decoded) > 0: decoded = decoded[0]
                    if isinstance(decoded, bytes): decoded = decoded.decode('utf-8')
                    words.append(str(decoded))
                    current_chunk.clear()

            for token_id in seq:
                if token_id in self.special_tokens_map:
                    decode_and_append()
                    words.append(self.special_tokens_map[token_id])
                else:
                    current_chunk.append(token_id)
            decode_and_append()
            decoded_outputs.append(" ".join(words).strip())

        return tf.convert_to_tensor(decoded_outputs[0]) if is_scalar else tf.convert_to_tensor(decoded_outputs)
