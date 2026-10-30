from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modern_bert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.models.text_embedder_preprocessor import (
    TextEmbedderPreprocessor,
)


@keras_hub_export("keras_hub.models.ModernBertTextEmbedderPreprocessor")
class ModernBertTextEmbedderPreprocessor(TextEmbedderPreprocessor):
    """A ModernBERT preprocessing layer which tokenizes and packs inputs
    for sentence embedding.

    This preprocessing layer will do three things:

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together using a `keras_hub.layers.MultiSegmentPacker`
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens, i.e.,
       adding a single `"[CLS]"` at the start of the entire sequence and a
       `"[SEP]"` at the end of each segment. This matches the Hugging Face
       ModernBERT template: `[CLS] A [SEP]` for single inputs and
       `[CLS] A [SEP] B [SEP]` for pairs.
    3. Construct a dictionary with keys `"token_ids"` and `"padding_mask"`
       that can be passed directly to a ModernBERT text embedder. ModernBERT
       has no segment embeddings, so no `"segment_ids"` are returned.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    Args:
        tokenizer: A `keras_hub.models.ModernBertTokenizer` instance.
        sequence_length: int. The length of the packed inputs. Defaults to
            `256`. When loading from a Hugging Face sentence-transformers
            checkpoint, this is overridden by the length sentence-transformers
            uses: `max_seq_length` from `sentence_bert_config.json` (e.g.
            `8192` for `nomic-ai/modernbert-embed-base`), or otherwise the
            model's maximum length (e.g. `8192` for
            `Alibaba-NLP/gte-modernbert-base`). Every input is padded to this
            length, so for faster inference on short texts, pass a smaller
            `sequence_length` to this class's `from_preset()` or set the
            `sequence_length` attribute after loading.
        truncate: string. The algorithm to truncate a list of batched segments
            to fit within `sequence_length`. The value can be either
            `round_robin` or `waterfall`:
            - `"round_robin"`: Available space is assigned one token at a
                time in a round-robin fashion to the inputs that still need
                some, until the limit is reached.
            - `"waterfall"`: The allocation of the budget is done using a
                "waterfall" algorithm that allocates quota in a
                left-to-right manner and fills up the buckets until we run
                out of budget. It supports an arbitrary number of segments.

    Call arguments:
        x: A tensor of single string sequences, or a tuple of multiple
            tensor sequences to be packed together. Inputs may be batched or
            unbatched. For single sequences, raw python inputs will be
            converted to tensors. For multiple sequences, pass tensors
            directly.
        y: Any label data. Will be passed through unaltered.
        sample_weight: Any label weight data. Will be passed through
            unaltered.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = (
        keras_hub.models.ModernBertTextEmbedderPreprocessor.from_preset(
            "hf://nomic-ai/modernbert-embed-base",
            sequence_length=128,
        )
    )

    # Tokenize and pack a single sentence.
    preprocessor("search_query: What is TSNE?")

    # Tokenize a batch of single sentences.
    preprocessor([
        "search_query: What is TSNE?",
        "search_document: TSNE is a dimensionality reduction algorithm.",
    ])
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = (
        keras_hub.models.ModernBertTextEmbedderPreprocessor.from_preset(
            "hf://nomic-ai/modernbert-embed-base",
            sequence_length=128,
        )
    )

    first = tf.constant(["The quick brown fox jumped.", "Call me Ishmael."])

    # Map unlabeled single sentences.
    ds = tf.data.Dataset.from_tensor_slices(first)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = ModernBertBackbone
    tokenizer_cls = ModernBertTokenizer

    def _format_output(self, token_ids, segment_ids):
        """Add the ModernBERT padding mask; omit `segment_ids` (not used)."""
        return {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }
