import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.bge.bge_backbone import BgeBackbone
from keras_hub.src.models.bge.bge_text_embedder_preprocessor import (
    BgeTextEmbedderPreprocessor,
)
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.BgeTextEmbedder")
class BgeTextEmbedder(Task):
    """An end-to-end BGE model for text embedding.

    This model encodes input text into L2-normalized dense embeddings suitable
    for semantic similarity, retrieval, clustering, and reranking tasks. It
    extracts the `[CLS]` token's hidden state from `BgeBackbone` and applies
    L2 normalization, producing unit-norm embeddings where cosine similarity
    equals the dot product.

    This model can optionally be configured with a `preprocessor` layer, in
    which case raw string inputs will be automatically tokenized during
    `predict()` and `evaluate()`. This is done by default when creating the
    model with `from_preset()`.

    Usage note: Cosine similarity between embeddings can be computed as
    a plain dot product after this model's output, because embeddings are
    already L2-normalized. For retrieval, encode queries and passages
    separately, then compute `query_emb @ passage_emb.T`.

    Query instruction: For retrieval tasks, BGE recommends prepending the
    instruction `"Represent this sentence for searching relevant passages: "`
    to query strings (not to passages). This is optional for similarity tasks.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://huggingface.co/BAAI/bge-small-en-v1.5).

    Args:
        backbone: A `keras_hub.models.BgeBackbone` instance.
        preprocessor: A `keras_hub.models.BgeTextEmbedderPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Encode a batch of sentences.
    ```python
    # Load with pre-trained weights and preprocessor.
    embedder = keras_hub.models.BgeTextEmbedder.from_preset("bge_small_en_v1.5")

    # Encode sentences — output shape: [batch_size, 384]
    embeddings = embedder.predict(["Hello world.", "The quick brown fox."])

    # Compute cosine similarity via dot product (embeddings are L2-normalized).
    import numpy as np
    similarity = np.dot(embeddings[0], embeddings[1])
    ```

    Retrieval: encode queries and passages separately.
    ```python
    embedder = keras_hub.models.BgeTextEmbedder.from_preset("bge_small_en_v1.5")

    query_instruction = (
        "Represent this sentence for searching relevant passages:"
    )
    queries = [query_instruction + "What is BGE?"]
    passages = ["BGE is a text embedding model by BAAI."]

    query_emb = embedder.predict(queries)    # [1, 384]
    passage_emb = embedder.predict(passages) # [1, 384]

    scores = query_emb @ passage_emb.T  # cosine similarity matrix
    ```

    Pre-tokenized integer inputs (no preprocessor).
    ```python
    embedder = keras_hub.models.BgeTextEmbedder.from_preset(
        "bge_small_en_v1.5",
        preprocessor=None,
    )
    inputs = {
        "token_ids": np.array([[101, 7592, 2088, 102, 0, 0]], dtype="int32"),
        "segment_ids": np.zeros((1, 6), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 0, 0]], dtype="int32"),
    }
    embeddings = embedder(inputs)  # [1, 384], L2-normalized
    ```

    Custom backbone and vocabulary.
    ```python
    vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab += ["hello", "world", "."]
    tokenizer = keras_hub.models.BgeTokenizer(vocabulary=vocab)
    preprocessor = keras_hub.models.BgeTextEmbedderPreprocessor(
        tokenizer=tokenizer,
        sequence_length=32,
    )
    backbone = keras_hub.models.BgeBackbone(
        vocabulary_size=len(vocab),
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        intermediate_dim=128,
        max_sequence_length=32,
    )
    embedder = keras_hub.models.BgeTextEmbedder(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    embedder.predict(["hello world."])
    ```
    """

    backbone_cls = BgeBackbone
    preprocessor_cls = BgeTextEmbedderPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        # Extract the [CLS] token hidden state from the encoder output.
        # sequence_output[:, 0, :] is the raw CLS representation — distinct
        # from pooled_output, which passes through an additional Tanh layer
        # and produces embeddings in a different space.
        cls_hidden = backbone(inputs)["sequence_output"][
            :, backbone.cls_token_index, :
        ]
        # L2-normalize so that dot product equals cosine similarity.
        outputs = keras.ops.normalize(cls_hidden, axis=-1)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        # Default to no loss/optimizer — this model is used for inference,
        # not end-to-end supervised training. Users who fine-tune with
        # contrastive loss should compile explicitly.
        if optimizer == "auto":
            optimizer = None
        if loss == "auto":
            loss = None
        if metrics == "auto":
            metrics = None
        super().compile(
            optimizer=optimizer, loss=loss, metrics=metrics, **kwargs
        )
