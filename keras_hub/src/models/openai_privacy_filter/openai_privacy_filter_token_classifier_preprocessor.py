from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.models.token_classifier_preprocessor import (
    TokenClassifierPreprocessor,
)


@keras_hub_export(
    "keras_hub.models.OpenAIPrivacyFilterTokenClassifierPreprocessor"
)
class OpenAIPrivacyFilterTokenClassifierPreprocessor(
    TokenClassifierPreprocessor
):
    """Preprocessor for OpenAI Privacy Filter token classification.

    This preprocessing layer tokenizes and pads/truncates input text into
    `token_ids` and `padding_mask` tensors suitable for the backbone. The
    model uses no start or end tokens, so the packed ids are the raw token
    ids followed only by padding.

    When per-token labels `y` are passed, they are padded or truncated to the
    same `sequence_length` (pad label `0`). A row of `y` with fewer labels than
    its (truncated) token count is rejected, with a `ValueError` on the Python
    workflow and an `InvalidArgumentError` inside `tf.data`, since the missing
    positions would otherwise be trained toward label `0`. When `sample_weight`
    is not provided, it defaults to `padding_mask` so that padded positions do
    not contribute to the loss. When an explicit `sample_weight` is provided,
    per-token weights are padded or truncated to `sequence_length`; a
    per-sample weight is broadcast over tokens and multiplied by the padding
    mask.

    The pad token is `<|endoftext|>`, as in the Hugging Face checkpoint. A
    literal `<|endoftext|>` inside the input text therefore receives the pad
    id and is masked out.

    Examples:

    ```python
    from keras_hub.models import OpenAIPrivacyFilterTokenClassifierPreprocessor

    # Load the preprocessor from a preset.
    preprocessor = OpenAIPrivacyFilterTokenClassifierPreprocessor.from_preset(
        "hf://openai/privacy-filter"
    )

    # Tokenize and pad/truncate a single input.
    preprocessor("The quick brown fox jumped.")

    # Tokenize and pad/truncate a batch of inputs.
    preprocessor(["The quick brown fox jumped.", "The fox slept."])
    ```

    Args:
        tokenizer: A `OpenAIPrivacyFilterTokenizer` instance.
        sequence_length: int. The length of the packed inputs. Sequences
            longer than this will be truncated. Defaults to `512`.
        truncate: string. The algorithm to truncate a list of batched
            segments. Defaults to `"round_robin"`.
    """

    backbone_cls = OpenAIPrivacyFilterBackbone
    tokenizer_cls = OpenAIPrivacyFilterTokenizer
