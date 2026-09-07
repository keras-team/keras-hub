from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.MuseGlimmerTokenizer",
        "keras_hub.models.MuseGlimmerTokenizer",
    ]
)
class MuseGlimmerTokenizer(BytePairTokenizer):
    """A MuseGlimmer byte-level BPE tokenizer.

    Uses the same Llama-3/GPT-4o-style regex pre-tokenizer split as
    `Llama3Tokenizer`. `image_token_id`/`video_token_id` are fixed
    placeholder ids (per `config.json`) that the preprocessor expands to
    per-patch image/video feature positions — they are not resolved from a
    named special-token string, since the checkpoint assigns them directly
    by id among its ~2,043 reserved special tokens.

    Args:
        vocabulary: dict or None. Maps token strings to integer ids.
        merges: list or None. BPE merge rules.
        bos_token: str. Beginning-of-sequence token.
        eos_token: str. End-of-sequence token.
        pad_token: str. Padding token.
        image_token_id: int. Placeholder token id expanded to per-patch
            image embeddings by the preprocessor. Defaults to `200092`.
        video_token_id: int. Placeholder token id expanded to per-frame
            video embeddings by the preprocessor. Defaults to `200091`.
    """

    backbone_cls = MuseGlimmerBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|finetune_right_pad|>",
        image_token_id=200092,
        video_token_id=200091,
        **kwargs,
    ):
        self._add_special_token(bos_token, "start_token")
        self._add_special_token(eos_token, "end_token")
        self._add_special_token(pad_token, "pad_token")
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
            }
        )
        return config
