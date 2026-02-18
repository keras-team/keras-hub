import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer


@keras_hub_export("keras_hub.models.Qwen2VLCausalLMPreprocessor")
class Qwen2VLCausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen2-VL Causal LM Preprocessor.

    Handles tokenization, image preprocessing, and assembly of the full
    input dict required by ``Qwen2VLBackbone``.

    When images are provided the preprocessor:
    1. Runs ``image_converter`` to get flat patches and ``grid_thw``.
    2. Computes the number of image placeholder tokens as
       ``grid_t * grid_h * grid_w // spatial_merge_size²``.
    3. Inserts ``<|vision_start|>`` + N × ``<|image_pad|>`` +
       ``<|vision_end|>`` tokens into the text token sequence at the
       position of the first ``<|image|>`` marker (or prepends them if
       no marker is present).
    4. Pads / truncates to ``sequence_length`` and builds ``padding_mask``.

    Returns a dict with keys:
    - ``"token_ids"``: int32 array of shape ``(seq_len,)``.
    - ``"padding_mask"``: int32 array of shape ``(seq_len,)``.
    - ``"patch_values"``: float32 array of shape
      ``(total_patches, patch_flat_dim)`` or ``None``.
    - ``"image_grid_thw"``: int32 array of shape ``(num_images, 3)`` or
      ``None``.

    Args:
        tokenizer: A ``keras_hub`` tokenizer instance.
        image_converter: A ``Qwen2VLImageConverter`` instance or ``None``.
        sequence_length: int. Maximum token sequence length. Defaults to
            ``1024``.
        spatial_merge_size: int. Must match the backbone's
            ``spatial_merge_size``. Used to compute the number of image
            placeholder tokens. Defaults to ``2``.
    """

    backbone_cls = Qwen2VLBackbone
    tokenizer_cls = Qwen2VLTokenizer
    image_converter_cls = Qwen2VLImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            **kwargs,
        )
        self.image_converter = image_converter
        self.spatial_merge_size = spatial_merge_size

    def generate_preprocess(self, x, sequence_length=None):
        """Preprocess a single example for generation.

        Args:
            x: Either a plain string, or a dict with keys ``"text"``
               (str) and optionally ``"images"`` (NumPy array).
            sequence_length: int or ``None``. Overrides
                ``self.sequence_length`` when provided.

        Returns:
            Dict with keys ``"token_ids"``, ``"padding_mask"``,
            ``"patch_values"``, ``"image_grid_thw"``.
        """
        seq_len = sequence_length or self.sequence_length

        if isinstance(x, dict):
            text = x.get("text", "")
            images = x.get("images", None)
        else:
            text = x
            images = None

        # Tokenize text
        token_ids = self.tokenizer(text)
        if hasattr(token_ids, "numpy"):
            token_ids = token_ids.numpy()
        token_ids = np.asarray(token_ids, dtype="int32").reshape(-1)

        patch_values = None
        grid_thw = None

        if images is not None and self.image_converter is not None:
            patches, grid_thw = self.image_converter.call(images)
            patch_values = patches

            # Build vision token blocks for all images.
            vision_blocks = []
            for i in range(grid_thw.shape[0]):
                gt = int(grid_thw[i, 0])
                gh = int(grid_thw[i, 1])
                gw = int(grid_thw[i, 2])
                num_vision_tokens = (gt * gh * gw) // (
                    self.spatial_merge_size**2
                )
                vision_block = np.array(
                    [self.tokenizer.vision_start_token_id]
                    + [self.tokenizer.image_pad_token_id] * num_vision_tokens
                    + [self.tokenizer.vision_end_token_id],
                    dtype="int32",
                )
                vision_blocks.append(vision_block)

            # Insert vision blocks: replace image markers if present,
            # otherwise prepend all blocks.
            img_marker_positions = np.where(
                token_ids == self.tokenizer.image_pad_token_id
            )[0]
            if len(img_marker_positions) > 0:
                # Replace the first marker with all vision blocks
                # concatenated (multi-image).
                pos = img_marker_positions[0]
                combined_block = np.concatenate(vision_blocks)
                token_ids = np.concatenate(
                    [token_ids[:pos], combined_block, token_ids[pos + 1 :]]
                )
            else:
                combined_block = np.concatenate(vision_blocks)
                token_ids = np.concatenate([combined_block, token_ids])

        # Pad or truncate to seq_len
        current_len = len(token_ids)
        if current_len >= seq_len:
            token_ids = token_ids[:seq_len]
            padding_mask = np.ones(seq_len, dtype="int32")
        else:
            pad_len = seq_len - current_len
            padding_mask = np.concatenate(
                [
                    np.ones(current_len, dtype="int32"),
                    np.zeros(pad_len, dtype="int32"),
                ]
            )
            token_ids = np.concatenate(
                [token_ids, np.zeros(pad_len, dtype="int32")]
            )

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "patch_values": patch_values,
            "image_grid_thw": grid_thw,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config
