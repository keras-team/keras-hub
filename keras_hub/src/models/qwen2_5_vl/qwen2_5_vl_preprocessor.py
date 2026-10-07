import math
import numpy as np
import keras
from keras import ops

from qwen2_5_vl_tokenizer import Qwen2_5_VLTokenizer


# ── CLIP normalization constants (from preprocessor_config.json) ───────────────
IMAGE_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
IMAGE_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# ── Default pixel budget (from preprocessor_config.json) ──────────────────────
DEFAULT_MIN_PIXELS = 3_136        # 56 × 56
DEFAULT_MAX_PIXELS = 12_845_056   # ~3584 × 3584


# ── Pure-Python image resize helpers ──────────────────────────────────────────

def _round_to_multiple(x, multiple):
    """Round x up to the nearest multiple."""
    return int(math.ceil(x / multiple) * multiple)


def _smart_resize(h, w, patch_size=14, min_pixels=DEFAULT_MIN_PIXELS,
                  max_pixels=DEFAULT_MAX_PIXELS):
    """
    Resize (h, w) so that:
      - Both dimensions are multiples of patch_size.
      - Total pixels stay within [min_pixels, max_pixels].
      - Aspect ratio is preserved as closely as possible.

    Returns
    -------
    (new_h, new_w) : int, int
    """
    # Snap to patch grid while preserving aspect ratio
    scale = math.sqrt(min_pixels / (h * w))
    if h * w < min_pixels:
        h = _round_to_multiple(h * scale, patch_size)
        w = _round_to_multiple(w * scale, patch_size)

    scale = math.sqrt(max_pixels / (h * w))
    if h * w > max_pixels:
        h = _round_to_multiple(h * scale, patch_size)
        w = _round_to_multiple(w * scale, patch_size)

    # Final snap — ensure multiples of patch_size
    h = _round_to_multiple(h, patch_size)
    w = _round_to_multiple(w, patch_size)

    # Clamp pixel budget
    h = max(h, patch_size)
    w = max(w, patch_size)

    return h, w


def _resize_image(image_np, new_h, new_w):
    """
    Resize a numpy image (H, W, 3) to (new_h, new_w, 3) using PIL.
    PIL is available in every Python environment and produces
    high-quality Lanczos resampling matching the HF implementation.
    """
    from PIL import Image
    import tensorflow as tf
    # Accept both numpy arrays and TF/Keras tensors
    if hasattr(image_np, "numpy"):
        image_np = image_np.numpy()
    image_np = np.asarray(image_np)
    img = Image.fromarray(image_np.astype(np.uint8))
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def _normalize(image_np):
    """
    Normalize image (H, W, 3) float32 in [0, 255] to CLIP range.
    """
    image_np = image_np / 255.0
    image_np = (image_np - IMAGE_MEAN) / IMAGE_STD
    return image_np.astype(np.float32)


def preprocess_image(image, patch_size=14, temporal_patch_size=2,
                     min_pixels=DEFAULT_MIN_PIXELS,
                     max_pixels=DEFAULT_MAX_PIXELS):
    """
    Preprocess a single image for Qwen2.5-VL.

    Follows the official Qwen2VLImageProcessor pipeline:
      1. Smart resize to patch-aligned dimensions within pixel budget
      2. CLIP normalization
      3. Pack into (T, H, W, 3) with T = temporal_patch_size (duplicate frame)

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W, 3) uint8 or float32.
    patch_size : int
    temporal_patch_size : int
    min_pixels : int
    max_pixels : int

    Returns
    -------
    pixel_values : np.ndarray  (T, new_h, new_w, 3)  float32
    grid_thw    : tuple        (T, new_h // patch_size, new_w // patch_size)
    """
    if image.ndim == 2:
        # Grayscale → RGB
        image = np.stack([image] * 3, axis=-1)

    h, w = image.shape[:2]
    new_h, new_w = _smart_resize(h, w, patch_size, min_pixels, max_pixels)

    image = _resize_image(image, new_h, new_w)
    image = _normalize(image)

    # Duplicate frame to satisfy temporal_patch_size = 2
    # (single images are treated as a 2-frame "video" with identical frames)
    frames = np.stack([image] * temporal_patch_size, axis=0)  # (T, H, W, 3)

    T  = temporal_patch_size
    pH = new_h // patch_size
    pW = new_w // patch_size

    return frames, (T, pH, pW)


def compute_num_vision_tokens(grid_thw, merge_size=2):
    """
    Compute the number of vision tokens after patch merging.

    After Conv3D patch embed:   N = T * pH * pW
    After 2×2 spatial merge:   N_merged = T * (pH // merge_size) * (pW // merge_size)

    Parameters
    ----------
    grid_thw : tuple  (T, pH, pW)
    merge_size : int

    Returns
    -------
    int : number of vision tokens
    """
    T, pH, pW = grid_thw
    return T * (pH // merge_size) * (pW // merge_size)


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLPreprocessor(keras.layers.Layer):
    """
    Qwen2.5-VL Preprocessor.

    Converts raw text + images into model-ready inputs:
      token_ids    : (B, S)        int32
      pixel_values : (B, T, H, W, 3) float32
      image_mask   : (B, S)        bool

    The image_mask marks the positions in the token sequence where vision
    tokens will be injected by the backbone. It contains exactly N_vision
    True entries per sample, where N_vision = compute_num_vision_tokens().

    Parameters
    ----------
    tokenizer : Qwen2_5_VLTokenizer
    patch_size : int
        Spatial patch size. Must match the backbone's patch_size.
    temporal_patch_size : int
    merge_size : int
        Spatial merge factor in PatchMerger. Must match the backbone.
    max_sequence_length : int or None
        If set, token sequences are truncated to this length.
    min_pixels : int
    max_pixels : int

    Example
    -------
    preprocessor = Qwen2_5_VLPreprocessor(tokenizer=tokenizer)

    # Text-only
    out = preprocessor(messages=[
        {"role": "user", "content": "Hello!"}
    ])
    # token_ids: (1, S)

    # Multimodal
    out = preprocessor(
        messages=[{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this."},
        ]}],
        images=[np.array(...)],   # list of (H, W, 3) uint8 arrays
    )
    # token_ids: (1, S), pixel_values: (1, T, H, W, 3), image_mask: (1, S)
    """

    def __init__(
        self,
        tokenizer,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        max_sequence_length=None,
        min_pixels=DEFAULT_MIN_PIXELS,
        max_pixels=DEFAULT_MAX_PIXELS,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer          = tokenizer
        self.patch_size         = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size         = merge_size
        self.max_sequence_length = max_sequence_length
        self.min_pixels         = min_pixels
        self.max_pixels         = max_pixels

    def build(self, input_shape=None):
        # Preprocessor has no trainable weights.
        # Marking as built prevents Keras from trying to trace call()
        # symbolically, which would fail because we use .numpy() calls.
        self.built = True

    # ── Main entry point ───────────────────────────────────────────────────

    def call(self, messages, images=None):
        """
        Process a single conversation turn.

        Parameters
        ----------
        messages : list of dict
            Conversation in the standard chat format.
        images : list of np.ndarray or None
            One (H, W, 3) uint8 array per image placeholder in messages.

        Returns
        -------
        dict with keys:
            token_ids    : (1, S) int32 tensor
            pixel_values : (1, T, H, W, 3) float32 tensor  [if images given]
            image_mask   : (1, S) bool tensor               [if images given]
        """
        # 1. Apply chat template → prompt string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # 2. Tokenize prompt → 1-D int32 tensor
        token_ids = self.tokenizer(prompt)                   # (S,)
        token_ids_np = token_ids.numpy().tolist()

        # 3. Truncate if needed
        if self.max_sequence_length is not None:
            token_ids_np = token_ids_np[:self.max_sequence_length]

        if images is None or len(images) == 0:
            token_ids_out = keras.ops.convert_to_tensor(
                [token_ids_np], dtype="int32"
            )
            return {"token_ids": token_ids_out}

        # 4. Preprocess each image
        all_frames = []
        all_grids  = []
        for img in images:
            # Accept TF tensors, Keras tensors, or numpy arrays
            if hasattr(img, "numpy"):
                img = img.numpy()
            img = np.asarray(img)
            frames, grid_thw = preprocess_image(
                img,
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            all_frames.append(frames)
            all_grids.append(grid_thw)

        # 5. Build image_mask
        # The tokenizer inserts one <|image_pad|> per image placeholder.
        # The backbone replaces each image_pad position with N_vision tokens.
        # We need to expand those single placeholder tokens into N_vision slots.
        image_pad_id  = self.tokenizer.image_pad_id
        vision_start_id = self.tokenizer.vision_start_id
        vision_end_id   = self.tokenizer.vision_end_id

        new_token_ids = []
        image_index   = 0
        i = 0

        while i < len(token_ids_np):
            tok = token_ids_np[i]

            if tok == vision_start_id:
                # Find the matching vision_end
                j = i + 1
                while j < len(token_ids_np) and token_ids_np[j] != vision_end_id:
                    j += 1

                # Replace the entire <|vision_start|>...<|vision_end|> span
                # with vision_start + N_vision × image_pad + vision_end
                if image_index < len(all_grids):
                    n_vision = compute_num_vision_tokens(
                        all_grids[image_index], self.merge_size
                    )
                    new_token_ids.append(vision_start_id)
                    new_token_ids.extend([image_pad_id] * n_vision)
                    new_token_ids.append(vision_end_id)
                    image_index += 1
                else:
                    # No image available — keep original tokens
                    new_token_ids.extend(token_ids_np[i:j + 1])

                i = j + 1
            else:
                new_token_ids.append(tok)
                i += 1

        # 6. Truncate expanded sequence if needed
        if self.max_sequence_length is not None:
            new_token_ids = new_token_ids[:self.max_sequence_length]

        seq_len = len(new_token_ids)

        # 7. Build image_mask: True at every image_pad_id position
        image_mask = [tid == image_pad_id for tid in new_token_ids]

        # 8. Stack pixel_values: use first image for now
        #    For multi-image support, each image gets its own pixel_values call
        frames = all_frames[0]                               # (T, H, W, 3)

        token_ids_out   = keras.ops.convert_to_tensor([new_token_ids], dtype="int32")
        pixel_values_out = keras.ops.convert_to_tensor(
            frames[np.newaxis], dtype="float32"
        )                                                    # (1, T, H, W, 3)
        image_mask_out  = keras.ops.convert_to_tensor([image_mask], dtype="bool")

        return {
            "token_ids":    token_ids_out,    # (1, S)
            "pixel_values": pixel_values_out, # (1, T, H, W, 3)
            "image_mask":   image_mask_out,   # (1, S)
        }

    # ── Serialization ──────────────────────────────────────────────────────

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer": keras.saving.serialize_keras_object(
                    self.tokenizer
                ),
                "patch_size":          self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "merge_size":          self.merge_size,
                "max_sequence_length": self.max_sequence_length,
                "min_pixels":          self.min_pixels,
                "max_pixels":          self.max_pixels,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["tokenizer"] = keras.saving.deserialize_keras_object(
            config.pop("tokenizer")
        )
        return cls(**config)