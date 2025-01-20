import keras
import math
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_ocr import ImageOCR

PARSEQ_ALPHABET = (
    "\00"
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)


@keras_hub_export("keras_hub.models.PARSeqOCR")
class PARSeqOCR(ImageOCR):
    """Scene Text Detection with PARSeq.

    Performs OCR in natural scenes using the PARSeq model described in [Scene
    Text Recognition with Permuted Autoregressive Sequence Models](
    https://arxiv.org/abs/2207.06966). PARSeq is a ViT-based model that allows
    iterative decoding by performing an autoregressive decoding phase, followed
    by a refinement phase.

    Args:
        backbone: A `keras_hub.models.PARSeqBackbone` instance.
        preprocessor: `None`, a `keras_hub.models.Preprocessor` instance,
            a `keras.Layer` instance, or a callable. If `None` no preprocessing
            will be applied to the inputs.
        num_permutations: int. The number of permutation to generate for
            training. Has no impact for model inference. Defaults to 6.
        add_forward_perms: bool. Whether to always include the autoregressive
            permutation for training. Has no impact for model inference.
            Defaults to True.
        add_mirrored_perms: True. Whether to include the permutations' reversed
            sequences for training. Has no impact for model inference.
            Defaults to True.
        perm_seed_generator: A `keras.random.SeedGenerator` instance or None.
            The seed generator to use when sampling training permutations. If
            not provided, a new `SeedGenerator` will be created.
        image_shape: tuple. The input shape without the batch size.
            Defaults to `(32, 128, 3)`.

    Examples:
    ```python
    input_data = np.random.uniform(0, 1, size=(2, 32, 128, 3))

    # Initialize a PARSeq instance
    backbone = keras_hub.models.PARSeqBackbone()
    model = keras_hub.models.PARSeqOCR(backbone)

    # Perform iterative autoregressive text recognition
    probabilities = model(input_data)

    # Transform the probability output to a text string
    model.detokenize(probabilities)
    ```
    """

    def __init__(
        self,
        backbone,
        preprocessor=None,
        num_permutations=6,
        add_forward_perms=True,
        add_mirrored_perms=True,
        perm_seed_generator=None,
        image_shape=(32, 128, 3),
        **kwargs,
    ):
        if perm_seed_generator is None:
            perm_seed_generator = keras.random.SeedGenerator(seed=42)

        # === Functional Model ===
        x = keras.Input(shape=image_shape)
        outputs = backbone(x)
        super().__init__(inputs=x, outputs=outputs, **kwargs)

        # === Config ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.add_forward_perms = add_forward_perms
        self.add_mirrored_perms = add_mirrored_perms
        self.num_permutations = num_permutations
        self.perm_seed_generator = perm_seed_generator

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "add_forward_perms": self.add_forward_perms,
                "add_mirrored_perms": self.add_mirrored_perms,
                "num_permutations": self.num_permutations,
                "perm_seed_generator": keras.saving.serialize_keras_object(
                    self.perm_seed_generator
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["perm_seed_generator"] = keras.saving.deserialize_keras_object(
            config["perm_seed_generator"]
        )
        return super().from_config(config)

    def detokenize(self, probabilities):
        probabilities = ops.convert_to_numpy(probabilities)
        predictions = []
        for probs_sequence in probabilities:
            tokens = probs_sequence.argmax(axis=1)
            length = (tokens == self.backbone.eos_id) or tokens.shape[0]
            predictions.append(
                "".join(PARSEQ_ALPHABET[i] for i in tokens[:length])
            )
        return predictions

    def compute_loss(self, x, y, y_pred, *args, **kwargs):
        # Unlike inference, training of PARSeq requires generating a number of
        # permutations that the model is trained against. We might implement
        # this specific training behaviour either in a `train_step` method, or
        # in a `compute_loss`, which ignores `y_pred`. Since implementing a
        # `train_step` is fairly complex due to the need to handle different
        # backends, we implement `compute_loss`.
        max_num_chars = y.shape[1] - 1  # -1 to account for the EOS token
        perms = self.generate_training_permutations(max_num_chars)
        logits_list = self.backbone(
            x, training_tokens=y, training_token_perms=perms, training=True
        )
        losses = []
        for i in range(logits_list.shape[0]):
            losses.append(
                super().compute_loss(x, y, logits_list[i], *args, **kwargs)
            )
        return ops.sum(losses, axis=0)

    def generate_training_permutations(self, max_num_chars):
        max_gen_perms = (
            self.num_permutations // 2
            if self.add_mirrored_perms
            else self.num_permutations
        )
        if max_num_chars == 1:
            return ops.expand_dims(ops.arange(3), axis=0)
        perms = [ops.arange(max_num_chars)] if self.add_forward_perms else []
        max_perms = math.factorial(max_num_chars)
        if self.add_mirrored_perms:
            max_perms //= 2
        num_gen_perms = min(max_gen_perms, max_perms)
        for _ in range(num_gen_perms - len(perms)):
            perms.append(
                keras.random.shuffle(
                    ops.arange(max_num_chars), seed=self.perm_seed_generator
                )
            )
        perms = ops.stack(perms)
        if self.add_mirrored_perms:
            comp = perms[:, ::-1]
            perms = ops.concatenate([perms, comp], axis=0)
        bos_idx = ops.zeros((ops.shape(perms)[0], 1), "int32")
        eos_idx = ops.full((ops.shape(perms)[0], 1), max_num_chars + 1, "int32")
        perms = ops.concatenate([bos_idx, perms + 1, eos_idx], axis=1)
        return perms
