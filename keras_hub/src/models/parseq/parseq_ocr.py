from itertools import permutations

import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task

# ParseQ is currently a 1:1 translation of
# https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/rec_vit_parseq.py
# https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/heads/rec_parseq_head.py
# https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/rec_postprocess.py#L586
# https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/rec_vit_parseq.yml
# I'm in the process of KerasHub'ifying it


@keras_hub_export("keras_hub.models.ParseQOCR")
class ParseQOCR(Task):  # TODO create a task for OCR
    def __init__(
        self,
        backbone,
        perm_num=6,
        perm_forward=True,
        perm_mirrored=True,
        preprocessor=None,
        **kwargs,
    ):
        # === Functional Model ===
        x = backbone.input
        outputs = backbone(x)
        super().__init__(inputs=x, outputs=outputs, **kwargs)

        # === Config ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        self.perm_num = perm_num

    def compute_loss(self, x, y, y_pred, *args, **kwargs):
        # Unlike inference, training of PARSEq requires generating a number of
        # permutations that the model is trained against. We might implement
        # this specific training behaviour either in a `train_step` method, or
        # in a `compute_loss`, which ignores `y_pred`. Since implementing a
        # `train_step` is fairly complex due to the need to handle different
        # backends, we implement `compute_loss`.
        max_num_chars = ops.shape(y)[1] - 2
        perms = self.gen_tgt_perms(max_num_chars)
        logits_list = self.backbone(
            x, training_tgts=y, training_tgt_perms=perms, training=True
        )
        losses = []
        for logits in logits_list:
            losses.append(super().compute_loss(x, y, logits, *args, **kwargs))
        return ops.sum(losses, axis=0)

    def gen_tgt_perms(self, max_num_chars):
        max_gen_perms = (
            self.perm_num // 2 if self.perm_mirrored else self.perm_num
        )
        if max_num_chars == 1:
            return np.expand_dims(np.arange(3), axis=0)

        perms = [np.arange(max_num_chars)] if self.perm_forward else []
        max_perms = np.math.factorial(max_num_chars)

        if self.perm_mirrored:
            max_perms //= 2

        num_gen_perms = min(max_gen_perms, max_perms)

        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = np.arange(max_perms)
            perm_pool = np.array(
                list(permutations(range(max_num_chars), max_num_chars))
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = np.stack(perms)
            if len(perm_pool) > 0:
                i = self.rng.choice(
                    len(perm_pool),
                    size=num_gen_perms - len(perms),
                    replace=False,
                )
                perms = np.concatenate([perms, perm_pool[i]], axis=0)
        else:
            for _ in range(num_gen_perms - len(perms)):
                perms.append(self.rng.permutation(max_num_chars))
            perms = np.stack(perms)

        if self.perm_mirrored:
            comp = perms[:, ::-1]
            perms = np.reshape(
                np.stack([perms, comp], axis=1), [-1, max_num_chars]
            )

        bos_idx = np.zeros((perms.shape[0], 1), dtype=np.int32)
        eos_idx = np.full((perms.shape[0], 1), max_num_chars + 1)
        perms = np.concatenate([bos_idx, perms + 1, eos_idx], axis=1)

        # ensures we always have the reverse AR permutation
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - np.arange(end=max_num_chars + 1)
        return perms
