"""Convert timm models to KerasHub."""

from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.utils.preset_utils import PresetLoader
from keras_hub.src.utils.preset_utils import jax_memory_cleanup
from keras_hub.src.utils.timm import convert_densenet
from keras_hub.src.utils.timm import convert_efficientnet
from keras_hub.src.utils.timm import convert_resnet
from keras_hub.src.utils.timm import convert_vgg
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader


class TimmPresetLoader(PresetLoader):
    def __init__(self, preset, config):
        super().__init__(preset, config)
        architecture = self.config["architecture"]
        if "resnet" in architecture:
            self.converter = convert_resnet
        elif "densenet" in architecture:
            self.converter = convert_densenet
        elif "vgg" in architecture:
            self.converter = convert_vgg
        elif "efficientnet" in architecture:
            self.converter = convert_efficientnet
        else:
            raise ValueError(
                "KerasHub has no converter for timm models "
                f"with architecture `'{architecture}'`."
            )

    def check_backbone_class(self):
        return self.converter.backbone_cls

    def load_backbone(self, cls, load_weights, **kwargs):
        keras_config = self.converter.convert_backbone_config(self.config)
        backbone = cls(**{**keras_config, **kwargs})
        if load_weights:
            jax_memory_cleanup(backbone)
            # Use prefix="" to avoid using `get_prefixed_key`.
            with SafetensorLoader(self.preset, prefix="") as loader:
                self.converter.convert_weights(backbone, loader, self.config)
        return backbone

    def load_task(self, cls, load_weights, load_task_weights, **kwargs):
        if not load_task_weights or not issubclass(cls, ImageClassifier):
            return super().load_task(
                cls, load_weights, load_task_weights, **kwargs
            )
        # Support loading the classification head for classifier models.
        kwargs["num_classes"] = self.config["num_classes"]
        task = super().load_task(cls, load_weights, load_task_weights, **kwargs)
        if load_task_weights:
            with SafetensorLoader(self.preset, prefix="") as loader:
                self.converter.convert_head(task, loader, self.config)
        return task

    def load_image_converter(self, cls, **kwargs):
        pretrained_cfg = self.config.get("pretrained_cfg", None)
        if not pretrained_cfg or "input_size" not in pretrained_cfg:
            return None
        # This assumes the same basic setup for all timm preprocessing, We may
        # need to extend this as we cover more model types.
        input_size = pretrained_cfg["input_size"]
        mean = pretrained_cfg["mean"]
        std = pretrained_cfg["std"]
        scale = [1.0 / 255.0 / s for s in std]
        offset = [-m / s for m, s in zip(mean, std)]
        interpolation = pretrained_cfg["interpolation"]
        if interpolation not in ("bilinear", "nearest", "bicubic"):
            interpolation = "bilinear"  # Unsupported interpolation type.
        return cls(
            image_size=input_size[1:],
            scale=scale,
            offset=offset,
            interpolation=interpolation,
        )
