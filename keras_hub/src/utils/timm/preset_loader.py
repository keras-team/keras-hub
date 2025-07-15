"""Convert timm models to KerasHub."""

from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.utils.preset_utils import PresetLoader
from keras_hub.src.utils.preset_utils import jax_memory_cleanup
from keras_hub.src.utils.timm import convert_cspnet
from keras_hub.src.utils.timm import convert_densenet
from keras_hub.src.utils.timm import convert_efficientnet
from keras_hub.src.utils.timm import convert_mobilenet
from keras_hub.src.utils.timm import convert_resnet
from keras_hub.src.utils.timm import convert_vgg
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader


class TimmPresetLoader(PresetLoader):
    def __init__(self, preset, config):
        super().__init__(preset, config)
        architecture = self.config["architecture"]
        if architecture.startswith("resnet"):
            self.converter = convert_resnet
        elif architecture.startswith(("csp", "dark")):
            self.converter = convert_cspnet
        elif architecture.startswith("densenet"):
            self.converter = convert_densenet
        elif architecture.startswith("mobilenet"):
            self.converter = convert_mobilenet
        elif architecture.startswith("vgg"):
            self.converter = convert_vgg
        elif architecture.startswith("efficientnet"):
            self.converter = convert_efficientnet
        else:
            raise ValueError(
                "KerasHub has no converter for timm models "
                f"with architecture `'{architecture}'`."
            )

    def check_backbone_class(self):
        return self.converter.backbone_cls

    def load_backbone(self, cls=None, load_weights=True, kwargs=None):
        kwargs = kwargs or {}
        keras_config = self.converter.convert_backbone_config(self.config)
        cls = self.check_backbone_class()
        backbone = cls(**{**keras_config, **kwargs})
        if load_weights:
            jax_memory_cleanup(backbone)
            # Use prefix="" to avoid using `get_prefixed_key`.
            with SafetensorLoader(self.preset, prefix="") as loader:
                self.converter.convert_weights(backbone, loader, self.config)
        return backbone

    def load_task(self, cls, load_weights=True, kwargs=None):
        kwargs = kwargs or {}
        if not issubclass(cls, ImageClassifier) or "num_classes" in kwargs:
            return super().load_task(cls, load_weights, kwargs)
        # Support loading the classification head for classifier models.
        if "num_classes" in self.config:
            kwargs["num_classes"] = self.config["num_classes"]
        # TODO: Move arch specific config to the converter.
        if (
            self.config["architecture"].startswith("mobilenet")
            and "num_features" not in kwargs
            and "num_features" in self.config
        ):
            kwargs["num_features"] = self.config["num_features"]
        task = super().load_task(cls, load_weights, kwargs)
        if load_weights:
            with SafetensorLoader(self.preset, prefix="") as loader:
                self.converter.convert_head(task, loader, self.config)
        return task

    def load_image_converter(self, cls=None, kwargs=None):
        kwargs = kwargs or {}
        cls = self.find_compatible_subclass(cls or ImageConverter)
        pretrained_cfg = self.config.get("pretrained_cfg", None)
        if not pretrained_cfg or "input_size" not in pretrained_cfg:
            return None
        # This assumes the same basic setup for all timm preprocessing, We may
        # need to extend this as we cover more model types.
        defaults = {}
        defaults["image_size"] = pretrained_cfg["input_size"][1:]
        mean = pretrained_cfg["mean"]
        std = pretrained_cfg["std"]
        defaults["scale"] = [1.0 / 255.0 / s for s in std]
        defaults["offset"] = [-m / s for m, s in zip(mean, std)]
        interpolation = pretrained_cfg["interpolation"]
        if interpolation not in ("bilinear", "nearest", "bicubic"):
            interpolation = "bilinear"  # Unsupported interpolation type.
        defaults["interpolation"] = interpolation
        return cls(**{**defaults, **kwargs})
