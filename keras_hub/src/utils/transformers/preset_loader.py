"""Convert huggingface models to KerasHub."""

from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.utils.preset_utils import PresetLoader
from keras_hub.src.utils.preset_utils import jax_memory_cleanup
from keras_hub.src.utils.transformers import convert_albert
from keras_hub.src.utils.transformers import convert_bart
from keras_hub.src.utils.transformers import convert_bert
from keras_hub.src.utils.transformers import convert_distilbert
from keras_hub.src.utils.transformers import convert_gemma
from keras_hub.src.utils.transformers import convert_gpt2
from keras_hub.src.utils.transformers import convert_llama3
from keras_hub.src.utils.transformers import convert_mistral
from keras_hub.src.utils.transformers import convert_pali_gemma
from keras_hub.src.utils.transformers import convert_vit
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader


class TransformersPresetLoader(PresetLoader):
    def __init__(self, preset, config):
        super().__init__(preset, config)
        model_type = self.config["model_type"]
        if model_type == "albert":
            self.converter = convert_albert
        elif model_type == "bart":
            self.converter = convert_bart
        elif model_type == "bert":
            self.converter = convert_bert
        elif model_type == "distilbert":
            self.converter = convert_distilbert
        elif model_type == "gemma" or model_type == "gemma2":
            self.converter = convert_gemma
        elif model_type == "gpt2":
            self.converter = convert_gpt2
        elif model_type == "llama":
            # TODO: handle other llama versions.
            self.converter = convert_llama3
        elif model_type == "mistral":
            self.converter = convert_mistral
        elif model_type == "paligemma":
            self.converter = convert_pali_gemma
        elif model_type == "vit":
            self.converter = convert_vit
        else:
            raise ValueError(
                "KerasHub has no converter for huggingface/transformers models "
                f"with model type `'{model_type}'`."
            )

    def check_backbone_class(self):
        return self.converter.backbone_cls

    def load_backbone(self, cls, load_weights, **kwargs):
        keras_config = self.converter.convert_backbone_config(self.config)
        backbone = cls(**{**keras_config, **kwargs})
        if load_weights:
            jax_memory_cleanup(backbone)
            with SafetensorLoader(self.preset) as loader:
                self.converter.convert_weights(backbone, loader, self.config)
        return backbone

    def load_task(self, cls, load_weights, load_task_weights, **kwargs):
        architecture = self.config["architectures"][0]
        if (
            not load_task_weights
            or not issubclass(cls, ImageClassifier)
            or architecture == "ViTModel"
        ):
            return super().load_task(
                cls, load_weights, load_task_weights, **kwargs
            )
        # Support loading the classification head for classifier models.
        if architecture == "ViTForImageClassification":
            kwargs["num_classes"] = len(self.config["id2label"])
        task = super().load_task(cls, load_weights, load_task_weights, **kwargs)
        if load_task_weights:
            with SafetensorLoader(self.preset, prefix="") as loader:
                self.converter.convert_head(task, loader, self.config)
        return task

    def load_tokenizer(self, cls, config_name="tokenizer.json", **kwargs):
        return self.converter.convert_tokenizer(cls, self.preset, **kwargs)

    def load_image_converter(self, cls, **kwargs):
        # TODO: set image size for pali gemma checkpoints.
        return None
