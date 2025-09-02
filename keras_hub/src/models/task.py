import keras
from rich import console as rich_console
from rich import markup
from rich import table as rich_table

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.tokenizers.tokenizer import Tokenizer
from keras_hub.src.utils.keras_utils import print_msg
from keras_hub.src.utils.pipeline_model import PipelineModel
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import find_subclass
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import get_preset_saver
from keras_hub.src.utils.python_utils import classproperty


@keras_hub_export("keras_hub.models.Task")
class Task(PipelineModel):
    """Base class for all Task models.

    A `Task` wraps a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be directly
    used for training, fine-tuning, and prediction for a given text problem.

    All `Task` models have `backbone` and `preprocessor` properties. By
    default `fit()`, `predict()` and `evaluate()` will preprocess all inputs
    automatically. To preprocess inputs separately or with a custom function,
    you can set `task.preprocessor = None`, which disable any automatic
    preprocessing on inputs.

    All `Task` classes include a `from_preset()` constructor which can be used
    to load a pre-trained config and weights. Calling `from_preset()` on a task
    will automatically instantiate a `keras_hub.models.Backbone` and
    `keras_hub.models.Preprocessor`.

    Args:
        compile: boolean, defaults to `True`. If `True` will compile the model
            with default parameters on construction. Model can still be
            recompiled with a new loss, optimizer and metrics before training.
    """

    backbone_cls = None
    preprocessor_cls = None

    def __init__(self, *args, compile=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )
        self._initialized = True
        if self.backbone is not None:
            self.dtype_policy = self._backbone.dtype_policy
        if compile:
            # Default compilation.
            self.compile()

    def preprocess_samples(self, x, y=None, sample_weight=None):
        # If `preprocessor` is `None`, return inputs unaltered.
        if self.preprocessor is None:
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
        # If `preprocessor` is `Preprocessor` subclass, pass labels as a kwarg.
        if isinstance(self.preprocessor, Preprocessor):
            return self.preprocessor(x, y=y, sample_weight=sample_weight)
        # For other layers and callable, do not pass the label.
        x = self.preprocessor(x)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def __setattr__(self, name, value):
        # Work around setattr issues for Keras 2 and Keras 3 torch backend.
        # Since all our state is covered by functional model we can route
        # around custom setattr calls.
        is_property = isinstance(getattr(type(self), name, None), property)
        is_unitialized = not hasattr(self, "_initialized")
        is_torch = keras.config.backend() == "torch"
        if is_torch and (is_property or is_unitialized):
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def backbone(self):
        """A `keras_hub.models.Backbone` model with the core architecture."""
        return getattr(self, "_backbone", None)

    @backbone.setter
    def backbone(self, value):
        self._backbone = value

    @property
    def preprocessor(self):
        """A `keras_hub.models.Preprocessor` layer used to preprocess input."""
        return getattr(self, "_preprocessor", None)

    @preprocessor.setter
    def preprocessor(self, value):
        self._preprocessor = value

    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to our Task constructors.
        return {
            "backbone": keras.layers.serialize(self.backbone),
            "preprocessor": keras.layers.serialize(self.preprocessor),
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        if "preprocessor" in config and isinstance(
            config["preprocessor"], dict
        ):
            config["preprocessor"] = keras.layers.deserialize(
                config["preprocessor"]
            )
        return cls(**config)

    @classproperty
    def presets(cls):
        """List built-in presets for a `Task` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate a `keras_hub.models.Task` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as
        one of:

        1. a built-in preset identifier like `'bert_base_en'`
        2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
        3. a Hugging Face handle like `'hf://user/bert_base_en'`
        4. a path to a local preset directory like `'./bert_base_en'`

        For any `Task` subclass, you can run `cls.presets.keys()` to list all
        built-in presets available on the class.

        This constructor can be called in one of two ways. Either from a task
        specific base class like `keras_hub.models.CausalLM.from_preset()`, or
        from a model class like
        `keras_hub.models.BertTextClassifier.from_preset()`.
        If calling from the a base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, saved weights will be loaded into
                the model architecture. If `False`, all weights will be
                randomly initialized.

        Examples:
        ```python
        # Load a Gemma generative task.
        causal_lm = keras_hub.models.CausalLM.from_preset(
            "gemma_2b_en",
        )

        # Load a Bert classification task.
        model = keras_hub.models.TextClassifier.from_preset(
            "bert_base_en",
            num_classes=2,
        )
        ```
        """
        if cls == Task:
            raise ValueError(
                "Do not call `Task.from_preset()` directly. Instead call a "
                "particular task class, e.g. "
                "`keras_hub.models.TextClassifier.from_preset()`."
            )

        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        # Detect the correct subclass if we need to.
        if (
            issubclass(backbone_cls, Backbone)
            and cls.backbone_cls != backbone_cls
        ):
            cls = find_subclass(preset, cls, backbone_cls)
        # Specifically for classifiers, we never load task weights if
        # num_classes is supplied. We handle this in the task base class because
        # it is the same logic for classifiers regardless of modality (text,
        # images, audio).
        load_task_weights = "num_classes" not in kwargs
        return loader.load_task(cls, load_weights, load_task_weights, **kwargs)

    def load_task_weights(self, filepath):
        """Load only the tasks specific weights not in the backbone."""
        if not str(filepath).endswith(".weights.h5"):
            raise ValueError(
                "The filename must end in `.weights.h5`. "
                f"Received: filepath={filepath}"
            )
        backbone_layer_ids = set(id(w) for w in self.backbone._flatten_layers())
        keras.saving.load_weights(
            self,
            filepath,
            objects_to_skip=backbone_layer_ids,
        )

    def has_task_weights(self):
        task_weight_ids = set(id(w) for w in self.weights)
        backbone_weight_ids = set(id(w) for w in self.backbone.weights)
        return not task_weight_ids.issubset(backbone_weight_ids)

    def save_task_weights(self, filepath):
        """Save only the tasks specific weights not in the backbone."""
        if not str(filepath).endswith(".weights.h5"):
            raise ValueError(
                "The filename must end in `.weights.h5`. "
                f"Received: filepath={filepath}"
            )

        backbone_layer_ids = set(id(w) for w in self.backbone._flatten_layers())
        if not self.has_task_weights():
            raise ValueError(
                f"Task {self} has no weights not in the `backbone`. "
                "`save_task_weights()` has nothing to save."
            )
        keras.saving.save_weights(
            self,
            filepath=filepath,
            objects_to_skip=backbone_layer_ids,
        )

    def save_to_preset(self, preset_dir, max_shard_size=10):
        """Save task to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
            max_shard_size: `int` or `float`. Maximum size in GB for each
                sharded file. If `None`, no sharding will be done. Defaults to
                `10`.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_task(self, max_shard_size=max_shard_size)

    @property
    def layers(self):
        # Remove preprocessor from layers so it does not show up in the summary.
        layers = super().layers
        if self.preprocessor and self.preprocessor in layers:
            layers.remove(self.preprocessor)
        return layers

    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        **kwargs,
    ):
        """Override `model.summary()` to show a preprocessor if set."""

        # Compat fixes for tf.keras.
        if not hasattr(self, "compiled"):
            self.compiled = getattr(self.optimizer, "_is_compiled", False)
        if (
            self.compiled
            and self.optimizer
            and not hasattr(self.optimizer, "built")
        ):
            self.optimizer.built = getattr(self.optimizer, "_built", False)

        # Below is copied from keras-core for now.
        # We should consider an API contract.
        line_length = line_length or 108

        if not print_fn and not keras.utils.is_interactive_logging_enabled():
            print_fn = print_msg

        def highlight_number(x):
            if x is None:
                return f"[color(45)]{x}[/]"
            return f"[color(34)]{x:,}[/]"  # Format number with commas.

        def highlight_symbol(x):
            return f"[color(33)]{x}[/]"

        def bold_text(x):
            return f"[bold]{x}[/]"

        def highlight_shape(shape):
            highlighted = [highlight_number(x) for x in shape]
            return "(" + ", ".join(highlighted) + ")"

        if self.preprocessor:
            # Create a rich console for printing. Capture for non-interactive
            # logging.
            if print_fn:
                console = rich_console.Console(
                    highlight=False, force_terminal=False, color_system=None
                )
                console.begin_capture()
            else:
                console = rich_console.Console(highlight=False)

            column_1 = rich_table.Column(
                "Layer (type)",
                justify="left",
                width=int(0.6 * line_length),
            )
            column_2 = rich_table.Column(
                "Config",
                justify="right",
                width=int(0.4 * line_length),
            )
            table = rich_table.Table(
                column_1, column_2, width=line_length, show_lines=True
            )

            def add_layer(layer, info):
                layer_name = markup.escape(layer.name)
                layer_class = highlight_symbol(
                    markup.escape(layer.__class__.__name__)
                )
                table.add_row(
                    f"{layer_name} ({layer_class})",
                    info,
                )

            # Since the preprocessor might be nested with multiple `Tokenizer`,
            # `ImageConverter`, `AudioConverter` and even other `Preprocessor`
            # instances, we should recursively iterate through them.
            preprocessor = self.preprocessor
            if preprocessor and isinstance(preprocessor, keras.Layer):
                for layer in preprocessor._flatten_layers(include_self=False):
                    if isinstance(layer, Tokenizer):
                        info = "Vocab size: "
                        info += highlight_number(layer.vocabulary_size())
                        add_layer(layer, info)
                    elif isinstance(layer, ImageConverter):
                        info = "Image size: "
                        image_size = layer.image_size
                        if image_size is None:
                            image_size = (None, None)
                        info += highlight_shape(image_size)
                        add_layer(layer, info)
                    elif isinstance(layer, AudioConverter):
                        info = "Audio shape: "
                        info += highlight_shape(layer.audio_shape())
                        add_layer(layer, info)

            # Print the to the console.
            preprocessor_name = markup.escape(preprocessor.name)
            console.print(bold_text(f'Preprocessor: "{preprocessor_name}"'))
            console.print(table)

            # Output captured summary for non-interactive logging.
            if print_fn:
                print_fn(console.end_capture(), line_break=False)

        super().summary(
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            **kwargs,
        )
