# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import keras_core
from rich import console as rich_console
from rich import markup
from rich import table as rich_table

from keras_nlp.backend import keras
from keras_nlp.models.lora import add_lora_layers
from keras_nlp.models.lora import merge_lora_layers
from keras_nlp.utils.keras_utils import print_msg
from keras_nlp.utils.pipeline_model import PipelineModel
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.saving.register_keras_serializable(package="keras_nlp")
class Task(PipelineModel):
    """Base class for Task models."""

    def __init__(self, *args, **kwargs):
        self._backbone = None
        self._preprocessor = None
        self.has_lora_layers = False
        super().__init__(*args, **kwargs)

    def _check_for_loss_mismatch(self, loss):
        """Check for a softmax/from_logits mismatch after compile.

        We cannot handle this in the general case, but we can handle this for
        the extremely common case of a single `SparseCategoricalCrossentropy`
        loss, and a `None` or `"softmax"` activation.
        """
        # Only handle a single loss.
        if isinstance(loss, (dict, list, tuple)):
            return
        # Only handle tasks with activation.
        if not hasattr(self, "activation"):
            return

        loss = keras.losses.get(loss)
        activation = keras.activations.get(self.activation)
        if isinstance(loss, keras.losses.SparseCategoricalCrossentropy):
            from_logits = loss.get_config()["from_logits"]
        elif loss == keras.losses.sparse_categorical_crossentropy:
            from_logits = False
        else:
            # Only handle sparse categorical crossentropy.
            return

        softmax_output = activation == keras.activations.softmax
        logit_output = activation == keras.activations.linear
        if softmax_output and from_logits:
            raise ValueError(
                "The `loss` passed to `compile()` expects logit output, but "
                "the model is configured to output softmax probabilities "
                "(`activation='softmax'`). This will not converge! Pass "
                "`from_logits=False` to your loss, e.g. "
                "`loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)`. "
            )
        if logit_output and not from_logits:
            raise ValueError(
                "The `loss` passed to `compile()` expects softmax probability "
                "output, but the model is configured to output logits "
                "(`activation=None`). This will not converge! Pass "
                "`from_logits=True` to your loss, e.g. "
                "`loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)`. "
            )

    def compile(self, optimizer="rmsprop", loss=None, **kwargs):
        self._check_for_loss_mismatch(loss)
        super().compile(optimizer=optimizer, loss=loss, **kwargs)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        return self.preprocessor(x, y=y, sample_weight=sample_weight)

    def __setattr__(self, name, value):
        # Work around torch setattr for properties.
        if name in ["backbone", "preprocessor"]:
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def backbone(self):
        """A `keras.Model` instance providing the backbone submodel."""
        return self._backbone

    @backbone.setter
    def backbone(self, value):
        self._backbone = value

    @property
    def preprocessor(self):
        """A `keras.layers.Layer` instance used to preprocess inputs."""
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self.include_preprocessing = value is not None
        self._preprocessor = value

    @property
    def lora_layer_paths(self):
        """The default layers to target with lora for this model."""
        return self.backbone.lora_layer_paths

    def add_lora_layers(
        self,
        lora_layer_paths=[],
        trainable_weight_paths=[],
        rank=8,
        alpha=32,
    ):
        """Add LoRA adapter layers for this task.

        This method provides high-level API for using [LoRA](https://arxiv.org/pdf/2106.09685.pdf)
        to fine-tune a model in a parameter efficient manner.

        The method will proceed in two steps:
        1) First, freeze all weights in the model besides a configurable set of
           extra trainable weigths. Note that all weights that are not part
           of the backbone will be left trainable (e.g. classification head
           weights).
        2) Second, replace a configurable set of of dense layers with
           `keras_nlp.layers.LoraDense` layers (the parameters are always
           trainable).

        The overall effect will be to dramatically reduce the number of
        trainable parameters in the model, moderately speeding training and
        dramatically reducing memory usage.

        Args:
            lora_layer_paths: A list of string regex paths that will be used to
                target dense layers to replace with `keras_nlp.layers.LoraDense`
                adapter layers. This matching will done by looking at the `path`
                of a dense layers variables. For example, if the `path` of a
                dense kernel is `decoder_layer_4/self_attention/query/kernel`,
                the layer could be matched with `"query"` (to target all query
                layers), or `"decoder_layer_4.*query"` (to target the
                specific layer). If this parameter is unset, the backbone
                defined defaults will be used (usually query and value
                projections).
            trainable_weight_paths: A list of string regex paths that will be
                used to target model parameters to keep trainable. Matching
                will be done in the same manner as with `lora_layer_paths`.
                Pass `["bias"]` to leave all bias paramaters accross the model
                trainable.
            rank: int The inner rank of the decomposed dense transformation. The
                lower this number, the less trainable parameters the layer will
                have.
            alpha: float. A constant value used for scaling the lora update. The
                lora update to the original dense transformation will be scaled by
                `alpha / rank`.
        """
        if not lora_layer_paths:
            lora_layer_paths = self.lora_layer_paths

        # Always leave task specific weights as trainable (e.g. classification
        # heads). We could consider opening this up as a boolean option.
        task_only_weights = set(id(w) for w in self.weights) - set(
            id(w) for w in self.backbone.weights
        )
        for weight in self.weights:
            if id(weight) in task_only_weights and weight.trainable:
                trainable_weight_paths.append(weight.path)

        add_lora_layers(
            self,
            lora_layer_paths=lora_layer_paths,
            trainable_weight_paths=trainable_weight_paths,
            rank=rank,
            alpha=alpha,
        )
        self.has_lora_layers = True

    def merge_lora_layers(self):
        """Merge all LoRA updates and remove LoRA adapter layers.

        This method will merge all LoRA update back into the original dense
        kernels, and restore the model architecture to it's orgiinal state.

        The method will proceed in two steps:
        1) First, merge all lora updates and remove all `LoraDense` layers.
        1) Second, unfreeze all model weights.

        After `merge_lora_layers`, the model will have the same checkpoint
        structure and inference latency as before calling `add_lora_layers()`.
        """
        merge_lora_layers(self)
        self.has_lora_layers = False

    def get_config(self):
        if self.has_lora_layers:
            raise ValueError(
                "Attempting to serialize a model with lora layers. Call "
                "`model.merge_lora_layers()` before saving, cloning, or "
                f"serializing a model. Received: model={self}"
            )
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
    def backbone_cls(cls):
        return None

    @classproperty
    def preprocessor_cls(cls):
        return None

    @classproperty
    def presets(cls):
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate {{model_task_name}} model from preset architecture and weights.

        Args:
            preset: string. Must be one of "{{preset_names}}".
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.

        Examples:
        ```python
        # Load architecture and weights from preset
        model = {{model_task_name}}.from_preset("{{example_preset_name}}")

        # Load randomly initialized model from preset architecture
        model = {{model_task_name}}.from_preset(
            "{{example_preset_name}}",
            load_weights=False
        )
        ```
        """
        if not cls.presets:
            raise NotImplementedError(
                "No presets have been created for this class."
            )

        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        if "preprocessor" not in kwargs:
            kwargs["preprocessor"] = cls.preprocessor_cls.from_preset(preset)

        # Check if preset is backbone-only model
        if preset in cls.backbone_cls.presets:
            backbone = cls.backbone_cls.from_preset(preset, load_weights)
            return cls(backbone, **kwargs)

        # Otherwise must be one of class presets
        metadata = cls.presets[preset]
        config = metadata["config"]
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )

        model.load_weights(weights)
        return model

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to setup a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define `from_preset`, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Task.from_preset.__doc__
            format_docstring(
                model_task_name=cls.__name__,
                example_preset_name=next(iter(cls.presets), ""),
                preset_names='", "'.join(cls.presets),
            )(cls.from_preset.__func__)

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
            return f"[color(45)]{x}[/]" if x is None else f"[color(34)]{x}[/]"

        def highlight_symbol(x):
            return f"[color(33)]{x}[/]"

        def bold_text(x):
            return f"[bold]{x}[/]"

        if self.preprocessor:
            # Create a rich console for printing. Capture for non-interactive logging.
            if print_fn:
                console = rich_console.Console(
                    highlight=False, force_terminal=False, color_system=None
                )
                console.begin_capture()
            else:
                console = rich_console.Console(highlight=False)

            column_1 = rich_table.Column(
                "Tokenizer (type)",
                justify="left",
                width=int(0.5 * line_length),
            )
            column_2 = rich_table.Column(
                "Vocab #",
                justify="right",
                width=int(0.5 * line_length),
            )
            table = rich_table.Table(
                column_1, column_2, width=line_length, show_lines=True
            )
            tokenizer = self.preprocessor.tokenizer
            tokenizer_name = markup.escape(tokenizer.name)
            tokenizer_class = highlight_symbol(
                markup.escape(tokenizer.__class__.__name__)
            )
            table.add_row(
                f"{tokenizer_name} ({tokenizer_class})",
                highlight_number(f"{tokenizer.vocabulary_size():,}"),
            )

            # Print the to the console.
            preprocessor_name = markup.escape(self.preprocessor.name)
            console.print(bold_text(f'Preprocessor: "{preprocessor_name}"'))
            console.print(table)

            # Output captured summary for non-interactive logging.
            if print_fn:
                print_fn(console.end_capture(), line_break=False)

        # Hardcode summary from keras_core for now.
        keras_core.Model.summary(
            self,
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            **kwargs,
        )
