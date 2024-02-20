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

from rich import console as rich_console
from rich import markup
from rich import table as rich_table

from keras_nlp.backend import config
from keras_nlp.backend import keras
from keras_nlp.utils.keras_utils import print_msg
from keras_nlp.utils.pipeline_model import PipelineModel
from keras_nlp.utils.preset_utils import check_preset_class
from keras_nlp.utils.preset_utils import load_from_preset
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.saving.register_keras_serializable(package="keras_nlp")
class Task(PipelineModel):
    """Base class for Task models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )
        self._initialized = True

    def __dir__(self):
        if config.keras_3():
            return super().__dir__()

        # Temporary fixes for Keras 2 saving. This mimics the following PR for
        # older version of Keras: https://github.com/keras-team/keras/pull/18982
        def filter_fn(attr):
            if attr in [
                "_layer_checkpoint_dependencies",
                "transformer_layers",
                "encoder_transformer_layers",
                "decoder_transformer_layers",
            ]:
                return False
            return id(getattr(self, attr)) not in self._functional_layer_ids

        return filter(filter_fn, super().__dir__())

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
        # Temporarily disable jit compilation on torch.
        if config.backend() == "torch":
            kwargs["jit_compile"] = False
        self._check_for_loss_mismatch(loss)
        super().compile(optimizer=optimizer, loss=loss, **kwargs)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        if self.preprocessor is not None:
            return self.preprocessor(x, y=y, sample_weight=sample_weight)
        else:
            return super().preprocess_samples(x, y, sample_weight)

    def __setattr__(self, name, value):
        # Work around setattr issues for Keras 2 and Keras 3 torch backend.
        # Since all our state is covered by functional model we can route
        # around custom setattr calls.
        is_property = isinstance(getattr(type(self), name, None), property)
        is_unitialized = not hasattr(self, "_initialized")
        is_torch = config.backend() == "torch"
        is_keras_2 = not config.keras_3()
        if is_torch and (is_property or is_unitialized):
            return object.__setattr__(self, name, value)
        if is_keras_2 and is_unitialized:
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def backbone(self):
        """A `keras.Model` instance providing the backbone sub-model."""
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
        if "backbone" in kwargs:
            raise ValueError(
                "You cannot pass a `backbone` argument to the `from_preset` "
                f"method. Instead, call the {cls.__name__} default "
                "constructor with a `backbone` argument. "
                f"Received: backbone={kwargs['backbone']}."
            )
        # We support short IDs for official presets, e.g. `"bert_base_en"`.
        # Map these to a Kaggle Models handle.
        if preset in cls.presets:
            preset = cls.presets[preset]["kaggle_handle"]

        preset_cls = check_preset_class(preset, (cls, cls.backbone_cls))

        # Backbone case.
        if preset_cls == cls.backbone_cls:
            # Forward dtype to the backbone.
            config_overrides = {}
            if "dtype" in kwargs:
                config_overrides["dtype"] = kwargs.pop("dtype")
            backbone = load_from_preset(
                preset,
                load_weights=load_weights,
                config_overrides=config_overrides,
            )
            if "preprocessor" in kwargs:
                preprocessor = kwargs.pop("preprocessor")
            else:
                tokenizer = load_from_preset(
                    preset,
                    config_file="tokenizer.json",
                )
                preprocessor = cls.preprocessor_cls(tokenizer=tokenizer)
            return cls(backbone=backbone, preprocessor=preprocessor, **kwargs)

        # Task case.
        return load_from_preset(
            preset,
            load_weights=load_weights,
            config_overrides=kwargs,
        )

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

        # Avoid `tf.keras.Model.summary()`, so the above output matches.
        if config.keras_3():
            super().summary(
                line_length=line_length,
                positions=positions,
                print_fn=print_fn,
                **kwargs,
            )
        else:
            import keras_core

            keras_core.Model.summary(
                self,
                line_length=line_length,
                positions=positions,
                print_fn=print_fn,
                **kwargs,
            )
