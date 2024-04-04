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

import json
import os

from rich import console as rich_console
from rich import markup
from rich import table as rich_table

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import config
from keras_nlp.backend import keras
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.utils.keras_utils import print_msg
from keras_nlp.utils.pipeline_model import PipelineModel
from keras_nlp.utils.preset_utils import CONFIG_FILE
from keras_nlp.utils.preset_utils import PREPROCESSOR_CONFIG_FILE
from keras_nlp.utils.preset_utils import TASK_CONFIG_FILE
from keras_nlp.utils.preset_utils import TASK_WEIGHTS_FILE
from keras_nlp.utils.preset_utils import check_config_class
from keras_nlp.utils.preset_utils import get_file
from keras_nlp.utils.preset_utils import list_presets
from keras_nlp.utils.preset_utils import list_subclasses
from keras_nlp.utils.preset_utils import load_from_preset
from keras_nlp.utils.preset_utils import recursive_pop
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.Task")
class Task(PipelineModel):
    """Base class for all Task models.

    A `Task` wraps a `keras_nlp.models.Backbone` and
    a `keras_nlp.models.Preprocessor` to create a model that can be directly
    used for training, fine-tuning, and prediction for a given text problem.

    All `Task` models have `backbone` and `preprocessor` properties. By
    default `fit()`, `predict()` and `evaluate()` will preprocess all inputs
    automatically. To preprocess inputs separately or with a custom function,
    you can set `task.preprocessor = None`, which disable any automatic
    preprocessing on inputs.

    All `Task` classes include a `from_preset()` constructor which can be used
    to load a pre-trained config and weights. Calling `from_preset()` on a task
    will automatically instantiate a `keras_nlp.models.Backbone` and
    `keras_nlp.models.Preprocessor`.
    """

    backbone_cls = None
    preprocessor_cls = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )
        self._initialized = True
        if self.backbone is not None:
            # Keras 2 and Keras 3 handle setting policy differently.
            if config.keras_3():
                self.dtype_policy = self._backbone.dtype_policy
            else:
                self._set_dtype_policy(self._backbone.dtype_policy)

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
        """A `keras_nlp.models.Backbone` model with the core architecture."""
        return getattr(self, "_backbone", None)

    @backbone.setter
    def backbone(self, value):
        self._backbone = value

    @property
    def preprocessor(self):
        """A `keras_nlp.models.Preprocessor` layer used to preprocess input."""
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
        presets = list_presets(cls)
        # We can also load backbone presets.
        if cls.backbone_cls is not None:
            presets.update(cls.backbone_cls.presets)
        for subclass in list_subclasses(cls):
            presets.update(subclass.presets)
        return presets

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate a `keras_nlp.models.Task` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as a
        one of:

        1. a built in preset identifier like `'bert_base_en'`
        2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
        3. a Hugging Face handle like `'hf://user/bert_base_en'`
        4. a path to a local preset directory like `'./bert_base_en'`

        For any `Task` subclass, you can run `cls.presets.keys()` to list all
        built-in presets available on the class.

        This constructor can be called in one of two ways. Either from a task
        specific base class like `keras_nlp.models.CausalLM.from_preset()`, or
        from a model class like `keras_nlp.models.BertClassifier.from_preset()`.
        If calling from the a base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        Args:
            preset: string. A built in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        # Load a Gemma generative task.
        causal_lm = keras_nlp.models.CausalLM.from_preset(
            "gemma_2b_en",
        )

        # Load a Bert classification task.
        model = keras_nlp.models.Classifier.from_preset(
            "bert_base_en",
            num_classes=2,
        )
        ```
        """
        if cls == Task:
            raise ValueError(
                "Do not call `Task.from_preset()` directly. Instead call a "
                "particular task class, e.g. "
                "`keras_nlp.models.Classifier.from_preset()` or "
                "`keras_nlp.models.BertClassifier.from_preset()`."
            )
        if "backbone" in kwargs:
            raise ValueError(
                "You cannot pass a `backbone` argument to the `from_preset` "
                f"method. Instead, call the {cls.__name__} default "
                "constructor with a `backbone` argument. "
                f"Received: backbone={kwargs['backbone']}."
            )

        task_config_path = os.path.join(preset, TASK_CONFIG_FILE)
        task_preset_cls = check_config_class(preset, TASK_CONFIG_FILE)
        backbone_preset_cls = check_config_class(preset)
        backbone_config_path = get_file(preset, CONFIG_FILE)
        with open(backbone_config_path) as config_file:
            backbone_config = json.load(config_file)

        # Load preprocessor from preset.
        preprocessor_config_path = os.path.join(
            preset, PREPROCESSOR_CONFIG_FILE
        )
        if os.path.exists(preprocessor_config_path):
            preprocessor_preset_cls = check_config_class(
                preset, PREPROCESSOR_CONFIG_FILE
            )
            if not issubclass(preprocessor_preset_cls, Preprocessor):
                raise ValueError(
                    f"`{PREPROCESSOR_CONFIG_FILE}` in `{preset}` should be a subclass of `Preprocessor`."
                )
            preprocessor = preprocessor_preset_cls.from_preset(preset)
        elif "preprocessor" in kwargs:
            preprocessor = kwargs.pop("preprocessor")
        else:
            tokenizer = load_from_preset(
                preset,
                config_file="tokenizer.json",
            )
            preprocessor = cls.preprocessor_cls(tokenizer=tokenizer)

        # Backbone case.
        if not os.path.exists(task_config_path) or not issubclass(
            task_preset_cls, cls
        ):
            if backbone_preset_cls is not cls.backbone_cls:
                subclasses = list_subclasses(cls)
                subclasses = tuple(
                    filter(
                        lambda x: x.backbone_cls == backbone_preset_cls,
                        subclasses,
                    )
                )
                if len(subclasses) == 0:
                    raise ValueError(
                        f"No registered subclass of `{cls.__name__}` can load "
                        f"a `{backbone_preset_cls.__name__}`."
                    )
                if len(subclasses) > 1:
                    names = ", ".join(f"`{x.__name__}`" for x in subclasses)
                    raise ValueError(
                        f"Ambiguous call to `{cls.__name__}.from_preset()`. "
                        f"Found multiple possible subclasses {names}. "
                        "Please call `from_preset` on a subclass directly."
                    )
                cls = subclasses[0]
            # Forward dtype to the backbone.
            config_overrides = {}
            if "dtype" in kwargs:
                config_overrides["dtype"] = kwargs.pop("dtype")
            backbone = load_from_preset(
                preset,
                load_weights=load_weights,
                config_overrides=config_overrides,
            )
            return cls(backbone=backbone, preprocessor=preprocessor, **kwargs)

        # Load task from preset if it exists.
        # TODO: I should probably move task loading logic to preset_utils.py?
        if not issubclass(cls, Task):
            raise ValueError(
                "`{cls.__name__}` should be subclass of Task!"
            )  # TODO: update error message
        task_config_class = check_config_class(
            preset, config_file=TASK_CONFIG_FILE
        )
        if not issubclass(task_config_class, cls):
            raise ValueError(
                f"`{TASK_CONFIG_FILE}` has type `{task_config_class.__name__}` "
                f"which is not a subclass of calling class `{cls.__name__}`. Call "
                f"`from_preset` directly on `{task_config_class.__name__}` instead."
            )

        with open(task_config_path, "r") as config_file:
            task_config = json.load(config_file)
        # TODO: add back backbone and preprocessor config when save_to_preset removes them (rn, save_to_preset, doesn't remove them!).
        # task_config.update(backbone_config)
        # task_config.update(preprocessor_config)
        task = keras.saving.deserialize_keras_object(task_config)
        if load_weights:
            if not task_config["weights"]:
                raise ValueError(
                    f"`weights` config is missing from `{TASK_CONFIG_FILE}` in "
                    f"preset directory `{preset}`."
                )
            if not backbone_config["weights"]:
                raise ValueError(
                    f"`weights` config is missing from `{CONFIG_FILE}` in "
                    f"preset directory `{preset}`."
                )
            task_weights_path = os.path.join(preset, task_config["weights"])
            task.load_task_weights(task_weights_path)
            backbone_weights_path = os.path.join(
                preset, backbone_config["weights"]
            )
            task.backbone.load_weights(backbone_weights_path)
            # TODO: is this assignment okay?
            task.preprocessor = preprocessor
            return task

    def load_task_weights(self, filepath, skip_mismatch=False):
        """Load only the tasks specific weights not in the backbone."""
        if not str(filepath).endswith(".weights.h5"):
            raise ValueError(
                "The filename must end in `.weights.h5`. "
                f"Received: filepath={filepath}"
            )
        weights_store = keras.src.saving.saving_lib.H5IOStore(
            filepath, mode="r"
        )
        backbone_layer_ids = set(id(w) for w in self.backbone._flatten_layers())
        # TODO: It's better not to use this private API here. Francois recommends chaning our public saving API and skip objects to it. Francoins will do this.
        keras.src.saving.saving_lib._load_state(
            self,
            weights_store=weights_store,
            assets_store=None,
            inner_path="",
            skip_mismatch=skip_mismatch,
            visited_trackables=backbone_layer_ids,
            failed_trackables=set(),
        )
        weights_store.close()

    def save_task_weights(self, filepath):
        """Save only the tasks specific weights not in the backbone."""
        if not str(filepath).endswith(".weights.h5"):
            raise ValueError(
                "The filename must end in `.weights.h5`. "
                f"Received: filepath={filepath}"
            )
        task_weight_ids = set(id(w) for w in self.weights)
        backbone_weight_ids = set(id(w) for w in self.backbone.weights)
        backbone_layer_ids = set(id(w) for w in self.backbone._flatten_layers())
        if task_weight_ids.issubset(backbone_weight_ids):
            raise ValueError(
                f"Task {self} has no weights not in the `backbone`. "
                "`save_task_weights()` has nothing to save."
            )
        weights_store = keras.src.saving.saving_lib.H5IOStore(
            filepath, mode="w"
        )
        keras.src.saving.saving_lib._save_state(
            self,
            weights_store=weights_store,
            assets_store=None,
            inner_path="",
            visited_trackables=backbone_layer_ids,
        )
        weights_store.close()

    # TODO: do we want to have a `save_weights` flag in this public save_to_preset? probably yes!
    def save_to_preset(self, preset):
        """TODO: add docstring"""
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor is not defined!"
            )  # TODO: improve error message

        self.preprocessor.save_to_preset(preset)
        self.backbone.save_to_preset(preset)
        weights_filename = TASK_WEIGHTS_FILE

        # TODO: the serialization and saving logic should probably be moved to preset_utils.py
        task_config_path = os.path.join(preset, TASK_CONFIG_FILE)
        task_config = keras.saving.serialize_keras_object(self)
        recursive_pop(task_config, "compile_config")
        recursive_pop(task_config, "build_config")
        # TODO: remove preprocessor and backbone from task.json to prevent redundancy in config files.
        # recursive_pop(task_config, "preprocessor")
        # recursive_pop(task_config, "backbone")
        task_config["weights"] = weights_filename
        with open(task_config_path, "w") as config_file:
            config_file.write(json.dumps(task_config, indent=4))
        task_weights_path = os.path.join(preset, weights_filename)
        self.save_task_weights(task_weights_path)

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
