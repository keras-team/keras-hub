## Code Review Guardrails (High-Signal & Anti-False-Positive Rules)

When performing code reviews on pull requests in `keras-team/keras-hub`, strictly adhere to these top-level rules before applying any guideline below:

1. **Check End-to-End Consistency Without Critiquing Untouched Code**:
   - **DO check**: When a PR adds or modifies a layer, backbone, converter, preprocessor, task, or checkpoint converter, verify that `__init__` attributes, `get_config()`, `build()`, `call()`, and sibling hooks stay synchronized, and flag genuine bugs or spelling typos in newly added identifiers/docstrings/tests.
   - **Do NOT flag**: Untouched context lines, moved/renamed code, or pre-existing entries/logic adjacent to a `+`/`-` line (such as unrelated lines in `conftest.py`). Never widen `start_line..end_line` across a diff hunk just to critique pre-existing code when the PR only made a narrow fix, rename, or migration nearby.
2. **Verify Imports, Base Classes & Package Versions Before Claiming Errors**:
   - Unchanged `*_test.py` files and `keras_hub/src/tests/test_case.py` are omitted from auxiliary `<FILES>` context (`keras_hub.src.tests.test_case.TestCase` inherits from `tf.test.TestCase` [which provides `get_temp_dir`] and `parameterized.TestCase`, and defines `assertAllClose`, `assertEqual`, `assertAllEqual`, `assertDTypeEqual`, `run_layer_test`, `run_preprocessing_layer_test`, `run_grain_preprocessing_test`, `run_preprocessor_test`, `run_serialization_test`, `run_precision_test`, `run_quantization_test`, `run_model_saving_test`, `run_litert_export_test`, `run_backbone_test`, `run_vision_backbone_test`, `run_task_test`, `run_preset_test`, `get_test_data_dir`, and `load_test_image`).
   - **Do NOT flag** unverified `NameError` or missing-method/attribute claims unless verified against full file imports, core Keras 3 exports (e.g., `keras.layers.ReversibleEmbedding` is in core Keras 3), and base classes, and **Do NOT claim** a third-party package version, pip extra, or GitHub Action tag does not exist based on your knowledge cutoff.
3. **Catch Real Bugs, Not Speculative Defensive Boilerplate**:
   - Check for uninitialized/`None` attributes dereferenced across control-flow paths, `assert` used instead of `ValueError` for user-input validation in library code, dtype mismatches in mixed precision (`float16`/`bfloat16`), and unguarded TensorFlow usage in shared paths.
   - **Do NOT flag** hypothetical `None`/empty inputs outside an internal helper's caller invariants, replacing intentional `config["key"]` or `config.pop("key")` lookups with silent `.get("key", default)` fallbacks, `getattr(backbone, ...)` fallbacks for attributes always set on that `Backbone`, `RuntimeWarning` wrappers in tests (`pyproject.toml` ignores `RuntimeWarning`), English i18n string nags, caching `keras.ops` tensor outputs with `functools.lru_cache`, or cold-path micro-optimizations/caching outside per-step hot paths (such as one-time setup, config parsing, `tools/checkpoint_conversion/**`, or `keras_hub/src/utils/transformers/**`).
4. **Skip Checks Enforced by CI**:
   - Pre-commit and CI run `ruff check` (`E`, `F`, `I`), `ruff format --check`, `shell/api_gen.sh` (`keras_hub/api/**`), and `zizmor`. **Do NOT flag** formatting, line length, import sorting, unused imports (`F401`), or tell authors to run `pre-commit run --all-files`.
5. **Respect Existing Repo Patterns**:
   - Before suggesting review comments, actively check existing design patterns across `keras-hub`. Do not suggest boilerplate or structures that contradict established repository conventions:
     - **Sub-layer & attribute tracking**: `Backbone`, `Task` (`CausalLM`, `TextClassifier`, etc.), and `Preprocessor` classes assign `self.backbone = backbone`, `self.preprocessor = preprocessor`, `self.tokenizer = tokenizer`, and sub-layers via normal attribute assignment so Keras tracks their weights/state (**Do NOT** ask to wrap normal sub-layer, `backbone`, `preprocessor`, or converter assignments in `object.__setattr__`; `object.__setattr__` is ONLY used when storing an auxiliary `Layer`/`Model` reference that must intentionally remain *untracked* by Keras, such as a cached external speculative-decoding assistant model).
     - **Internal layer serialization**: Unexported internal model sub-layers (not decorated with `@keras_hub_export`) serialize via Keras's module-path mechanism (**Do NOT** ask to add `@keras.saving.register_keras_serializable` to unexported internal helper layers).
     - **Keras 3 constructor & tokenizer idioms**: `keras.Sequential` does NOT accept a `dtype` keyword argument (`TypeError`), standard Keras `Layer` constructors already accept `dtype` strings/policies/serialized dicts without manual `keras.dtype_policies.get(dtype)`, and `Tokenizer._add_special_token(...)` is intentionally designed to run **before** `super().__init__(...)`.

---

## Key Principles

- **Modularity**: Models are broken down into distinct, reusable components: `Backbone`, `Converter` (`Tokenizer`, `ImageConverter`, `AudioConverter`, etc.), `Preprocessor`, and `Task`.
- **Consistency**: Strict adherence to naming and file structure conventions across models is crucial for predictability and maintainability.
- **Validation**: When porting a **new model**, components must be validated against the original model's implementation, with numerical equivalence demonstrated via Colab notebooks or verification output in the PR description (**Do NOT flag** missing Colab links on bug fixes, refactors, CI/infra changes, or updates to existing models).
- **Reusability**: Prioritize using existing layers from `keras.layers` and `keras_hub.layers` before implementing custom logic. Custom layers should only be added for architectural differences not covered by existing components.
- **Backend Agnostic**: All modeling code must be Keras 3 backend-agnostic, supporting TensorFlow, JAX, and PyTorch backends.
- **Model license**: New model ports must use a permissive/approved license (Apache 2.0, MIT, Llama Community License, or Gemma, per `CONTRIBUTING.md`). If a newly contributed model is under a copyleft license (e.g., GPL, AGPL, CC-BY-NC), flag this immediately, and the pull request should be marked to be closed.

---

## Directory Structure and Naming Conventions

### New Model Directory Structure
New model contributions live under `keras_hub/src/models/<model_name>/` with standardized component files (`__init__.py`, `<model_name>_backbone.py`, `<model_name>_presets.py`, modality converters `<model_name>_{tokenizer,image_converter,video_converter,audio_converter}.py`, `<model_name>_*preprocessor.py`, task classes `<model_name>_<task>.py`, and corresponding `*_test.py` files). Checkpoint conversion scripts live in `tools/checkpoint_conversion/convert_<model_name>_checkpoints.py`; HuggingFace converters live in `keras_hub/src/utils/transformers/convert_<model_name>.py` (and `convert_<model_name>_test.py`); Timm converters live in `keras_hub/src/utils/timm/`. (**Do NOT** demand every component file in a single PR when a contributor ports Backbone, Tokenizer, Presets, or Tasks across staged PRs, or when a modality is not applicable.)

### Naming Rules
- **Files**: Lowercase `snake_case` matching `<model_name>_<component_type>.py` (e.g., `distil_bert_backbone.py`, `xlm_roberta_tokenizer.py`; keep acronyms as a single segment).
- **Classes**: `PascalCase` matching `<ModelName><ComponentType>` (e.g., `DistilBertBackbone`, `GPT2CausalLM`, `XLMRobertaBackbone`; capitalize acronyms like `GPT2`/`LSTM`, treat pronounceable acronyms like `Bert`/`Deberta` as words).
- **Functions, Methods, and Presets**: Lowercase `snake_case` (e.g., `from_preset()`, `bert_base_en_uncased`; do not abbreviate preset sizes like `extra_small`/`extra_large`).
- **Standardized Model Inputs & Args**: Use `token_ids` and `padding_mask` for text models, `pixel_values` for image models, `audio_features` for audio models, and keep hyperparameter names consistent with sibling models in the repository (`vocabulary_size`, `num_layers`, `num_heads`, `hidden_dim`, `intermediate_dim`, `dropout`).

---

## Component & Layer Implementation Guidelines

### Backbones, Converters, Preprocessors, Tasks, and Presets
- **Backbone (`<model_name>_backbone.py`)**: Subclass `Backbone` (`keras_hub.src.models.backbone.Backbone`), export with `@keras_hub_export("keras_hub.models.<ModelName>Backbone")`, instantiate sub-layers in `__init__`, build the Keras Functional API graph (`inputs={...}`, `outputs=...`) inside `__init__`, and pass `dtype=dtype, **kwargs` to `super().__init__`.
- **Data Converters (`<model_name>_tokenizer.py`, etc.)**: Subclass the appropriate KerasHub base converter (`WordPieceTokenizer`, `BytePairTokenizer`, `SentencePieceTokenizer`, `ImageConverter`, `AudioConverter`), export under both `keras_hub.tokenizers.*` / `keras_hub.layers.*` and `keras_hub.models.*` where appropriate, and set `backbone_cls`.
- **Preprocessors (`<model_name>_*preprocessor.py`)**: Subclass the appropriate base `Preprocessor` (e.g., `TextClassifierPreprocessor`, `CausalLMPreprocessor`), set `backbone_cls` and `tokenizer_cls` (or modality converter classes), and format outputs as a dictionary matching the backbone's input signature (**Do NOT** ask to re-serialize attributes in `get_config()` that `super().get_config()` already serializes, and **Do NOT** insert `keras.ops.*` calls such as `ops.cast` on `bool` `padding_mask` in `tf.data`-mapped preprocessor paths where non-TF backend ops raise `NotImplementedError`).
- **Task Models (`<model_name>_<task>.py`)**: Subclass `Task` (`CausalLM`, `TextClassifier`, `MaskedLM`, etc.), set `backbone_cls` and `preprocessor_cls`, wire `backbone` and task head in `__init__` via the Functional API (`super().__init__(inputs=inputs, outputs=outputs, **kwargs)`), and support `fit()`, `predict()`, and `generate()` where applicable (**Do NOT** require `__init__` or `get_config` on abstract base task classes like `TextClassifier` or `TokenClassifier` whose concrete subclasses build the head).
- **Presets (`<model_name>_presets.py`)**: Define `backbone_presets` with `snake_case` preset names, `metadata` (`description`, `params`, `path`), and `kaggle_handle`.

### Ideal Layer & Serialization Style
1. **Accept `**kwargs`** in `__init__` and forward to `super().__init__(**kwargs)`.
2. **Store `__init__` arguments** as Python attributes on `self` with matching names and passed values (clone shared initializer instances across distinct sub-layers rather than reusing a single stateful initializer object). Instantiate shape-independent sub-layers (such as `Dropout` or `Softmax`) in `__init__` rather than `build()`.
3. **Implement `get_config()`** chaining `config = super().get_config()` and `config.update({...})` for every `__init__` argument introduced by the class so `from_config` / `run_serialization_test` round-trips idempotently.
4. **Avoid `assert` for user-input validation**: Raise `ValueError` (or `ImportError` for missing optional packages) with a clear message including the received value.

---

## Multi-Backend, Shapes, Dtypes & TensorFlow Decoupling

### Import Conventions & TensorFlow Decoupling
- **Internal vs. public imports**: Inside `keras_hub/src/**`, import `keras` as a top-level module (`import keras`, `from keras import ops`, `from keras import tree`) and import internal KerasHub symbols from `keras_hub.src.*` (never `import keras_hub` inside `keras_hub/src/**`, except in `integration_tests/**` and docstring examples, and avoid importing private symbols from `keras.src.backend`).
- **No unguarded TensorFlow imports in `keras_hub/src/**`**: Never add a top-level `import tensorflow as tf` or unguarded `import tensorflow_text` in `keras_hub/src/**` (outside TF-specific tests). Import `tf` via `from keras_hub.src.utils.tensor_utils import tf` (which returns `None` when TensorFlow is uninstalled) so layers, models, tokenizers, and preprocessors can be imported, constructed, and run on the pure-Python / Grain path without TensorFlow (**Do NOT flag** `import tensorflow` in standalone `tools/**` scripts, TF-specific tests, local imports guarded behind `keras.config.backend() == "tensorflow"` / `in_tf_function()`, or files that already guard `tf` via `tensor_utils`).
- **Mixed-tensor NumPy conversion in preprocessing/metrics**: In preprocessors, tokenizers, and metrics that may receive `tf.Tensor`, `tf.RaggedTensor`, backend tensors, NumPy arrays, or Python lists, prefer `tensor_utils.convert_to_numpy(x)` over `keras.ops.convert_to_numpy(x)`. Note that `CausalLM` `generate_preprocess()` and `generate_postprocess()` run eagerly on concrete host inputs during `generate()`, never inside `tf.data` or a traced `tf.function`/`jax.jit` (**Do NOT** claim `np.asarray` or Python control flow inside `generate_preprocess`-only helpers fails under `tf.function`/`tf.data`).

### Multi-Backend Tensor, Shape & Dtype Rules
- **No NumPy/eager conversions in layer `call()`**: Support XLA/JIT compilation on trainable modeling layers where applicable. Inside `Layer.call()` / `Backbone` forward passes, use `keras.ops`—never call `ops.convert_to_numpy`, `.numpy()`, NumPy, or SciPy on symbolic/backend tensors, and use `ops.less`, `ops.greater`, `ops.equal`, etc. instead of Python `<`, `>`, `==` on tensors (plain Python operators `+`, `-`, `*`, `//` are right for shape dimensions).
- **Dynamic shapes & `jax.jit` / `tf.function` accuracy**:
  - In `call()`, use `ops.shape(x)` instead of `x.shape` when indexing a batch/sequence/time dimension that can be `None` symbolically (`x.shape` and `len(x.shape)` are fine in `build(self, input_shape)`, `compute_output_shape`, or for static rank/channel dims).
  - Under `jax.jit`, tensor shapes (`ops.shape(x)[i]`) are concrete static Python `int`s at trace time (only tensor values are tracers), whereas under `tf.function` unknown dimensions in `ops.shape(x)[i]` are dynamic `tf.Tensor`s (**Do NOT** branch with Python `if` on a potentially unknown `ops.shape(x)[i]` in `tf.function`, and **Do NOT** claim `ops.shape(x)[i]` is a dynamic tracer that breaks `ops.slice` or `ops.reshape` under `jax.jit`).
  - `keras.ops.slice(inputs, start_indices, shape)` takes the **slice `shape` (sizes)** as its third argument (same convention as `tf.slice`), **NOT** exclusive `limit_indices`—**never** claim `ops.slice` expects `limit_indices`.
- **Dtypes & numerical stability**:
  - Python `int` and `float` literals (`0`, `1`, `0.0`, `1.0`) are **weakly typed** in Keras 3 ops (`ops.where(cond, x, 0.0)` preserves `float16`/`bfloat16`)—**Do NOT** claim Python float scalars upcast `float16`/`bfloat16` to `float32`.
  - When dividing by a mask sum in `float16` (such as masked mean pooling), clamp the divisor with `ops.maximum(sum_mask, 1.0)` or `ops.divide_no_nan` rather than `1e-9` (which underflows to `0.0` in `float16`). Note `ops.softmax` already subtracts `max` internally for numerical stability.

---

## Docstrings, Converters & Testing

### Docstrings, Comments and Type Hints
- **Google-style docstrings**: Document public `@keras_hub_export` classes, functions, and methods with a concise one-line summary (on the opening `"""` line), description, paper `Reference:` (if applicable), `Args:` (with types formatted as `arg_name: type. Description.`), `Returns:`/`Raises:` where relevant, comprehensive `Example:`/`Examples:` using full `keras_hub.*` paths, and edge cases/limitations where applicable. Keep documentation up-to-date with code changes. Document constructor arguments on the class docstring rather than `__init__`; `build()` and `call()` generally do not need separate docstrings (**Do NOT flag** missing `Examples:` on internal helpers or unexported sub-layers).
- **Comments describe the code, not the PR**: Flag comments that narrate PR history or "previously" instead of explaining non-obvious invariants (**Do NOT flag** concise "why" comments or intentional removal of obsolete comments).
- **No signature type annotations**: KerasHub does **not** use Python type hints in function or `__init__` signatures (default values like `None` or `"channels_last"` are fine); put types in docstring `Args:`.

### Checkpoint & HuggingFace / Timm Converters
- **HuggingFace / Timm converters (`keras_hub/src/utils/{transformers,timm}/`)**: Define `backbone_cls`, `convert_backbone_config(transformers_config)`, and `convert_weights(backbone, loader, transformers_config)`. Index required config keys directly (`transformers_config["vocab_size"]`) so missing keys fail loudly rather than silently defaulting with `.get(...)`, while handling keys that are legitimately `None`/absent across upstream checkpoints. Open JSON files with UTF-8 encoding (e.g., `load_json`). Test converters with `load_weights=False`, `Backbone.from_preset` / task class detection, and numerical equivalence tests when possible.
- **Checkpoint conversion scripts (`tools/checkpoint_conversion/`)**: Scripts must be reusable and well-documented, handle all presets for the model, include proper error handling and validation, and verify numerical parity against the reference implementation using elementwise `np.testing.assert_allclose` / **maximum absolute difference** (`atol` around `1e-4`–`1e-3`, using `torch.no_grad()` for PyTorch forward passes)—**Do NOT** ask authors to replace maximum absolute difference or elementwise `assert_allclose` checks with mean logits difference tolerances (mean over `(batch, seq, vocab)` hides localized bugs).

### Unit Tests
- **Corresponding `_test.py` & fast inputs**: Logic modules in `keras_hub/src/` must have unit tests in `<module>_test.py` using tiny fast configs/inputs (e.g., `(16, 16, 3)` images, `hidden_dim=16`, `num_layers=2`) (**Do NOT** demand `_test.py` files for `__init__.py`, `*_presets.py`, `version.py`, `tools/checkpoint_conversion/**`, docs, `.gemini/`, or `.github/`).
- **Use `TestCase` helpers & meaningful assertions**: Prefer `self.run_backbone_test`, `self.run_vision_backbone_test`, `self.run_layer_test`, `self.run_preprocessing_layer_test`, `self.run_preprocessor_test`, `self.run_task_test`, `self.run_preset_test` (`@pytest.mark.large` / `@pytest.mark.extra_large`), `self.run_model_saving_test`, and `self.run_litert_export_test`. Tests must check values/outputs and be able to fail if the feature/fix under test is broken. Write temporary files only under `self.get_temp_dir()`. (**Do NOT** flag explicit `atol=0.0, rtol=0.0` exact-match assertions when testing deterministic/lossless operations.)
