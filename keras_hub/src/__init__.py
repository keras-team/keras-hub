"""KerasHub source package.

This module is imported whenever any ``keras_hub.src.*`` submodule is loaded.
"""


def _patch_keras_ops_for_jax_x64():
    """Work around JAX x64 mixed-index dtype errors for slice ops.

    Under ``JAX_ENABLE_X64=1`` (or any JAX configuration where Python integers
    are canonicalized to ``int64``), passing a list like ``[0, index]`` to
    ``keras.ops.slice`` / ``keras.ops.slice_update`` can produce
    ``TypeError: index arguments to dynamic_slice must be integers of the same
    type, got: int64, int32`` when ``index`` is an ``int32`` tracer.

    We normalize the start indices to a homogeneous ``int32`` array so that
    KerasHub's many ``ops.slice`` / ``ops.slice_update`` call sites continue
    to work regardless of the active JAX integer dtype configuration.

    This patch is only installed when the active backend is JAX, so torch/TF
    imports of ``keras_hub`` do not mutate ``keras.ops``.
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        return

    import keras

    # Only install the wrapper when the active backend is JAX. Torch/TF
    # imports of keras_hub should leave keras.ops untouched.
    if keras.config.backend() != "jax":
        return

    _orig_slice = keras.ops.slice
    _orig_slice_update = keras.ops.slice_update

    def _normalize_start_indices(start_indices):
        # Only JAX is strict about mixed Python-int / tracer dtypes.
        if keras.config.backend() != "jax":
            return start_indices
        try:
            # ``jnp.asarray`` accepts Python ints and tracers and returns a
            # single homogeneous array, satisfying lax.dynamic_slice/update.
            return jnp.asarray(start_indices, dtype=jnp.int32)
        except Exception:
            return start_indices

    def slice(inputs, start_indices, shape):
        return _orig_slice(
            inputs, _normalize_start_indices(start_indices), shape
        )

    def slice_update(inputs, start_indices, updates):
        return _orig_slice_update(
            inputs, _normalize_start_indices(start_indices), updates
        )

    # Patch both the module attribute and the public namespace so that
    # ``from keras import ops`` and ``keras.ops.slice`` see the same wrapper.
    keras.ops.slice = slice
    keras.ops.slice_update = slice_update


_patch_keras_ops_for_jax_x64()
