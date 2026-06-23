"""KerasHub source package.

This module is imported whenever any ``keras_hub.src.*`` submodule is loaded.
"""


def _patch_keras_ops_for_jax_x64():
    """Work around JAX x64 mixed-index dtype errors for slice ops.

    KerasHub assumes the standard Keras/JAX 32-bit default dtype regime. Some
    downstream dependencies may enable ``JAX_ENABLE_X64``, which causes Python
    scalars and unqualified ``jnp`` creation routines to produce ``int64`` and
    ``float64`` values. That in turn breaks KerasHub's control-flow ops
    (``cond``/``while_loop`` branches with mismatched dtypes) and many layers
    that expect ``int32``/``float32`` tensors.

    We therefore disable x64 at import time, and additionally normalize
    ``keras.ops.slice`` / ``keras.ops.slice_update`` start indices to a
    homogeneous ``int32`` array so that KerasHub's many slice call sites
    continue to work even if x64 is enabled externally.
    """
    try:
        import jax
    except ImportError:
        return

    # Ensure the standard 32-bit default dtype regime. This must happen before
    # any JAX arrays are created.
    jax.config.update("jax_enable_x64", False)

    import jax.numpy as jnp
    import keras

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
