from keras.api import backend


# redundant once https://github.com/keras-team/keras/issues/21137 is addressed
def keras_print(*args, **kwargs):
    back_end = backend.backend()
    if back_end == "tensorflow":
        import tensorflow as tf

        return tf.print(*args, **kwargs)
    elif back_end == "jax":
        import jax.debug

        return jax.debug.print(*args, **kwargs)
    else:
        return print(*args, **kwargs)
    # print_fn = {"jax": jax.debug.print,
    #             "tensorflow": keras_print}.get(backend, print)
    # "torch"
    #   pytorch.org/docs/stable/generated/torch.set_printoptions.html ?
    # "openvino"
    # "numpy"
    # return print_fn(*args, **kwargs)


__all__ = ["keras_print"]
