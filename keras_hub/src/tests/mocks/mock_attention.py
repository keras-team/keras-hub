from contextlib import contextmanager
from unittest.mock import patch

import keras

from keras_hub.src.utils.keras_utils import running_on_gpu
from keras_hub.src.utils.keras_utils import running_on_tpu


@contextmanager
def patch_dot_product_attention():
    """Patch `keras.ops.dot_product_attention` so a test can observe it.

    The mock keeps the real signature and calls through to the real op. Both
    matter:

    - `autospec=True` preserves the signature. `_use_fused_attention_op()`
      decides on TPU whether the fused op is usable by inspecting that
      signature for `attn_logits_soft_cap`. A bare `MagicMock` reports
      `(*args, **kwargs)`, which silently disables the very path the test is
      trying to observe.
    - `side_effect` calls through, so `generate()` receives a real tensor
      rather than a `MagicMock`.

    Yields:
        The mock standing in for `keras.ops.dot_product_attention`.
    """
    # Bind the real op before entering the `with`. Reading it inside the body
    # instead would pick up the mock and make it its own side effect.
    real_dot_product_attention = keras.ops.dot_product_attention
    with patch(
        "keras.ops.dot_product_attention",
        autospec=True,
        side_effect=real_dot_product_attention,
    ) as mock_func:
        yield mock_func


def assert_fused_attention_used(mock_func):
    """Assert the fused op was used on exactly the devices that support it.

    Only valid for a caller that has already established the rest of
    `_use_fused_attention_op()`'s gate: the JAX backend, with
    `fused_attention_op_available()` and `gpu_supports_fused_attention_op()`
    both true, a layer built with `dropout=0`, and no logit soft cap. Given
    that, the gate reduces to the device check made here.

    The TPU arm only opens up when the patched op reports a real signature,
    which is why the mock has to come from `patch_dot_product_attention`.

    Args:
        mock_func: The mock yielded by `patch_dot_product_attention`.
    """
    if running_on_gpu() or running_on_tpu():
        mock_func.assert_called()
    else:
        mock_func.assert_not_called()
