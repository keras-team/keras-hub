import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.EinsumDense")
class EinsumDense(keras.layers.EinsumDense):
    """EinsumDense subclass that keeps LoRA weights in float32.

    When using mixed precision (bfloat16/float16), the parent class creates
    LoRA weights in the layer's dtype. Adam's second-moment estimate can
    underflow in low precision, producing near-zero updates. This subclass
    re-creates the LoRA variables in float32 after the parent initialises
    them, and casts them back to compute_dtype inside ``call()``.
    """

    def enable_lora(
        self, rank, a_initializer="he_uniform", b_initializer="zeros"
    ):
        super().enable_lora(rank, a_initializer, b_initializer)

        # Determine the dtype string regardless of whether .dtype is a
        # string or an object with a .name attribute.
        lora_a_dtype = self.lora_kernel_a.dtype
        if hasattr(lora_a_dtype, "name"):
            lora_a_dtype = lora_a_dtype.name

        if lora_a_dtype in ("float16", "bfloat16"):
            self._tracker.unlock()

            if self.lora_kernel_a in self._variables:
                self._variables.remove(self.lora_kernel_a)
            if self.lora_kernel_b in self._variables:
                self._variables.remove(self.lora_kernel_b)
            if self.lora_kernel_a in self._trainable_variables:
                self._trainable_variables.remove(self.lora_kernel_a)
            if self.lora_kernel_b in self._trainable_variables:
                self._trainable_variables.remove(self.lora_kernel_b)

            original_a_regularizer = getattr(self.lora_kernel_a, "regularizer", None)
            original_b_regularizer = getattr(self.lora_kernel_b, "regularizer", None)

            self.lora_kernel_a = self.add_weight(
                name="lora_kernel_a",
                shape=self.lora_kernel_a.shape,
                initializer=a_initializer,
                regularizer=original_a_regularizer,
                trainable=True,
                dtype="float32",
            )
            self.lora_kernel_b = self.add_weight(
                name="lora_kernel_b",
                shape=self.lora_kernel_b.shape,
                initializer=b_initializer,
                regularizer=original_b_regularizer,
                trainable=True,
                dtype="float32",
            )
            self._tracker.lock()

    @property
    def lora_kernel_a(self):
        val = getattr(self, "_lora_kernel_a", None)
        if val is not None and getattr(self, "_in_call", False):
            return ops.cast(val, self.compute_dtype)
        return val

    @lora_kernel_a.setter
    def lora_kernel_a(self, value):
        self._lora_kernel_a = value

    @property
    def lora_kernel_b(self):
        val = getattr(self, "_lora_kernel_b", None)
        if val is not None and getattr(self, "_in_call", False):
            return ops.cast(val, self.compute_dtype)
        return val

    @lora_kernel_b.setter
    def lora_kernel_b(self, value):
        self._lora_kernel_b = value

    def call(self, inputs):
        if getattr(self, "lora_rank", None):
            self._in_call = True
            try:
                return super().call(inputs)
            finally:
                self._in_call = False
        return super().call(inputs)
