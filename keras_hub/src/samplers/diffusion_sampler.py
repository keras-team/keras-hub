from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.samplers.sampler import Sampler


@keras_hub_export("keras_hub.samplers.DiffusionSampler")
class DiffusionSampler(Sampler):
    """Base class for block-diffusion samplers.

    Unlike `Sampler`, which decodes one autoregressive token at a time, a
    `DiffusionSampler` denoises an entire fixed-length canvas over several
    steps. Subclasses override `__call__` directly with their own denoising
    algorithm — the `get_next_token` extension point `Sampler` subclasses
    use does not apply here.

    A `DiffusionSampler` is only compatible with `BlockDiffusionLM` models,
    not standard autoregressive `CausalLM` models. `BlockDiffusionLM`
    checks `isinstance(sampler, DiffusionSampler)` and raises a clear error
    if a standard `Sampler` is passed instead.

    Call arguments:
        next: Callable accepting `(canvas, prev_logits, step)` and returning
            logits for the current denoising step.
        canvas: int tensor of shape `(B, canvas_length)` containing the
            initial token assignment.
        max_steps: int. Maximum number of denoising steps.
        model: Optional Keras model, used by JAX stateless scopes.
    """
