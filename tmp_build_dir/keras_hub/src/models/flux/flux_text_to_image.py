from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.flux.flux_model import FluxBackbone
from keras_hub.src.models.flux.flux_text_to_image_preprocessor import (
    FluxTextToImagePreprocessor,
)
from keras_hub.src.models.text_to_image import TextToImage


@keras_hub_export("keras_hub.models.FluxTextToImage")
class FluxTextToImage(TextToImage):
    """An end-to-end Flux model for text-to-image generation.

    This model has a `generate()` method, which generates image based on a
    prompt.

    Args:
        backbone: A `keras_hub.models.FluxBackbone` instance.
        preprocessor: A
            `keras_hub.models.FluxTextToImagePreprocessor` instance.

    Examples:

    Use `generate()` to do image generation.
    ```python
    prompt = (
        "Astronaut in a jungle, cold color palette, muted colors, "
        "detailed, 8k"
    )
    text_to_image = keras_hub.models.FluxTextToImage.from_preset(
        "TBA", height=512, width=512
    )
    text_to_image.generate(
        prompt
    )

    # Generate with batched prompts.
    text_to_image.generate(
        ["cute wallpaper art of a cat", "cute wallpaper art of a dog"]
    )

    # Generate with different `num_steps` and `guidance_scale`.
    text_to_image.generate(
        prompt,
        num_steps=50,
        guidance_scale=5.0,
    )

    # Generate with `negative_prompts`.
    text_to_image.generate(
        {
            "prompts": prompt,
            "negative_prompts": "green color",
        }
    )
    ```
    """

    backbone_cls = FluxBackbone
    preprocessor_cls = FluxTextToImagePreprocessor

    def __init__(
        self,
        backbone,
        preprocessor,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        outputs = backbone.output
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Currently, `fit` is not supported for `FluxTextToImage`."
        )

    def generate_step(
        self,
        latents,
        token_ids,
        num_steps,
        guidance_scale,
    ):
        """A compilable generation function for batched of inputs.

        This function represents the inner, XLA-compilable, generation function
        for batched inputs.

        Args:
            latents: A (batch_size, height, width, channels) tensor
                containing the latents to start generation from. Typically, this
                tensor is sampled from the Gaussian distribution.
            token_ids: A pair of (batch_size, num_tokens) tensor containing the
                tokens based on the input prompts and negative prompts.
            num_steps: int. The number of diffusion steps to take.
            guidance_scale: float. The classifier free guidance scale defined in
                [Classifier-Free Diffusion Guidance](
                https://arxiv.org/abs/2207.12598). Higher scale encourages to
                generate images that are closely linked to prompts, usually at
                the expense of lower image quality.
        """
        token_ids, negative_token_ids = token_ids

        # Encode prompts.
        embeddings = self.backbone.encode_text_step(
            token_ids, negative_token_ids
        )

        # Denoise.
        def body_fun(step, latents):
            return self.backbone.denoise_step(
                latents,
                embeddings,
                step,
                num_steps,
                guidance_scale,
            )

        latents = ops.fori_loop(0, num_steps, body_fun, latents)

        # Decode.
        return self.backbone.decode_step(latents)

    def generate(
        self,
        inputs,
        num_steps=28,
        guidance_scale=7.0,
        seed=None,
    ):
        return super().generate(
            inputs,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
