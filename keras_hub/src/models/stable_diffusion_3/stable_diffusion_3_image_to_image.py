from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_to_image import ImageToImage
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (  # noqa: E501
    StableDiffusion3Backbone,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image_preprocessor import (  # noqa: E501
    StableDiffusion3TextToImagePreprocessor,
)


@keras_hub_export("keras_hub.models.StableDiffusion3ImageToImage")
class StableDiffusion3ImageToImage(ImageToImage):
    """An end-to-end Stable Diffusion 3 model for image-to-image generation.

    This model has a `generate()` method, which generates images based
    on a combination of a reference image and a text prompt.

    Args:
        backbone: A `keras_hub.models.StableDiffusion3Backbone` instance.
        preprocessor: A
            `keras_hub.models.StableDiffusion3TextToImagePreprocessor` instance.

    Examples:

    Use `generate()` to do image generation.
    ```python
    prompt = (
        "Astronaut in a jungle, cold color palette, muted colors, "
        "detailed, 8k"
    )
    image_to_image = keras_hub.models.StableDiffusion3ImageToImage.from_preset(
        "stable_diffusion_3_medium", image_shape=(512, 512, 3)
    )
    image_to_image.generate(
        {
            "images": np.ones((512, 512, 3), dtype="float32"),
            "prompts": prompt,
        }
    )

    # Generate with batched prompts.
    image_to_image.generate(
        {
            "images": np.ones((2, 512, 512, 3), dtype="float32"),
            "prompts": [
                "cute wallpaper art of a cat",
                "cute wallpaper art of a dog",
            ],
        }
    )

    # Generate with different `num_steps`, `guidance_scale` and `strength`.
    image_to_image.generate(
        {
            "images": np.ones((512, 512, 3), dtype="float32"),
            "prompts": prompt,
        }
        num_steps=50,
        guidance_scale=5.0,
        strength=0.6,
    )

    # Generate with `negative_prompts`.
    text_to_image.generate(
        {
            "images": np.ones((512, 512, 3), dtype="float32"),
            "prompts": prompt,
            "negative_prompts": "green color",
        }
    )
    ```
    """

    backbone_cls = StableDiffusion3Backbone
    preprocessor_cls = StableDiffusion3TextToImagePreprocessor

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
            "Currently, `fit` is not supported for "
            "`StableDiffusion3ImageToImage`."
        )

    def generate_step(
        self,
        images,
        noises,
        token_ids,
        starting_step,
        num_steps,
        guidance_scale,
    ):
        """A compilable generation function for batched of inputs.

        This function represents the inner, XLA-compilable, generation function
        for batched inputs.

        Args:
            images: A (batch_size, image_height, image_width, 3) tensor
                containing the reference images.
            noises: A (batch_size, latent_height, latent_width, channels) tensor
                containing the noises to be added to the latents. Typically,
                this tensor is sampled from the Gaussian distribution.
            token_ids: A pair of (batch_size, num_tokens) tensor containing the
                tokens based on the input prompts and negative prompts.
            starting_step: int. The number of the starting diffusion step.
            num_steps: int. The number of diffusion steps to take.
            guidance_scale: float. The classifier free guidance scale defined in
                [Classifier-Free Diffusion Guidance](
                https://arxiv.org/abs/2207.12598). Higher scale encourages to
                generate images that are closely linked to prompts, usually at
                the expense of lower image quality.
        """
        token_ids, negative_token_ids = token_ids

        # Encode images.
        latents = self.backbone.encode_image_step(images)

        # Add noises to latents.
        latents = self.backbone.add_noise_step(
            latents, noises, starting_step, num_steps
        )

        # Encode inputs.
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

        latents = ops.fori_loop(starting_step, num_steps, body_fun, latents)

        # Decode.
        return self.backbone.decode_step(latents)

    def generate(
        self,
        inputs,
        num_steps=50,
        strength=0.8,
        guidance_scale=7.0,
        seed=None,
    ):
        return super().generate(
            inputs,
            num_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            seed=seed,
        )
