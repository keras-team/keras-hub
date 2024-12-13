from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.inpaint import Inpaint
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (  # noqa: E501
    StableDiffusion3Backbone,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image_preprocessor import (  # noqa: E501
    StableDiffusion3TextToImagePreprocessor,
)


@keras_hub_export("keras_hub.models.StableDiffusion3Inpaint")
class StableDiffusion3Inpaint(Inpaint):
    """An end-to-end Stable Diffusion 3 model for inpaint generation.

    This model has a `generate()` method, which generates images based
    on a combination of a reference image, mask and a text prompt.

    Args:
        backbone: A `keras_hub.models.StableDiffusion3Backbone` instance.
        preprocessor: A
            `keras_hub.models.StableDiffusion3TextToImagePreprocessor` instance.

    Examples:

    Use `generate()` to do image generation.
    ```python
    reference_image = np.ones((1024, 1024, 3), dtype="float32")
    reference_mask = np.ones((1024, 1024), dtype="float32")
    inpaint = keras_hub.models.StableDiffusion3Inpaint.from_preset(
        "stable_diffusion_3_medium", image_shape=(512, 512, 3)
    )
    inpaint.generate(
        reference_image,
        reference_mask,
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )

    # Generate with batched prompts.
    reference_images = np.ones((2, 512, 512, 3), dtype="float32")
    reference_mask = np.ones((2, 1024, 1024), dtype="float32")
    inpaint.generate(
        reference_images,
        reference_mask,
        ["cute wallpaper art of a cat", "cute wallpaper art of a dog"]
    )

    # Generate with different `num_steps`, `guidance_scale` and `strength`.
    inpaint.generate(
        reference_image,
        reference_mask,
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        num_steps=50,
        guidance_scale=5.0,
        strength=0.6,
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
            "`StableDiffusion3Inpaint`."
        )

    def generate_step(
        self,
        images,
        masks,
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
            masks: A (batch_size, image_height, image_width) tensor
                containing the reference masks.
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

        # Get masked images.
        masks = ops.cast(ops.expand_dims(masks, axis=-1) > 0.5, images.dtype)
        masks_latent_size = ops.image.resize(
            masks,
            (self.backbone.latent_shape[1], self.backbone.latent_shape[2]),
            interpolation="nearest",
        )

        # Encode images.
        image_latents = self.backbone.encode_image_step(images)

        # Add noises to latents.
        latents = self.backbone.add_noise_step(
            image_latents, noises, starting_step, num_steps
        )

        # Encode inputs.
        embeddings = self.backbone.encode_text_step(
            token_ids, negative_token_ids
        )

        # Denoise.
        def body_fun(step, latents):
            latents = self.backbone.denoise_step(
                latents,
                embeddings,
                step,
                num_steps,
                guidance_scale,
            )

            # Compute the previous latents x_t -> x_t-1.
            def true_fn():
                next_step = ops.add(step, 1)
                return self.backbone.add_noise_step(
                    image_latents, noises, next_step, num_steps
                )

            init_latents = ops.cond(
                step < ops.subtract(num_steps, 1),
                true_fn,
                lambda: ops.cast(image_latents, noises.dtype),
            )
            latents = ops.add(
                ops.multiply(
                    ops.subtract(1.0, masks_latent_size), init_latents
                ),
                ops.multiply(masks_latent_size, latents),
            )
            return latents

        latents = ops.fori_loop(starting_step, num_steps, body_fun, latents)

        # Decode.
        return self.backbone.decode_step(latents)

    def generate(
        self,
        inputs,
        num_steps=50,
        strength=0.6,
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
