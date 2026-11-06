from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.flux.flux_backbone import FluxBackbone
from keras_hub.src.models.flux.flux_text_to_image_preprocessor import (
    FluxTextToImagePreprocessor,
)
from keras_hub.src.models.text_to_image import TextToImage


@keras_hub_export("keras_hub.models.FluxTextToImage")
class FluxTextToImage(TextToImage):
    """An end-to-end Flux model for text-to-image generation.

    Note: FLUX.1 generation is not supported yet. Only the FLUX.1
    transformer (`keras_hub.models.FluxBackbone`) is ported, and the
    `flux1_schnell` preset holds just the transformer weights, without the
    CLIP-L and T5-XXL text encoders or the VAE that generation needs.
    `generate()` raises a `NotImplementedError` until they are added. For
    end-to-end generation, use `keras_hub.models.Flux2TextToImage`.

    Args:
        backbone: A `keras_hub.models.FluxBackbone` instance.
        preprocessor: A
            `keras_hub.models.FluxTextToImagePreprocessor` instance.

    Examples:

    Load the pretrained FLUX.1 [schnell] transformer.
    ```python
    backbone = keras_hub.models.FluxBackbone.from_preset("flux1_schnell")
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
        raise NotImplementedError(
            "FLUX.1 generation is not supported yet: only the FLUX.1 "
            "transformer (`FluxBackbone`) is ported, without the CLIP-L and "
            "T5-XXL text encoders and the VAE. Use "
            "`keras_hub.models.Flux2TextToImage` for text-to-image generation."
        )
