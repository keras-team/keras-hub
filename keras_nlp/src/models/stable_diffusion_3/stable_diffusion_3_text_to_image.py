# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (
    StableDiffusion3Backbone,
)
from keras_nlp.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image_preprocessor import (
    StableDiffusion3TextToImagePreprocessor,
)
from keras_nlp.src.models.text_to_image import TextToImage


@keras_nlp_export("keras_nlp.models.StableDiffusion3TextToImage")
class StableDiffusion3TextToImage(TextToImage):
    """An end-to-end Stable Diffusion 3 model for text-to-image generation.

    This model has a `generate()` method, which generates image based on a
    prompt.

    Args:
        backbone: A `keras_nlp.models.StableDiffusion3Backbone` instance.
        preprocessor: A
            `keras_nlp.models.StableDiffusion3TextToImagePreprocessor` instance.

    Examples:

    Use `generate()` to do image generation.
    ```python
    text_to_image = keras_nlp.models.StableDiffusion3TextToImage.from_preset(
        "stable_diffusion_3_medium", height=512, width=512
    )
    text_to_image.generate(
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    # Generate with batched prompts.
    text_to_image.generate(
        ["cute wallpaper art of a cat", "cute wallpaper art of a dog"]
    )

    # Generate with different `num_steps` and `classifier_free_guidance_scale`.
    text_to_image.generate(
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        num_steps=50,
        classifier_free_guidance_scale=5.0,
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
        # TODO: Can we define the model here?
        super().__init__(**kwargs)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Currently, `fit` is not supported for "
            "`StableDiffusion3TextToImage`."
        )

    def encode_step(self, token_ids, negative_token_ids):
        clip_hidden_dim = self.backbone.clip_hidden_dim
        t5_hidden_dim = self.backbone.t5_hidden_dim

        def encode(token_ids):
            clip_l_outputs = self.backbone.clip_l(
                token_ids["clip_l"], training=False
            )
            clip_g_outputs = self.backbone.clip_g(
                token_ids["clip_g"], training=False
            )
            clip_l_projection = self.backbone.clip_l_projection(
                clip_l_outputs["sequence_output"],
                token_ids["clip_l"],
                training=False,
            )
            clip_g_projection = self.backbone.clip_g_projection(
                clip_g_outputs["sequence_output"],
                token_ids["clip_g"],
                training=False,
            )
            pooled_embeddings = ops.concatenate(
                [clip_l_projection, clip_g_projection],
                axis=-1,
            )
            embeddings = ops.concatenate(
                [
                    clip_l_outputs["intermediate_output"],
                    clip_g_outputs["intermediate_output"],
                ],
                axis=-1,
            )
            embeddings = ops.pad(
                embeddings,
                [[0, 0], [0, 0], [0, t5_hidden_dim - clip_hidden_dim]],
            )
            if self.backbone.t5 is not None:
                t5_outputs = self.backbone.t5(token_ids["t5"], training=False)
                embeddings = ops.concatenate([embeddings, t5_outputs], axis=-2)
            else:
                padded_size = self.preprocessor.sequence_length
                embeddings = ops.pad(
                    embeddings, [[0, 0], [0, padded_size], [0, 0]]
                )
            return embeddings, pooled_embeddings

        positive_embeddings, positive_pooled_embeddings = encode(token_ids)
        negative_embeddings, negative_pooled_embeddings = encode(
            negative_token_ids
        )

        # Concatenation for classifier-free guidance.
        embeddings = ops.concatenate(
            [positive_embeddings, negative_embeddings], axis=0
        )
        pooled_embeddings = ops.concatenate(
            [positive_pooled_embeddings, negative_pooled_embeddings], axis=0
        )
        return embeddings, pooled_embeddings

    def denoise_step(
        self,
        latents,
        embeddings,
        steps,
        num_steps,
        classifier_free_guidance_scale,
    ):
        contexts, pooled_projections = embeddings
        sigma = self.backbone.scheduler.get_sigma(steps, num_steps)
        sigma_next = self.backbone.scheduler.get_sigma(steps + 1, num_steps)

        # Sigma to timestep.
        timestep = self.backbone.scheduler._sigma_to_timestep(sigma)
        timestep = ops.broadcast_to(timestep, ops.shape(latents)[:1])

        # Diffusion.
        predicted_noise = self.backbone.mmdit(
            {
                "latent": ops.concatenate([latents, latents], axis=0),
                "context": contexts,
                "pooled_projection": pooled_projections,
                "timestep": ops.concatenate([timestep, timestep], axis=0),
            },
            training=False,
        )
        predicted_noise = ops.cast(predicted_noise, "float32")

        # Classifier-free guidance.
        classifier_free_guidance_scale = ops.cast(
            classifier_free_guidance_scale, predicted_noise.dtype
        )
        positive_noise, negative_noise = ops.split(predicted_noise, 2, axis=0)
        predicted_noise = negative_noise + classifier_free_guidance_scale * (
            positive_noise - negative_noise
        )

        # Euler step.
        return self.backbone.scheduler.step(
            latents, predicted_noise, sigma, sigma_next
        )

    def decode_step(self, latents):
        # Latent calibration.
        latents = ops.add(
            ops.divide(latents, self.backbone.vae.scaling_factor),
            self.backbone.vae.shift_factor,
        )

        # Decoding.
        return self.backbone.vae(latents, training=False)

    def generate_step(
        self,
        latents,
        token_ids,
        negative_token_ids,
        num_steps,
        classifier_free_guidance_scale,
    ):
        """A compilable generation function for batched of inputs.

        This function represents the inner, XLA-compilable, generation function
        for batched inputs.

        Args:
            latents: A <float>[batch_size, height, width, channels] tensor
                containing the latents to start generation from. Typically, this
                tensor is sampled from the Gaussian distribution.
            token_ids: A <int>[batch_size, num_tokens] tensor containing the
                tokens based on the input prompts.
            negative_token_ids: A <int>[batch_size, num_tokens] tensor
                 containing the negative tokens based on the input prompts.
            num_steps: int. The number of diffusion steps to take.
            classifier_free_guidance_scale: float. The scale defined in
                [Classifier-Free Diffusion Guidance](
                https://arxiv.org/abs/2207.12598). Higher scale encourages to
                generate images that are closely linked to prompts, usually at
                the expense of lower image quality.
        """
        # Encode inputs.
        embeddings = self.encode_step(token_ids, negative_token_ids)

        # Denoise.
        def body_fun(step, latents):
            latents = self.denoise_step(
                latents,
                embeddings,
                step,
                num_steps,
                classifier_free_guidance_scale,
            )
            return latents

        latents = ops.fori_loop(0, num_steps, body_fun, latents)

        # Decode.
        return self.decode_step(latents)

    def generate(
        self,
        inputs,
        negative_inputs=None,
        num_steps=28,
        classifier_free_guidance_scale=7.0,
        seed=None,
    ):
        return super().generate(
            inputs,
            negative_inputs=negative_inputs,
            num_steps=num_steps,
            classifier_free_guidance_scale=classifier_free_guidance_scale,
            seed=seed,
        )
