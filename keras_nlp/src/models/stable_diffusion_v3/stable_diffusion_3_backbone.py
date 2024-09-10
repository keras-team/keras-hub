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
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.stable_diffusion_v3.clip_text_encoder import (
    CLIPTextEncoder,
)
from keras_nlp.src.models.stable_diffusion_v3.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler,
)
from keras_nlp.src.models.stable_diffusion_v3.mmdit import MMDiT
from keras_nlp.src.models.stable_diffusion_v3.t5_text_encoder import (
    T5TextEncoder,
)
from keras_nlp.src.models.stable_diffusion_v3.vae_image_decoder import (
    VAEImageDecoder,
)
from keras_nlp.src.utils.keras_utils import standardize_data_format


@keras_nlp_export("keras_nlp.models.StableDiffusion3Backbone")
class StableDiffusion3Backbone(Backbone):
    def __init__(
        self,
        height,
        width,
        # CLIP L
        clip_l_vocabulary_size,
        clip_l_sequence_length,
        clip_l_embedding_dim,
        clip_l_hidden_dim,
        clip_l_num_layers,
        clip_l_num_heads,
        clip_l_intermediate_dim,
        clip_l_intermediate_activation,
        clip_l_intermediate_output_index,
        # CLIP G
        clip_g_vocabulary_size,
        clip_g_sequence_length,
        clip_g_embedding_dim,
        clip_g_hidden_dim,
        clip_g_num_layers,
        clip_g_num_heads,
        clip_g_intermediate_dim,
        clip_g_intermediate_activation,
        clip_g_intermediate_output_index,
        # T5
        t5_vocabulary_size,
        t5_num_layers,
        t5_num_heads,
        t5_hidden_dim,
        t5_intermediate_dim,
        # MMDiT
        mmdit_patch_size,
        mmdit_num_heads,
        mmdit_hidden_dim,
        mmdit_depth,
        mmdit_position_size,
        mmdit_output_dim,
        mmdit_mlp_ratio,
        # VAE
        vae_stackwise_num_filters,
        vae_stackwise_num_blocks,
        vae_output_channels,
        # Scheduler
        num_train_timesteps,
        shift,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "`height` and `width` must be divisible by 8. "
                f"Received: height={height}, width={width}"
            )
        latent_shape = (height // 8, width // 8, 16)
        context_shape = (None, 4096)
        pooled_projection_shape = (2048,)
        t5_hidden_dim = t5_hidden_dim or 4096  # Defaults to 4096
        data_format = standardize_data_format(data_format)

        # === Layers ===
        self.clip_l_text_encoder = None
        self.clip_g_text_encoder = None
        self.t5_xxl_text_encoder = None
        self.clip_l_text_encoder = CLIPTextEncoder(
            clip_l_embedding_dim,
            clip_l_hidden_dim,
            clip_l_num_layers,
            clip_l_num_heads,
            clip_l_intermediate_dim,
            clip_l_intermediate_activation,
            clip_l_intermediate_output_index,
            clip_l_vocabulary_size,
            clip_l_sequence_length,
            dtype=dtype,
            name="clip_l_text_encoder",
        )
        self.clip_g_text_encoder = CLIPTextEncoder(
            clip_g_embedding_dim,
            clip_g_hidden_dim,
            clip_g_num_layers,
            clip_g_num_heads,
            clip_g_intermediate_dim,
            clip_g_intermediate_activation,
            clip_g_intermediate_output_index,
            clip_g_vocabulary_size,
            clip_g_sequence_length,
            dtype=dtype,
            name="clip_g_text_encoder",
        )
        if t5_vocabulary_size is not None:
            self.t5_text_encoder = T5TextEncoder(
                t5_vocabulary_size,
                t5_num_layers,
                t5_num_heads,
                t5_hidden_dim,
                t5_intermediate_dim,
                activation="gelu",
                dtype=dtype,
                name="t5_text_encoder",
            )
        self.mmdit_diffuser = MMDiT(
            mmdit_patch_size,
            mmdit_num_heads,
            mmdit_hidden_dim,
            mmdit_depth,
            mmdit_position_size,
            mmdit_output_dim,
            mmdit_mlp_ratio,
            latent_shape=latent_shape,
            context_shape=context_shape,
            pooled_projection_shape=pooled_projection_shape,
            data_format=data_format,
            dtype=dtype,
            name="mmdit_diffuser",
        )
        self.vae_image_decoder = VAEImageDecoder(
            vae_stackwise_num_filters,
            vae_stackwise_num_blocks,
            vae_output_channels,
            latent_shape=latent_shape,
            data_format=data_format,
            dtype=dtype,
            name="vae_image_decoder",
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )

        # === Functional Model ===
        # TODO: Define functional model here.
        super().__init__(
            dtype=dtype,
            **kwargs,
        )
        self._initialized = False

        # === Config ===
        self.height = height
        self.width = width
        # CLIP L
        self.clip_l_vocabulary_size = clip_l_vocabulary_size
        self.clip_l_sequence_length = clip_l_sequence_length
        self.clip_l_embedding_dim = clip_l_embedding_dim
        self.clip_l_hidden_dim = clip_l_hidden_dim
        self.clip_l_num_layers = clip_l_num_layers
        self.clip_l_num_heads = clip_l_num_heads
        self.clip_l_intermediate_dim = clip_l_intermediate_dim
        self.clip_l_intermediate_activation = clip_l_intermediate_activation
        self.clip_l_intermediate_output_index = clip_l_intermediate_output_index
        # CLIP G
        self.clip_g_vocabulary_size = clip_g_vocabulary_size
        self.clip_g_sequence_length = clip_g_sequence_length
        self.clip_g_embedding_dim = clip_g_embedding_dim
        self.clip_g_hidden_dim = clip_g_hidden_dim
        self.clip_g_num_layers = clip_g_num_layers
        self.clip_g_num_heads = clip_g_num_heads
        self.clip_g_intermediate_dim = clip_g_intermediate_dim
        self.clip_g_intermediate_activation = clip_g_intermediate_activation
        self.clip_g_intermediate_output_index = clip_g_intermediate_output_index
        # T5
        self.t5_vocabulary_size = t5_vocabulary_size
        self.t5_num_layers = t5_num_layers
        self.t5_num_heads = t5_num_heads
        self.t5_hidden_dim = t5_hidden_dim
        self.t5_intermediate_dim = t5_intermediate_dim
        # MMDiT
        self.mmdit_patch_size = mmdit_patch_size
        self.mmdit_num_heads = mmdit_num_heads
        self.mmdit_hidden_dim = mmdit_hidden_dim
        self.mmdit_depth = mmdit_depth
        self.mmdit_position_size = mmdit_position_size
        self.mmdit_output_dim = mmdit_output_dim
        self.mmdit_mlp_ratio = mmdit_mlp_ratio
        # VAE
        self.vae_stackwise_num_filters = vae_stackwise_num_filters
        self.vae_stackwise_num_blocks = vae_stackwise_num_blocks
        self.vae_output_channels = vae_output_channels
        # Scheduler
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps

        self.data_format = data_format

    @property
    def latent_shape(self):
        if self.data_format == "channels_last":
            return (None, self.height // 8, self.width // 8, 16)
        else:
            return (None, 16, self.height // 8, self.width // 8)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "clip_l_vocabulary_size": self.clip_l_vocabulary_size,
                "clip_l_sequence_length": self.clip_l_sequence_length,
                "clip_l_embedding_dim": self.clip_l_embedding_dim,
                "clip_l_hidden_dim": self.clip_l_hidden_dim,
                "clip_l_num_layers": self.clip_l_num_layers,
                "clip_l_num_heads": self.clip_l_num_heads,
                "clip_l_intermediate_dim": self.clip_l_intermediate_dim,
                "clip_l_intermediate_activation": self.clip_l_intermediate_activation,
                "clip_l_intermediate_output_index": self.clip_l_intermediate_output_index,
                "clip_g_vocabulary_size": self.clip_g_vocabulary_size,
                "clip_g_sequence_length": self.clip_g_sequence_length,
                "clip_g_embedding_dim": self.clip_g_embedding_dim,
                "clip_g_hidden_dim": self.clip_g_hidden_dim,
                "clip_g_num_layers": self.clip_g_num_layers,
                "clip_g_num_heads": self.clip_g_num_heads,
                "clip_g_intermediate_dim": self.clip_g_intermediate_dim,
                "clip_g_intermediate_activation": self.clip_g_intermediate_activation,
                "clip_g_intermediate_output_index": self.clip_g_intermediate_output_index,
                "t5_vocabulary_size": self.t5_vocabulary_size,
                "t5_num_layers": self.t5_num_layers,
                "t5_num_heads": self.t5_num_heads,
                "t5_hidden_dim": self.t5_hidden_dim,
                "t5_intermediate_dim": self.t5_intermediate_dim,
                "mmdit_patch_size": self.mmdit_patch_size,
                "mmdit_num_heads": self.mmdit_num_heads,
                "mmdit_hidden_dim": self.mmdit_hidden_dim,
                "mmdit_depth": self.mmdit_depth,
                "mmdit_position_size": self.mmdit_position_size,
                "mmdit_output_dim": self.mmdit_output_dim,
                "mmdit_mlp_ratio": self.mmdit_mlp_ratio,
                "vae_stackwise_num_filters": self.vae_stackwise_num_filters,
                "vae_stackwise_num_blocks": self.vae_stackwise_num_blocks,
                "vae_output_channels": self.vae_output_channels,
                "num_train_timesteps": self.num_train_timesteps,
                "shift": self.shift,
            }
        )
        return config
