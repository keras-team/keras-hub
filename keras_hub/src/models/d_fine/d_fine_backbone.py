import math

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.d_fine.d_fine_decoder import DFineDecoder
from keras_hub.src.models.d_fine.d_fine_hybrid_encoder import DFineHybridEncoder
from keras_hub.src.models.d_fine.d_fine_layers import DFineAnchorGenerator
from keras_hub.src.models.d_fine.d_fine_layers import (
    DFineContrastiveDenoisingGroupGenerator,
)
from keras_hub.src.models.d_fine.d_fine_layers import (
    DFineInitialQueryAndReferenceGenerator,
)
from keras_hub.src.models.d_fine.d_fine_layers import DFineMLPPredictionHead
from keras_hub.src.models.d_fine.d_fine_layers import DFineSourceFlattener
from keras_hub.src.models.d_fine.d_fine_layers import (
    DFineSpatialShapesExtractor,
)
from keras_hub.src.models.d_fine.d_fine_utils import d_fine_kernel_initializer
from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras.saving.register_keras_serializable(package="keras_hub")
class DFineDenoisingTensorProcessor(keras.layers.Layer):
    """Processes and prepares tensors for contrastive denoising.

    This layer is a helper used within the `DFineBackbone`'s functional model
    definition. Its primary role is to take the outputs from the
    `DFineContrastiveDenoisingGroupGenerator` and prepare them for the dynamic,
    per-batch forward pass, mostly since this functionality cannot be integrated
    directly into the `DFineBackbone` in the symbolic forward pass.

    The layer takes a tuple of `(pixel_values, input_query_class,
    denoising_bbox_unact, attention_mask)` and an optional
    `denoising_meta_values` dictionary as input to its `call` method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, denoising_meta_values=None):
        (
            pixel_values,
            input_query_class,
            denoising_bbox_unact,
            attention_mask,
        ) = inputs
        input_query_class_tensor = keras.ops.convert_to_tensor(
            input_query_class, dtype="int32"
        )
        denoising_bbox_unact_tensor = keras.ops.convert_to_tensor(
            denoising_bbox_unact, dtype=pixel_values.dtype
        )
        attention_mask_tensor = keras.ops.convert_to_tensor(
            attention_mask, dtype=pixel_values.dtype
        )
        outputs = {
            "input_query_class": input_query_class_tensor,
            "denoising_bbox_unact": denoising_bbox_unact_tensor,
            "attention_mask": attention_mask_tensor,
        }

        if denoising_meta_values is not None:
            batch_size = keras.ops.shape(pixel_values)[0]
            dn_positive_idx = denoising_meta_values["dn_positive_idx"]
            c_batch_size = keras.ops.shape(dn_positive_idx)[0]
            if c_batch_size == 0:
                outputs["dn_positive_idx"] = keras.ops.zeros(
                    (batch_size,) + keras.ops.shape(dn_positive_idx)[1:],
                    dtype=dn_positive_idx.dtype,
                )
            else:
                num_repeats = (batch_size + c_batch_size - 1) // c_batch_size
                dn_positive_idx_tiled = keras.ops.tile(
                    dn_positive_idx,
                    (num_repeats,)
                    + (1,) * (keras.ops.ndim(dn_positive_idx) - 1),
                )
                outputs["dn_positive_idx"] = dn_positive_idx_tiled[:batch_size]
            dn_num_group = denoising_meta_values["dn_num_group"]
            outputs["dn_num_group"] = keras.ops.tile(
                keras.ops.expand_dims(dn_num_group, 0), (batch_size,)
            )
            dn_num_split = denoising_meta_values["dn_num_split"]
            outputs["dn_num_split"] = keras.ops.tile(
                keras.ops.expand_dims(dn_num_split, 0), (batch_size, 1)
            )

        return outputs


@keras_hub_export("keras_hub.models.DFineBackbone")
class DFineBackbone(Backbone):
    """D-FINE Backbone for Object Detection.

    This class implements the core D-FINE architecture, which serves as the
    backbone for `DFineObjectDetector`. It integrates a `HGNetV2Backbone` for
    initial feature extraction, a `DFineHybridEncoder` for multi-scale feature
    fusion using FPN/PAN pathways, and a `DFineDecoder` for refining object
    queries.

    The backbone orchestrates the entire forward pass, from processing raw
    pixels to generating intermediate predictions. Key steps include:
    1.  Extracting multi-scale feature maps using the HGNetV2 backbone.
    2.  Fusing these features with the hybrid encoder.
    3.  Generating anchor proposals and selecting the top-k to initialize
        decoder queries and reference points.
    4.  Generating noisy queries for contrastive denoising (if the `labels`
        argument is provided).
    5.  Passing the queries and fused features through the transformer decoder
        to produce iterative predictions for bounding boxes and class logits.

    Args:
        decoder_in_channels: list, Channel dimensions of the multi-scale
            features from the hybrid encoder. This should typically be a list
            of `encoder_hidden_dim` repeated for each feature level.
        encoder_hidden_dim: int, Hidden dimension size for the encoder layers.
        num_labels: int, Number of object classes for detection.
        num_denoising: int, Number of denoising queries for contrastive
            denoising training. Set to `0` to disable denoising.
        learn_initial_query: bool, Whether to learn initial query embeddings.
            Defaults to `False`.
        num_queries: int, Number of object queries for detection.
        anchor_image_size: tuple, Size of the anchor image as `(height, width)`.
        feat_strides: list, List of feature stride values for different pyramid
            levels.
        batch_norm_eps: float, Epsilon value for batch normalization layers.
        num_feature_levels: int, Number of feature pyramid levels to use.
        hidden_dim: int, Hidden dimension size for the model.
        layer_norm_eps: float, Epsilon value for layer normalization.
        encoder_in_channels: list, Channel dimensions of the feature maps from
            the backbone (`HGNetV2Backbone`) that are fed into the hybrid
            encoder.
        encode_proj_layers: list, List specifying projection layer
            configurations.
        positional_encoding_temperature: float, Temperature parameter for
            positional encoding.
        eval_size: tuple, Evaluation image size.
        normalize_before: bool, Whether to apply layer normalization before
            attention layers.
        num_attention_heads: int, Number of attention heads in encoder layers.
        dropout: float, Dropout rate for encoder layers.
        encoder_activation_function: str, Activation function for encoder
            (e.g., `"gelu"`, `"relu"`).
        activation_dropout: float, Dropout rate for activation layers.
        encoder_ffn_dim: int, Feed-forward network dimension in encoder.
        encoder_layers: int, Number of encoder layers.
        hidden_expansion: float, Hidden dimension expansion factor.
        depth_mult: float, Depth multiplier for the backbone.
        eval_idx: int, Index for evaluation (`-1` for last layer).
        decoder_layers: int, Number of decoder layers.
        reg_scale: float, Regression scale factor.
        max_num_bins: int, Maximum number of bins for discrete coordinate
            prediction.
        up: float, Upsampling factor.
        decoder_attention_heads: int, Number of attention heads in decoder
            layers.
        attention_dropout: float, Dropout rate for attention layers.
        decoder_activation_function: str, Activation function for decoder
            layers.
        decoder_ffn_dim: int, Feed-forward network dimension in decoder.
        decoder_offset_scale: float, Scale factor for decoder offset
            predictions.
        decoder_method: str, Decoder method (`"default"` or `"discrete"`).
        decoder_n_points: list, Number of sampling points for deformable
            attention.
        top_prob_values: int, Number of top probability values to consider.
        lqe_hidden_dim: int, Hidden dimension for learned query embedding.
        lqe_layers_count: int, Number of layers in learned query embedding.
        hidden_act: str, Hidden activation function for backbone layers.
        stem_channels: list, List of channel dimensions for stem layers.
        use_learnable_affine_block: bool, Whether to use learnable affine
            blocks.
        stackwise_stage_filters: list, Configuration for backbone stage filters.
            Each element is a list of `[in_channels, mid_channels, out_channels,
            num_blocks, num_layers, kernel_size]`.
        apply_downsample: list, List of booleans indicating whether to apply
            downsampling at each stage.
        use_lightweight_conv_block: list, List of booleans indicating whether
            to use lightweight convolution blocks at each stage.
        depths: list, List of depths for each backbone stage.
        hidden_sizes: list, List of hidden sizes for each backbone stage.
        embedding_size: int, Embedding dimension size.
        layer_scale: float, Layer scale parameter for residual connections.
            Defaults to `1.0`.
        label_noise_ratio: float, Ratio of label noise for denoising training.
            Defaults to `0.5`.
        initializer_bias_prior_prob: float, optional, Prior probability for
            the bias of the classification head. Used to initialize the bias
            of the `class_embed` and `enc_score_head` layers. Defaults to
            `None`, and `prior_prob` computed as `prior_prob = 1 /
            (num_labels + 1)` while initializing model weights.
        initializer_range: float, optional, The standard deviation for the
            `RandomNormal` initializer. Defaults to `0.01`.
        box_noise_scale: float, Scale factor for box noise in denoising
            training. Defaults to `1.0`.
        labels: list or None, Ground truth labels for denoising training. This
            is passed during model initialization to construct the training
            graph for contrastive denoising. Each element should be a
            dictionary with `"boxes"` (numpy array of shape `[N, 4]` with
            normalized coordinates) and `"labels"` (numpy array of shape `[N]`
            with class indices). Required when `num_denoising > 0`.
        seed: int or None, Random seed for reproducibility.
        image_shape: tuple, Shape of input images as `(height, width,
            channels)`. Height and width can be `None` for variable input sizes.
        out_features: list or None, List of feature names to output from
            backbone. If `None`, uses the last `len(decoder_in_channels)`
            features.
        data_format: str, Data format (`"channels_first"` or `"channels_last"`).
        dtype: str, Data type for model parameters.
        **kwargs: Additional keyword arguments passed to the base class.

    Example:
    ```python
    import keras
    import numpy as np
    from keras_hub.models import DFineBackbone

    # Example 1: Basic usage without denoising.
    backbone = DFineBackbone(
        decoder_in_channels=[128, 128],
        encoder_hidden_dim=128,
        num_labels=80,
        num_denoising=0,  # Disable denoising
        hidden_dim=128,
        num_queries=300,
        anchor_image_size=(256, 256),
        feat_strides=[16, 32],
        batch_norm_eps=1e-5,
        num_feature_levels=2,
        layer_norm_eps=1e-5,
        encoder_in_channels=[512, 1024],
        encode_proj_layers=[1],
        positional_encoding_temperature=10000,
        num_attention_heads=8,
        encoder_activation_function="gelu",
        encoder_ffn_dim=512,
        encoder_layers=1,
        decoder_layers=3,
        decoder_attention_heads=8,
        decoder_activation_function="relu",
        decoder_ffn_dim=512,
        stem_channels=[3, 16, 16],
        stackwise_stage_filters=[
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ],
        apply_downsample=[False, True, True, True],
        use_lightweight_conv_block=[False, False, True, True],
        depths=[1, 1, 2, 1],
        hidden_sizes=[64, 256, 512, 1024],
        embedding_size=16,
        image_shape=(None, None, 3),
    )

    # Prepare input data.
    input_data = {
        "pixel_values": keras.random.uniform((2, 256, 256, 3)),
    }

    # Forward pass.
    outputs = backbone(input_data)

    # Example 2: With contrastive denoising training.
    labels = [
        {
            "boxes": np.array([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
            "labels": np.array([1, 10]),
        },
        {
            "boxes": np.array([[0.6, 0.6, 0.3, 0.3]]),
            "labels": np.array([20]),
        },
    ]

    backbone_with_denoising = DFineBackbone(
        decoder_in_channels=[128, 128],
        encoder_hidden_dim=128,
        num_labels=80,
        num_denoising=100,  # Enable denoising
        hidden_dim=128,
        num_queries=300,
        anchor_image_size=(256, 256),
        feat_strides=[16, 32],
        batch_norm_eps=1e-5,
        num_feature_levels=2,
        layer_norm_eps=1e-5,
        encoder_in_channels=[512, 1024],
        encode_proj_layers=[1],
        positional_encoding_temperature=10000,
        num_attention_heads=8,
        encoder_activation_function="gelu",
        encoder_ffn_dim=512,
        encoder_layers=1,
        decoder_layers=3,
        decoder_attention_heads=8,
        decoder_activation_function="relu",
        decoder_ffn_dim=512,
        stem_channels=[3, 16, 16],
        stackwise_stage_filters=[
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ],
        apply_downsample=[False, True, True, True],
        use_lightweight_conv_block=[False, False, True, True],
        depths=[1, 1, 2, 1],
        hidden_sizes=[64, 256, 512, 1024],
        embedding_size=16,
        image_shape=(None, None, 3),
        # Denoising parameters
        box_noise_scale=1.0,
        label_noise_ratio=0.5,
        labels=labels,  # Required for denoising training
        seed=0,
    )

    # Forward pass with denoising.
    outputs_with_denoising = backbone_with_denoising(input_data)
    ```
    """

    def __init__(
        self,
        decoder_in_channels,
        encoder_hidden_dim,
        num_labels,
        num_denoising,
        learn_initial_query,
        num_queries,
        anchor_image_size,
        feat_strides,
        batch_norm_eps,
        num_feature_levels,
        hidden_dim,
        layer_norm_eps,
        encoder_in_channels,
        encode_proj_layers,
        positional_encoding_temperature,
        eval_size,
        normalize_before,
        num_attention_heads,
        dropout,
        encoder_activation_function,
        activation_dropout,
        encoder_ffn_dim,
        encoder_layers,
        hidden_expansion,
        depth_mult,
        eval_idx,
        decoder_layers,
        reg_scale,
        max_num_bins,
        up,
        decoder_attention_heads,
        attention_dropout,
        decoder_activation_function,
        decoder_ffn_dim,
        decoder_offset_scale,
        decoder_method,
        decoder_n_points,
        top_prob_values,
        lqe_hidden_dim,
        lqe_layers_count,
        hidden_act,
        stem_channels,
        use_learnable_affine_block,
        stackwise_stage_filters,
        apply_downsample,
        use_lightweight_conv_block,
        depths,
        hidden_sizes,
        embedding_size,
        layer_scale=1.0,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        initializer_bias_prior_prob=None,
        initializer_range=0.01,
        labels=None,
        seed=None,
        image_shape=(None, None, 3),
        out_features=None,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if decoder_method not in ["default", "discrete"]:
            decoder_method = "default"
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1
        self.stackwise_stage_filters = stackwise_stage_filters
        spatial_shapes_list = []
        for s in feat_strides:
            h = anchor_image_size[0] // s
            w = anchor_image_size[1] // s
            spatial_shapes_list.append((h, w))
        stage_names = ["stem"] + [
            f"stage{i + 1}" for i in range(len(self.stackwise_stage_filters))
        ]
        out_features = (
            out_features
            if out_features is not None
            else stage_names[-len(decoder_in_channels) :]
        )
        initializer = d_fine_kernel_initializer(
            initializer_range=initializer_range
        )

        # === Layers ===
        self.encoder = DFineHybridEncoder(
            encoder_in_channels=encoder_in_channels,
            feat_strides=feat_strides,
            encoder_hidden_dim=encoder_hidden_dim,
            encode_proj_layers=encode_proj_layers,
            positional_encoding_temperature=positional_encoding_temperature,
            eval_size=eval_size,
            normalize_before=normalize_before,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            encoder_activation_function=encoder_activation_function,
            activation_dropout=activation_dropout,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_layers=encoder_layers,
            batch_norm_eps=batch_norm_eps,
            hidden_expansion=hidden_expansion,
            depth_mult=depth_mult,
            kernel_initializer=initializer,
            bias_initializer="zeros",
            channel_axis=channel_axis,
            data_format=data_format,
            dtype=dtype,
            name="encoder",
        )
        self.decoder = DFineDecoder(
            layer_scale=layer_scale,
            eval_idx=eval_idx,
            decoder_layers=decoder_layers,
            dropout=dropout,
            hidden_dim=hidden_dim,
            reg_scale=reg_scale,
            max_num_bins=max_num_bins,
            up=up,
            decoder_attention_heads=decoder_attention_heads,
            attention_dropout=attention_dropout,
            decoder_activation_function=decoder_activation_function,
            activation_dropout=activation_dropout,
            layer_norm_eps=layer_norm_eps,
            decoder_ffn_dim=decoder_ffn_dim,
            num_feature_levels=num_feature_levels,
            decoder_offset_scale=decoder_offset_scale,
            decoder_method=decoder_method,
            decoder_n_points=decoder_n_points,
            top_prob_values=top_prob_values,
            lqe_hidden_dim=lqe_hidden_dim,
            lqe_layers_count=lqe_layers_count,
            num_labels=num_labels,
            spatial_shapes_list=spatial_shapes_list,
            dtype=dtype,
            initializer_bias_prior_prob=initializer_bias_prior_prob,
            num_queries=num_queries,
            name="decoder",
        )
        self.anchor_generator = DFineAnchorGenerator(
            anchor_image_size=anchor_image_size,
            feat_strides=feat_strides,
            dtype=dtype,
            name="anchor_generator",
        )
        self.contrastive_denoising_group_generator = (
            DFineContrastiveDenoisingGroupGenerator(
                num_labels=num_labels,
                num_denoising=num_denoising,
                label_noise_ratio=label_noise_ratio,
                box_noise_scale=box_noise_scale,
                seed=seed,
                dtype=dtype,
                name="contrastive_denoising_group_generator",
            )
        )
        if num_denoising > 0:
            self.denoising_class_embed = keras.layers.Embedding(
                input_dim=num_labels + 1,
                output_dim=hidden_dim,
                embeddings_initializer="glorot_uniform",
                name="denoising_class_embed",
                dtype=dtype,
            )
            self.denoising_class_embed.build(None)
        else:
            self.denoising_class_embed = None

        self.source_flattener = DFineSourceFlattener(
            dtype=dtype,
            name="source_flattener",
            channel_axis=channel_axis,
            data_format=data_format,
        )
        self.initial_query_reference_generator = (
            DFineInitialQueryAndReferenceGenerator(
                num_queries=num_queries,
                learn_initial_query=learn_initial_query,
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="initial_query_reference_generator",
            )
        )
        self.spatial_shapes_extractor = DFineSpatialShapesExtractor(
            dtype=dtype,
            data_format=data_format,
            name="spatial_shapes_extractor",
        )
        self.hgnetv2_backbone = HGNetV2Backbone(
            depths=depths,
            embedding_size=embedding_size,
            hidden_sizes=hidden_sizes,
            stem_channels=stem_channels,
            hidden_act=hidden_act,
            use_learnable_affine_block=use_learnable_affine_block,
            stackwise_stage_filters=stackwise_stage_filters,
            apply_downsample=apply_downsample,
            use_lightweight_conv_block=use_lightweight_conv_block,
            image_shape=image_shape,
            data_format=data_format,
            out_features=out_features,
            dtype=dtype,
            name="hgnetv2_backbone",
        )
        num_backbone_outs = len(decoder_in_channels)
        self.encoder_input_proj = []
        for i in range(num_backbone_outs):
            proj_layer = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        filters=encoder_hidden_dim,
                        kernel_size=1,
                        use_bias=False,
                        kernel_initializer=initializer,
                        bias_initializer="zeros",
                        data_format=data_format,
                        name=f"encoder_input_proj_conv_{i}",
                    ),
                    keras.layers.BatchNormalization(
                        epsilon=batch_norm_eps,
                        axis=channel_axis,
                        name=f"encoder_input_proj_bn_{i}",
                    ),
                ],
                name=f"encoder_input_proj_{i}",
            )
            self.encoder_input_proj.append(proj_layer)
        self.enc_output = keras.Sequential(
            [
                keras.layers.Dense(hidden_dim, name="enc_output_dense"),
                keras.layers.LayerNormalization(
                    epsilon=layer_norm_eps, name="enc_output_ln"
                ),
            ],
            name="enc_output",
        )
        if initializer_bias_prior_prob is None:
            prior_prob = 1 / (num_labels + 1)
        else:
            prior_prob = initializer_bias_prior_prob
        enc_score_head_bias = float(-math.log((1 - prior_prob) / prior_prob))
        self.enc_score_head = keras.layers.Dense(
            num_labels,
            name="enc_score_head",
            dtype=dtype,
            kernel_initializer="glorot_uniform",
            bias_initializer=keras.initializers.Constant(enc_score_head_bias),
        )
        self.enc_bbox_head = DFineMLPPredictionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=4,
            num_layers=3,
            name="enc_bbox_head",
            dtype=dtype,
            kernel_initializer=initializer,
            last_layer_initializer="zeros",
        )
        self.decoder_input_proj = []
        for i in range(num_backbone_outs):
            if hidden_dim == decoder_in_channels[-1]:
                proj_layer = keras.layers.Identity(
                    name=f"decoder_input_proj_identity_{i}"
                )
            else:
                proj_layer = keras.Sequential(
                    [
                        keras.layers.Conv2D(
                            filters=hidden_dim,
                            kernel_size=1,
                            use_bias=False,
                            kernel_initializer=initializer,
                            bias_initializer="zeros",
                            data_format=data_format,
                            name=f"decoder_input_proj_conv1_{i}",
                        ),
                        keras.layers.BatchNormalization(
                            epsilon=batch_norm_eps,
                            axis=channel_axis,
                            name=f"decoder_input_proj_bn1_{i}",
                        ),
                    ],
                    name=f"decoder_input_proj_{i}",
                )
            self.decoder_input_proj.append(proj_layer)
        for i in range(num_feature_levels - num_backbone_outs):
            idx = num_backbone_outs + i
            if hidden_dim == decoder_in_channels[-1]:
                proj_layer = keras.layers.Identity(
                    name=f"decoder_input_proj_identity_{idx}"
                )
            else:
                proj_layer = keras.Sequential(
                    [
                        keras.layers.Conv2D(
                            filters=hidden_dim,
                            kernel_size=3,
                            strides=2,
                            padding="same",
                            use_bias=False,
                            kernel_initializer=initializer,
                            bias_initializer="zeros",
                            data_format=data_format,
                            name=f"decoder_input_proj_conv3_{idx}",
                        ),
                        keras.layers.BatchNormalization(
                            epsilon=batch_norm_eps,
                            axis=channel_axis,
                            name=f"decoder_input_proj_bn3_{idx}",
                        ),
                    ],
                    name=f"decoder_input_proj_{idx}",
                    dtype=dtype,
                )
            self.decoder_input_proj.append(proj_layer)

        # === Functional Model ===
        pixel_values = keras.Input(
            shape=image_shape, name="pixel_values", dtype="float32"
        )
        feature_maps_output = self.hgnetv2_backbone(pixel_values)
        feature_maps_list = [
            feature_maps_output[stage] for stage in out_features
        ]
        feature_maps_output_tuple = tuple(feature_maps_list)
        proj_feats = [
            self.encoder_input_proj[level](feature_map)
            for level, feature_map in enumerate(feature_maps_output_tuple)
        ]
        encoder_outputs = self.encoder(
            inputs_embeds_list=proj_feats,
            output_hidden_states=True,
            output_attentions=True,
        )
        encoder_last_hidden_state = encoder_outputs[0]
        encoder_hidden_states = (
            encoder_outputs[1] if len(encoder_outputs) > 1 else None
        )
        encoder_attentions = (
            encoder_outputs[2] if len(encoder_outputs) > 2 else None
        )
        last_hidden_state = encoder_outputs[0]
        sources = [
            self.decoder_input_proj[level](source)
            for level, source in enumerate(last_hidden_state)
        ]
        if num_feature_levels > len(sources):
            _len_sources = len(sources)
            sources.append(
                self.decoder_input_proj[_len_sources](last_hidden_state[-1])
            )
            for i in range(_len_sources + 1, num_feature_levels):
                sources.append(
                    self.decoder_input_proj[i](last_hidden_state[-1])
                )
        spatial_shapes_tensor = self.spatial_shapes_extractor(sources)
        source_flatten = self.source_flattener(sources)
        if num_denoising > 0 and labels is not None:
            (
                input_query_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = self.contrastive_denoising_group_generator(
                targets=labels,
                num_queries=num_queries,
            )
        else:
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = None, None, None, None

        if num_denoising > 0 and labels is not None:
            denoising_processor = DFineDenoisingTensorProcessor(
                name="denoising_processor"
            )
            denoising_tensors = denoising_processor(
                [
                    pixel_values,
                    input_query_class,
                    denoising_bbox_unact,
                    attention_mask,
                ],
                denoising_meta_values=denoising_meta_values,
            )
            input_query_class_tensor = denoising_tensors["input_query_class"]
            denoising_bbox_unact = denoising_tensors["denoising_bbox_unact"]
            attention_mask = denoising_tensors["attention_mask"]
            denoising_class = self.denoising_class_embed(
                input_query_class_tensor
            )

        anchors, valid_mask = self.anchor_generator(sources)
        memory = keras.ops.where(valid_mask, source_flatten, 0.0)
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory)
        _enc_outputs_coord_logits_plus_anchors = (
            enc_outputs_coord_logits + anchors
        )
        init_reference_points, target, enc_topk_logits, enc_topk_bboxes = (
            self.initial_query_reference_generator(
                (
                    enc_outputs_class,
                    _enc_outputs_coord_logits_plus_anchors,
                    output_memory,
                    sources[-1],
                ),
                denoising_bbox_unact=denoising_bbox_unact,
                denoising_class=denoising_class,
            )
        )
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes_tensor,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        last_hidden_state = decoder_outputs[0]
        intermediate_hidden_states = decoder_outputs[1]
        intermediate_logits = decoder_outputs[2]
        intermediate_reference_points = decoder_outputs[3]
        intermediate_predicted_corners = decoder_outputs[4]
        initial_reference_points = decoder_outputs[5]
        decoder_hidden_states = (
            decoder_outputs[6] if len(decoder_outputs) > 6 else None
        )
        decoder_attentions = (
            decoder_outputs[7] if len(decoder_outputs) > 7 else None
        )
        cross_attentions = (
            decoder_outputs[8] if len(decoder_outputs) > 8 else None
        )
        outputs = {
            "last_hidden_state": last_hidden_state,
            "intermediate_hidden_states": intermediate_hidden_states,
            "intermediate_logits": intermediate_logits,
            "intermediate_reference_points": intermediate_reference_points,
            "intermediate_predicted_corners": intermediate_predicted_corners,
            "initial_reference_points": initial_reference_points,
            "decoder_hidden_states": decoder_hidden_states,
            "decoder_attentions": decoder_attentions,
            "cross_attentions": cross_attentions,
            "encoder_last_hidden_state": encoder_last_hidden_state[0],
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attentions": encoder_attentions,
            "init_reference_points": init_reference_points,
            "enc_topk_logits": enc_topk_logits,
            "enc_topk_bboxes": enc_topk_bboxes,
            "enc_outputs_class": enc_outputs_class,
            "enc_outputs_coord_logits": enc_outputs_coord_logits,
        }

        if num_denoising > 0 and labels is not None:
            outputs["dn_positive_idx"] = denoising_tensors["dn_positive_idx"]
            outputs["dn_num_group"] = denoising_tensors["dn_num_group"]
            outputs["dn_num_split"] = denoising_tensors["dn_num_split"]

        outputs = {k: v for k, v in outputs.items() if v is not None}
        super().__init__(
            inputs=pixel_values,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.decoder_in_channels = decoder_in_channels
        self.encoder_hidden_dim = encoder_hidden_dim
        self.num_labels = num_labels
        self.num_denoising = num_denoising
        self.learn_initial_query = learn_initial_query
        self.num_queries = num_queries
        self.anchor_image_size = anchor_image_size
        self.feat_strides = feat_strides
        self.batch_norm_eps = batch_norm_eps
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.encoder_in_channels = encoder_in_channels
        self.encode_proj_layers = encode_proj_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.eval_size = eval_size
        self.normalize_before = normalize_before
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.encoder_activation_function = encoder_activation_function
        self.activation_dropout = activation_dropout
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.hidden_expansion = hidden_expansion
        self.depth_mult = depth_mult
        self.eval_idx = eval_idx
        self.box_noise_scale = box_noise_scale
        self.label_noise_ratio = label_noise_ratio
        self.decoder_layers = decoder_layers
        self.reg_scale = reg_scale
        self.max_num_bins = max_num_bins
        self.up = up
        self.decoder_attention_heads = decoder_attention_heads
        self.attention_dropout = attention_dropout
        self.decoder_activation_function = decoder_activation_function
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_offset_scale = decoder_offset_scale
        self.decoder_method = decoder_method
        self.decoder_n_points = decoder_n_points
        self.top_prob_values = top_prob_values
        self.lqe_hidden_dim = lqe_hidden_dim
        self.lqe_layers_count = lqe_layers_count
        self.hidden_act = hidden_act
        self.stem_channels = stem_channels
        self.use_learnable_affine_block = use_learnable_affine_block
        self.apply_downsample = apply_downsample
        self.use_lightweight_conv_block = use_lightweight_conv_block
        self.data_format = data_format
        self.layer_scale = layer_scale
        self.initializer_bias_prior_prob = initializer_bias_prior_prob
        self.seed = seed
        self.initializer_range = initializer_range
        self.image_shape = image_shape
        self.hidden_sizes = hidden_sizes
        self.embedding_size = embedding_size
        self.channel_axis = channel_axis
        self.spatial_shapes_list = spatial_shapes_list
        self.stage_names = stage_names
        self.out_features = out_features
        self.depths = depths
        self.initializer = initializer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decoder_in_channels": self.decoder_in_channels,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "num_labels": self.num_labels,
                "num_denoising": self.num_denoising,
                "learn_initial_query": self.learn_initial_query,
                "num_queries": self.num_queries,
                "anchor_image_size": self.anchor_image_size,
                "feat_strides": self.feat_strides,
                "batch_norm_eps": self.batch_norm_eps,
                "num_feature_levels": self.num_feature_levels,
                "hidden_dim": self.hidden_dim,
                "layer_norm_eps": self.layer_norm_eps,
                "encoder_in_channels": self.encoder_in_channels,
                "encode_proj_layers": self.encode_proj_layers,
                "positional_encoding_temperature": self.positional_encoding_temperature,  # noqa: E501
                "eval_size": self.eval_size,
                "normalize_before": self.normalize_before,
                "num_attention_heads": self.num_attention_heads,
                "dropout": self.dropout,
                "encoder_activation_function": self.encoder_activation_function,
                "activation_dropout": self.activation_dropout,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "encoder_layers": self.encoder_layers,
                "hidden_expansion": self.hidden_expansion,
                "depth_mult": self.depth_mult,
                "eval_idx": self.eval_idx,
                "box_noise_scale": self.box_noise_scale,
                "label_noise_ratio": self.label_noise_ratio,
                "decoder_layers": self.decoder_layers,
                "reg_scale": self.reg_scale,
                "max_num_bins": self.max_num_bins,
                "up": self.up,
                "decoder_attention_heads": self.decoder_attention_heads,
                "attention_dropout": self.attention_dropout,
                "decoder_activation_function": self.decoder_activation_function,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "decoder_offset_scale": self.decoder_offset_scale,
                "decoder_method": self.decoder_method,
                "decoder_n_points": self.decoder_n_points,
                "top_prob_values": self.top_prob_values,
                "lqe_hidden_dim": self.lqe_hidden_dim,
                "lqe_layers_count": self.lqe_layers_count,
                "hidden_act": self.hidden_act,
                "stem_channels": self.stem_channels,
                "use_learnable_affine_block": self.use_learnable_affine_block,
                "stackwise_stage_filters": self.stackwise_stage_filters,
                "apply_downsample": self.apply_downsample,
                "use_lightweight_conv_block": self.use_lightweight_conv_block,
                "layer_scale": self.layer_scale,
                "seed": self.seed,
                "depths": self.depths,
                "initializer_bias_prior_prob": (
                    self.initializer_bias_prior_prob
                ),
                "initializer_range": self.initializer_range,
                "hidden_sizes": self.hidden_sizes,
                "embedding_size": self.embedding_size,
                "image_shape": self.image_shape,
                "data_format": self.data_format,
                "out_features": self.out_features,
                "channel_axis": self.channel_axis,
            }
        )
        return config
