import math

import keras
import numpy as np

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
from keras_hub.src.utils.keras_utils import standardize_data_format


class DFineDenoisingPreprocessorLayer(keras.layers.Layer):
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

    def __init__(self, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

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
            denoising_bbox_unact, dtype=self.compute_dtype
        )
        attention_mask_tensor = keras.ops.convert_to_tensor(
            attention_mask, dtype=self.compute_dtype
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
        backbone: A `keras.Model` instance that serves as the feature extractor.
            While any `keras.Model` can be used, we highly recommend using a
            `keras_hub.models.HGNetV2Backbone` instance, as this architecture is
            optimized for its outputs. If a custom backbone is provided, it
            must have a `stage_names` attribute, or the `out_features` argument
            for this model must be specified. This requirement helps prevent
            hard-to-debug downstream dimensionality errors.
        decoder_in_channels: list, Channel dimensions of the multi-scale
            features from the hybrid encoder. This should typically be a list
            of `encoder_hidden_dim` repeated for each feature level.
        encoder_hidden_dim: int, Hidden dimension size for the encoder layers.
        num_labels: int, Number of object classes for detection.
        num_denoising: int, Number of denoising queries for contrastive
            denoising training. Set to `0` to disable denoising.
        learn_initial_query: bool, Whether to learn initial query embeddings.
        num_queries: int, Number of object queries for detection.
        anchor_image_size: tuple, Size of the anchor image as `(height, width)`.
        feat_strides: list, List of feature stride values for different pyramid
            levels.
        num_feature_levels: int, Number of feature pyramid levels to use.
        hidden_dim: int, Hidden dimension size for the model.
        encoder_in_channels: list, Channel dimensions of the feature maps from
            the backbone (`HGNetV2Backbone`) that are fed into the hybrid
            encoder.
        encode_proj_layers: list, List specifying projection layer
            configurations.
        num_attention_heads: int, Number of attention heads in encoder layers.
        encoder_ffn_dim: int, Feed-forward network dimension in encoder.
        num_encoder_layers: int, Number of encoder layers.
        hidden_expansion: float, Hidden dimension expansion factor.
        depth_multiplier: float, Depth multiplier for the backbone.
        eval_idx: int, Index for evaluation. Defaults to `-1` for the last
            layer.
        num_decoder_layers: int, Number of decoder layers.
        decoder_attention_heads: int, Number of attention heads in decoder
            layers.
        decoder_ffn_dim: int, Feed-forward network dimension in decoder.
        decoder_method: str, Decoder method. Can be either `"default"` or
            `"discrete"`. Defaults to `"default"`.
        decoder_n_points: list, Number of sampling points for deformable
            attention.
        lqe_hidden_dim: int, Hidden dimension for learned query embedding.
        num_lqe_layers: int, Number of layers in learned query embedding.
        label_noise_ratio: float, Ratio of label noise for denoising
            training. Defaults to `0.5`.
        box_noise_scale: float, Scale factor for box noise in denoising
            training. Defaults to `1.0`.
        labels: list or None, Ground truth labels for denoising training. This
            is passed during model initialization to construct the training
            graph for contrastive denoising. Each element should be a
            dictionary with `"boxes"` (numpy array of shape `[N, 4]` with
            normalized coordinates) and `"labels"` (numpy array of shape `[N]`
            with class indices). Required when `num_denoising > 0`. Defaults to
            `None`.
        seed: int or None, Random seed for reproducibility. Defaults to `None`.
        image_shape: tuple, Shape of input images as `(height, width,
            channels)`. Height and width can be `None` for variable input sizes.
            Defaults to `(None, None, 3)`.
        out_features: list or None, List of feature names to output from
            backbone. If `None`, uses the last `len(decoder_in_channels)`
            features from the backbone's `stage_names`. Defaults to `None`.
        data_format: str, The data format of the image channels. Can be either
            `"channels_first"` or `"channels_last"`. If `None` is specified,
            it will use the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. Defaults to `None`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the base class.

    Example:
    ```python
    import keras
    import numpy as np
    from keras_hub.models import DFineBackbone
    from keras_hub.models import HGNetV2Backbone

    # Example 1: Basic usage without denoising.
    # First, build the `HGNetV2Backbone` instance.
    hgnetv2 = HGNetV2Backbone(
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
        use_learnable_affine_block=True,
        hidden_act="relu",
        image_shape=(None, None, 3),
        out_features=["stage3", "stage4"],
        data_format="channels_last",
    )

    # Then, pass the backbone instance to `DFineBackbone`.
    backbone = DFineBackbone(
        backbone=hgnetv2,
        decoder_in_channels=[128, 128],
        encoder_hidden_dim=128,
        num_denoising=0,  # Disable denoising
        num_labels=80,
        hidden_dim=128,
        learn_initial_query=False,
        num_queries=300,
        anchor_image_size=(256, 256),
        feat_strides=[16, 32],
        num_feature_levels=2,
        encoder_in_channels=[512, 1024],
        encode_proj_layers=[1],
        num_attention_heads=8,
        encoder_ffn_dim=512,
        num_encoder_layers=1,
        hidden_expansion=0.34,
        depth_multiplier=0.5,
        eval_idx=-1,
        num_decoder_layers=3,
        decoder_attention_heads=8,
        decoder_ffn_dim=512,
        decoder_n_points=[6, 6],
        lqe_hidden_dim=64,
        num_lqe_layers=2,
        out_features=["stage3", "stage4"],
        image_shape=(None, None, 3),
        data_format="channels_last",
        seed=0,
    )

    # Prepare input data.
    input_data = keras.random.uniform((2, 256, 256, 3))

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

    # Pass the `HGNetV2Backbone` instance to `DFineBackbone`.
    backbone_with_denoising = DFineBackbone(
        backbone=hgnetv2,
        decoder_in_channels=[128, 128],
        encoder_hidden_dim=128,
        num_denoising=100,  # Enable denoising
        num_labels=80,
        hidden_dim=128,
        learn_initial_query=False,
        num_queries=300,
        anchor_image_size=(256, 256),
        feat_strides=[16, 32],
        num_feature_levels=2,
        encoder_in_channels=[512, 1024],
        encode_proj_layers=[1],
        num_attention_heads=8,
        encoder_ffn_dim=512,
        num_encoder_layers=1,
        hidden_expansion=0.34,
        depth_multiplier=0.5,
        eval_idx=-1,
        num_decoder_layers=3,
        decoder_attention_heads=8,
        decoder_ffn_dim=512,
        decoder_n_points=[6, 6],
        lqe_hidden_dim=64,
        num_lqe_layers=2,
        out_features=["stage3", "stage4"],
        image_shape=(None, None, 3),
        seed=0,
        labels=labels,
    )

    # Forward pass with denoising.
    outputs_with_denoising = backbone_with_denoising(input_data)
    ```
    """

    def __init__(
        self,
        backbone,
        decoder_in_channels,
        encoder_hidden_dim,
        num_labels,
        num_denoising,
        learn_initial_query,
        num_queries,
        anchor_image_size,
        feat_strides,
        num_feature_levels,
        hidden_dim,
        encoder_in_channels,
        encode_proj_layers,
        num_attention_heads,
        encoder_ffn_dim,
        num_encoder_layers,
        hidden_expansion,
        depth_multiplier,
        eval_idx,
        num_decoder_layers,
        decoder_attention_heads,
        decoder_ffn_dim,
        decoder_n_points,
        lqe_hidden_dim,
        num_lqe_layers,
        decoder_method="default",
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
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
        self.backbone = backbone
        # Re-instantiate the backbone if its data_format mismatches the parents.
        if (
            hasattr(self.backbone, "data_format")
            and self.backbone.data_format != data_format
        ):
            backbone_config = self.backbone.get_config()
            backbone_config["data_format"] = data_format
            if (
                "image_shape" in backbone_config
                and backbone_config["image_shape"] is not None
                and len(backbone_config["image_shape"]) == 3
            ):
                backbone_config["image_shape"] = tuple(
                    reversed(backbone_config["image_shape"])
                )
            self.backbone = self.backbone.__class__.from_config(backbone_config)
        spatial_shapes = []
        for s in feat_strides:
            h = anchor_image_size[0] // s
            w = anchor_image_size[1] // s
            spatial_shapes.append((h, w))
        # NOTE: While `HGNetV2Backbone` is handled automatically, `out_features`
        # must be specified for custom backbones. This design choice prevents
        # hard-to-debug dimension mismatches by placing the onus on the user for
        # ensuring compatibility.
        if not hasattr(self.backbone, "stage_names") and out_features is None:
            raise ValueError(
                "`out_features` must be specified when using a custom "
                "backbone that does not have a `stage_names` attribute."
            )
        stage_names = getattr(self.backbone, "stage_names", out_features)
        out_features = (
            out_features
            if out_features is not None
            else stage_names[-len(decoder_in_channels) :]
        )
        initializer = d_fine_kernel_initializer(
            initializer_range=0.01,
        )

        # === Layers ===
        self.encoder = DFineHybridEncoder(
            encoder_in_channels=encoder_in_channels,
            feat_strides=feat_strides,
            encoder_hidden_dim=encoder_hidden_dim,
            encode_proj_layers=encode_proj_layers,
            positional_encoding_temperature=10000,
            eval_size=None,
            normalize_before=False,
            num_attention_heads=num_attention_heads,
            dropout=0.0,
            layer_norm_eps=1e-5,
            encoder_activation_function="gelu",
            activation_dropout=0.0,
            encoder_ffn_dim=encoder_ffn_dim,
            num_encoder_layers=num_encoder_layers,
            batch_norm_eps=1e-5,
            hidden_expansion=hidden_expansion,
            depth_multiplier=depth_multiplier,
            kernel_initializer=initializer,
            bias_initializer="zeros",
            channel_axis=channel_axis,
            data_format=data_format,
            dtype=dtype,
            name="hybrid_encoder",
        )
        self.decoder = DFineDecoder(
            layer_scale=1.0,
            eval_idx=eval_idx,
            num_decoder_layers=num_decoder_layers,
            dropout=0.0,
            hidden_dim=hidden_dim,
            reg_scale=4.0,
            max_num_bins=32,
            upsampling_factor=0.5,
            decoder_attention_heads=decoder_attention_heads,
            attention_dropout=0.0,
            decoder_activation_function="relu",
            activation_dropout=0.0,
            layer_norm_eps=1e-5,
            decoder_ffn_dim=decoder_ffn_dim,
            num_feature_levels=num_feature_levels,
            decoder_offset_scale=0.5,
            decoder_method=decoder_method,
            decoder_n_points=decoder_n_points,
            top_prob_values=4,
            lqe_hidden_dim=lqe_hidden_dim,
            num_lqe_layers=num_lqe_layers,
            num_labels=num_labels,
            spatial_shapes=spatial_shapes,
            dtype=dtype,
            initializer_bias_prior_prob=None,
            num_queries=num_queries,
            name="decoder",
        )
        self.anchor_generator = DFineAnchorGenerator(
            anchor_image_size=anchor_image_size,
            feat_strides=feat_strides,
            data_format=data_format,
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
        num_backbone_outs = len(decoder_in_channels)
        self.encoder_input_proj_layers = []
        for i in range(num_backbone_outs):
            self.encoder_input_proj_layers.append(
                [
                    keras.layers.Conv2D(
                        filters=encoder_hidden_dim,
                        kernel_size=1,
                        use_bias=False,
                        kernel_initializer=initializer,
                        bias_initializer="zeros",
                        data_format=data_format,
                        name=f"encoder_input_proj_conv_{i}",
                        dtype=dtype,
                    ),
                    keras.layers.BatchNormalization(
                        epsilon=1e-5,
                        axis=channel_axis,
                        name=f"encoder_input_proj_bn_{i}",
                        dtype=dtype,
                    ),
                ]
            )
        self.enc_output_layers = [
            keras.layers.Dense(
                hidden_dim,
                name="enc_output_dense",
                dtype=dtype,
            ),
            keras.layers.LayerNormalization(
                epsilon=1e-5,
                name="enc_output_ln",
                dtype=dtype,
            ),
        ]
        prior_prob = 1 / (num_labels + 1)
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
        self.decoder_input_proj_layers = []
        for i in range(num_backbone_outs):
            if hidden_dim == decoder_in_channels[-1]:
                proj_layer = keras.layers.Identity(
                    name=f"decoder_input_proj_identity_{i}",
                    dtype=dtype,
                )
                self.decoder_input_proj_layers.append(proj_layer)
            else:
                self.decoder_input_proj_layers.append(
                    [
                        keras.layers.Conv2D(
                            filters=hidden_dim,
                            kernel_size=1,
                            use_bias=False,
                            kernel_initializer=initializer,
                            bias_initializer="zeros",
                            data_format=data_format,
                            name=f"decoder_input_proj_conv1_{i}",
                            dtype=dtype,
                        ),
                        keras.layers.BatchNormalization(
                            epsilon=1e-5,
                            axis=channel_axis,
                            name=f"decoder_input_proj_bn1_{i}",
                            dtype=dtype,
                        ),
                    ]
                )
        for i in range(num_feature_levels - num_backbone_outs):
            idx = num_backbone_outs + i
            if hidden_dim == decoder_in_channels[-1]:
                proj_layer = keras.layers.Identity(
                    name=f"decoder_input_proj_identity_{idx}",
                    dtype=dtype,
                )
                self.decoder_input_proj_layers.append(proj_layer)
            else:
                self.decoder_input_proj_layers.append(
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
                            dtype=dtype,
                        ),
                        keras.layers.BatchNormalization(
                            epsilon=1e-5,
                            axis=channel_axis,
                            name=f"decoder_input_proj_bn3_{idx}",
                            dtype=dtype,
                        ),
                    ]
                )
        self.dn_split_point = None

        # === Functional Model ===
        pixel_values = keras.Input(
            shape=image_shape, name="pixel_values", dtype="float32"
        )
        feature_maps_output = self.backbone(pixel_values)
        feature_maps = [feature_maps_output[stage] for stage in out_features]
        feature_maps_output_tuple = tuple(feature_maps)
        proj_feats = []
        for level, feature_map in enumerate(feature_maps_output_tuple):
            x = self.encoder_input_proj_layers[level][0](feature_map)
            x = self.encoder_input_proj_layers[level][1](x)
            proj_feats.append(x)
        encoder_outputs = self.encoder(
            inputs_embeds=proj_feats,
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
        sources = []
        # NOTE: Handle both no-op (identity mapping) and an actual projection
        # using Conv2D and BatchNorm with `isinstance(proj, list)`.
        for level, source in enumerate(last_hidden_state):
            proj = self.decoder_input_proj_layers[level]
            if isinstance(proj, list):
                x = proj[0](source)
                x = proj[1](x)
                sources.append(x)
            else:
                sources.append(proj(source))
        if num_feature_levels > len(sources):
            len_sources = len(sources)
            proj = self.decoder_input_proj_layers[len_sources]
            if isinstance(proj, list):
                x = proj[0](last_hidden_state[-1])
                x = proj[1](x)
                sources.append(x)
            else:
                sources.append(proj(last_hidden_state[-1]))
            for i in range(len_sources + 1, num_feature_levels):
                proj = self.decoder_input_proj_layers[i]
                if isinstance(proj, list):
                    x = proj[0](sources[-1])
                    x = proj[1](x)
                    sources.append(x)
                else:
                    sources.append(proj(sources[-1]))
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
            self.dn_split_point = int(denoising_meta_values["dn_num_split"][0])
        else:
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = None, None, None, None

        if num_denoising > 0 and labels is not None:
            denoising_processor = DFineDenoisingPreprocessorLayer(
                name="denoising_processor", dtype=dtype
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
        output_memory = self.enc_output_layers[0](memory)
        output_memory = self.enc_output_layers[1](output_memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory)
        enc_outputs_coord_logits_plus_anchors = (
            enc_outputs_coord_logits + anchors
        )
        init_reference_points, target, enc_topk_logits, enc_topk_bboxes = (
            self.initial_query_reference_generator(
                (
                    enc_outputs_class,
                    enc_outputs_coord_logits_plus_anchors,
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
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim
        self.encoder_in_channels = encoder_in_channels
        self.encode_proj_layers = encode_proj_layers
        self.num_attention_heads = num_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_encoder_layers = num_encoder_layers
        self.hidden_expansion = hidden_expansion
        self.depth_multiplier = depth_multiplier
        self.eval_idx = eval_idx
        self.box_noise_scale = box_noise_scale
        self.labels = labels
        self.label_noise_ratio = label_noise_ratio
        self.num_decoder_layers = num_decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_method = decoder_method
        self.decoder_n_points = decoder_n_points
        self.lqe_hidden_dim = lqe_hidden_dim
        self.num_lqe_layers = num_lqe_layers
        self.data_format = data_format
        self.seed = seed
        self.image_shape = image_shape
        self.channel_axis = channel_axis
        self.spatial_shapes = spatial_shapes
        self.stage_names = stage_names
        self.out_features = out_features
        self.initializer = initializer

    def get_config(self):
        config = super().get_config()
        serializable_labels = None
        if self.labels is not None:
            serializable_labels = []
            for target in self.labels:
                serializable_target = {}
                for key, value in target.items():
                    if hasattr(value, "tolist"):
                        serializable_target[key] = value.tolist()
                    else:
                        serializable_target[key] = value
                serializable_labels.append(serializable_target)
        config.update(
            {
                "backbone": keras.layers.serialize(self.backbone),
                "decoder_in_channels": self.decoder_in_channels,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "num_labels": self.num_labels,
                "num_denoising": self.num_denoising,
                "learn_initial_query": self.learn_initial_query,
                "num_queries": self.num_queries,
                "anchor_image_size": self.anchor_image_size,
                "feat_strides": self.feat_strides,
                "num_feature_levels": self.num_feature_levels,
                "hidden_dim": self.hidden_dim,
                "encoder_in_channels": self.encoder_in_channels,
                "encode_proj_layers": self.encode_proj_layers,
                "num_attention_heads": self.num_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "num_encoder_layers": self.num_encoder_layers,
                "hidden_expansion": self.hidden_expansion,
                "depth_multiplier": self.depth_multiplier,
                "eval_idx": self.eval_idx,
                "box_noise_scale": self.box_noise_scale,
                "label_noise_ratio": self.label_noise_ratio,
                "labels": serializable_labels,
                "num_decoder_layers": self.num_decoder_layers,
                "decoder_attention_heads": self.decoder_attention_heads,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "decoder_method": self.decoder_method,
                "decoder_n_points": self.decoder_n_points,
                "lqe_hidden_dim": self.lqe_hidden_dim,
                "num_lqe_layers": self.num_lqe_layers,
                "seed": self.seed,
                "image_shape": self.image_shape,
                "data_format": self.data_format,
                "out_features": self.out_features,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        if "labels" in config and config["labels"] is not None:
            labels = config["labels"]
            deserialized_labels = []
            for target in labels:
                deserialized_target = {}
                for key, value in target.items():
                    if isinstance(value, list):
                        deserialized_target[key] = np.array(value)
                    else:
                        deserialized_target[key] = value
                deserialized_labels.append(deserialized_target)
            config["labels"] = deserialized_labels
        if "dtype" in config and config["dtype"] is not None:
            dtype_config = config["dtype"]
            if "dtype" not in config["backbone"]["config"]:
                config["backbone"]["config"]["dtype"] = dtype_config
        config["backbone"] = keras.layers.deserialize(
            config["backbone"], custom_objects=custom_objects
        )
        return cls(**config)
