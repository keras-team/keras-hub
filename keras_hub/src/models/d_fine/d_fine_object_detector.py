import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.non_max_supression import NonMaxSuppression
from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
from keras_hub.src.models.d_fine.d_fine_loss import compute_box_losses
from keras_hub.src.models.d_fine.d_fine_loss import compute_local_losses
from keras_hub.src.models.d_fine.d_fine_loss import compute_vfl_loss
from keras_hub.src.models.d_fine.d_fine_loss import get_cdn_matched_indices
from keras_hub.src.models.d_fine.d_fine_loss import hungarian_matcher
from keras_hub.src.models.d_fine.d_fine_object_detector_preprocessor import (
    DFineObjectDetectorPreprocessor,
)
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.utils.tensor_utils import assert_bounding_box_support


@keras_hub_export("keras_hub.models.DFineObjectDetector")
class DFineObjectDetector(ObjectDetector):
    """D-FINE Object Detector model.

    This class wraps the `DFineBackbone` and adds the final prediction and loss
    computation logic for end-to-end object detection. It is responsible for:
    1.  Defining the functional model that connects the `DFineBackbone` to the
        input layers.
    2.  Implementing the `compute_loss` method, which uses a Hungarian matcher
        to assign predictions to ground truth targets and calculates a weighted
        sum of multiple loss components (classification, bounding box, etc.).
    3.  Post-processing the raw outputs from the backbone into final, decoded
        predictions (boxes, labels, confidence scores) during inference.

    Args:
        backbone: A `keras_hub.models.Backbone` instance, specifically a
            `DFineBackbone`, serving as the feature extractor for the object
            detector.
        num_classes: An integer representing the number of object classes to
            detect.
        bounding_box_format: A string specifying the format of the bounding
            boxes. Defaults to `"yxyx"`. Must be a supported format (e.g.,
            `"yxyx"`, `"xyxy"`).
        preprocessor: Optional. An instance of `DFineObjectDetectorPreprocessor`
            for input data preprocessing.
        matcher_class_cost: A float representing the cost for class mismatch in
            the Hungarian matcher. Defaults to `2.0`.
        matcher_bbox_cost: A float representing the cost for bounding box
            mismatch in the Hungarian matcher. Defaults to `5.0`.
        matcher_ciou_cost: A float representing the cost for complete IoU
            mismatch in the Hungarian matcher. Defaults to `2.0`.
        use_focal_loss: A boolean indicating whether to use focal loss for
            classification. Defaults to `True`.
        matcher_alpha: A float parameter for the focal loss alpha. Defaults to
            `0.25`.
        matcher_gamma: A float parameter for the focal loss gamma. Defaults to
            `2.0`.
        weight_loss_vfl: Weight for the classification loss. Defaults to `1.0`.
        weight_loss_bbox: Weight for the bounding box regression loss. Default
            is `5.0`.
        weight_loss_ciou: Weight for the complete IoU loss. Defaults to
            `2.0`.
        weight_loss_fgl: Weight for the focal grid loss. Defaults to `0.15`.
        weight_loss_ddf: Weight for the DDF loss. Defaults to `1.5`.
        ddf_temperature: A float temperature scaling factor for the DDF loss.
            Defaults to `5.0`.
        prediction_decoder: Optional. A `keras.layers.Layer` instance that
            decodes raw predictions. If not provided, a `NonMaxSuppression`
            layer is used.
        activation: Optional. The activation function to apply to the logits
            before decoding. Defaults to `None`.

    Examples:

    **Creating a DFineObjectDetector without labels:**

    ```python
    import numpy as np
    from keras_hub.src.models.d_fine.d_fine_object_detector import (
        DFineObjectDetector
    )
    from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
    from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone

    # Initialize the backbone without labels.
    hgnetv2_backbone = HGNetV2Backbone(
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
        image_shape=(256, 256, 3),
        out_features=["stage3", "stage4"],
    )

    # Initialize the backbone without labels.
    backbone = DFineBackbone(
        backbone=hgnetv2_backbone,
        decoder_in_channels=[128, 128],
        encoder_hidden_dim=128,
        num_denoising=100,
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
        image_shape=(256, 256, 3),
    )

    # Create the detector.
    detector = DFineObjectDetector(
        backbone=backbone,
        num_classes=80,
        bounding_box_format="yxyx",
    )
    ```

    **Creating a DFineObjectDetector with labels for the backbone:**

    ```python
    import numpy as np
    from keras_hub.src.models.d_fine.d_fine_object_detector import (
        DFineObjectDetector
    )
    from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
    from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone

    # Define labels for the backbone.
    labels = [
        {
            "boxes": np.array([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
            "labels": np.array([1, 10])
        },
        {"boxes": np.array([[0.6, 0.6, 0.3, 0.3]]), "labels": np.array([20])},
    ]

    hgnetv2_backbone = HGNetV2Backbone(
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
        image_shape=(256, 256, 3),
        out_features=["stage3", "stage4"],
    )

    # Backbone is initialized with labels.
    backbone = DFineBackbone(
        backbone=hgnetv2_backbone,
        decoder_in_channels=[128, 128],
        encoder_hidden_dim=128,
        num_denoising=100,
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
        image_shape=(256, 256, 3),
        labels=labels,
        box_noise_scale=1.0,
        label_noise_ratio=0.5,
    )

    # Create the detector.
    detector = DFineObjectDetector(
        backbone=backbone,
        num_classes=80,
        bounding_box_format="yxyx",
    )
    ```

    **Using the detector for training:**

    ```python
    import numpy as np
    from keras_hub.src.models.d_fine.d_fine_object_detector import (
        DFineObjectDetector
    )
    from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
    from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone

    # Initialize backbone and detector.
    hgnetv2_backbone = HGNetV2Backbone(
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
        image_shape=(256, 256, 3),
        out_features=["stage3", "stage4"],
    )
    backbone = DFineBackbone(
        backbone=hgnetv2_backbone,
        decoder_in_channels=[128, 128],
        encoder_hidden_dim=128,
        num_denoising=100,
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
        image_shape=(256, 256, 3),
    )
    detector = DFineObjectDetector(
        backbone=backbone,
        num_classes=80,
        bounding_box_format="yxyx",
    )

    # Sample training data.
    images = np.random.uniform(
        low=0, high=255, size=(2, 256, 256, 3)
    ).astype("float32")
    bounding_boxes = {
        "boxes": [
            np.array([[10.0, 20.0, 20.0, 30.0], [20.0, 30.0, 30.0, 40.0]]),
            np.array([[15.0, 25.0, 25.0, 35.0]]),
        ],
        "labels": [
            np.array([0, 2]), np.array([1])
        ],
    }

    # Compile the model.
    detector.compile(
        optimizer="adam",
        loss=detector.compute_loss,
    )

    # Train the model.
    detector.fit(x=images, y=bounding_boxes, epochs=1, batch_size=1)
    ```

    **Making predictions:**

    ```python
    import numpy as np
    from keras_hub.src.models.d_fine.d_fine_object_detector import (
        DFineObjectDetector
    )
    from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
    from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone

    # Initialize backbone and detector.
    hgnetv2_backbone = HGNetV2Backbone(
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
        image_shape=(256, 256, 3),
        out_features=["stage3", "stage4"],
    )
    backbone = DFineBackbone(
        backbone=hgnetv2_backbone,
        decoder_in_channels=[128, 128],
        encoder_hidden_dim=128,
        num_denoising=100,
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
        image_shape=(256, 256, 3),
    )
    detector = DFineObjectDetector(
        backbone=backbone,
        num_classes=80,
        bounding_box_format="yxyx",
    )

    # Sample test image.
    test_image = np.random.uniform(
        low=0, high=255, size=(1, 256, 256, 3)
    ).astype("float32")

    # Make predictions.
    predictions = detector.predict(test_image)

    # Access predictions.
    boxes = predictions["boxes"]                    # Shape: (1, 100, 4)
    labels = predictions["labels"]                  # Shape: (1, 100)
    confidence = predictions["confidence"]          # Shape: (1, 100)
    num_detections = predictions["num_detections"]  # Shape: (1,)
    ```
    """

    backbone_cls = DFineBackbone
    preprocessor_cls = DFineObjectDetectorPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format="yxyx",
        preprocessor=None,
        matcher_class_cost=2.0,
        matcher_bbox_cost=5.0,
        matcher_ciou_cost=2.0,
        use_focal_loss=True,
        matcher_alpha=0.25,
        matcher_gamma=2.0,
        weight_loss_vfl=1.0,
        weight_loss_bbox=5.0,
        weight_loss_ciou=2.0,
        weight_loss_fgl=0.15,
        weight_loss_ddf=1.5,
        ddf_temperature=5.0,
        prediction_decoder=None,
        activation=None,
        **kwargs,
    ):
        assert_bounding_box_support(self.__class__.__name__)

        # === Functional Model ===
        image_input = keras.layers.Input(
            shape=backbone.image_shape, name="images"
        )
        outputs = backbone(image_input)
        intermediate_logits = outputs["intermediate_logits"]
        intermediate_reference_points = outputs["intermediate_reference_points"]
        intermediate_predicted_corners = outputs[
            "intermediate_predicted_corners"
        ]
        initial_reference_points = outputs["initial_reference_points"]
        logits = intermediate_logits[:, -1, :, :]
        pred_boxes = intermediate_reference_points[:, -1, :, :]
        model_outputs = {
            "logits": logits,
            "pred_boxes": pred_boxes,
            "intermediate_logits": intermediate_logits,
            "intermediate_reference_points": intermediate_reference_points,
            "intermediate_predicted_corners": intermediate_predicted_corners,
            "initial_reference_points": initial_reference_points,
            "enc_topk_logits": outputs["enc_topk_logits"],
            "enc_topk_bboxes": outputs["enc_topk_bboxes"],
        }
        if "dn_num_group" in outputs:
            model_outputs["dn_positive_idx"] = outputs["dn_positive_idx"]
            model_outputs["dn_num_group"] = outputs["dn_num_group"]
            model_outputs["dn_num_split"] = outputs["dn_num_split"]
        super().__init__(
            inputs=image_input,
            outputs=model_outputs,
            **kwargs,
        )

        # === Config ===
        self.backbone = backbone
        self.num_classes = num_classes
        self.bounding_box_format = bounding_box_format
        self.preprocessor = preprocessor
        self.matcher_class_cost = matcher_class_cost
        self.matcher_bbox_cost = matcher_bbox_cost
        self.matcher_ciou_cost = matcher_ciou_cost
        self.use_focal_loss = use_focal_loss
        self.matcher_alpha = matcher_alpha
        self.matcher_gamma = matcher_gamma
        self.weight_dict = {
            "loss_vfl": weight_loss_vfl,
            "loss_bbox": weight_loss_bbox,
            "loss_ciou": weight_loss_ciou,
            "loss_fgl": weight_loss_fgl,
            "loss_ddf": weight_loss_ddf,
        }
        self.ddf_temperature = ddf_temperature
        self.activation = activation
        self._prediction_decoder = prediction_decoder or NonMaxSuppression(
            from_logits=(self.activation != keras.activations.sigmoid),
            bounding_box_format=self.bounding_box_format,
            max_detections=backbone.num_queries,
        )

    def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):
        gt_boxes = y["boxes"]
        gt_labels = y["labels"]
        batch_size = keras.ops.shape(gt_labels)[0]
        num_objects = keras.ops.shape(gt_labels)[1]
        num_targets_per_image = keras.ops.tile(
            keras.ops.expand_dims(num_objects, 0), [batch_size]
        )
        labels_for_item = keras.ops.reshape(gt_labels, [-1])
        boxes_for_item = keras.ops.reshape(gt_boxes, [-1, 4])
        targets = {"labels": labels_for_item, "boxes": boxes_for_item}

        intermediate_logits_all = y_pred["intermediate_logits"]
        intermediate_ref_points_all = y_pred["intermediate_reference_points"]
        predicted_corners_all = y_pred["intermediate_predicted_corners"]
        initial_ref_points_all = y_pred["initial_reference_points"]
        enc_topk_logits = y_pred["enc_topk_logits"]
        enc_topk_bboxes = y_pred["enc_topk_bboxes"]
        if "dn_num_group" in y_pred:
            denoising_meta_values = {
                "dn_positive_idx": y_pred["dn_positive_idx"],
                "dn_num_group": y_pred["dn_num_group"],
                "dn_num_split": y_pred["dn_num_split"],
            }
            dn_split_point = self.backbone.dn_split_point
            (
                dn_intermediate_logits,
                matching_intermediate_logits,
            ) = keras.ops.split(
                intermediate_logits_all, [dn_split_point], axis=2
            )
            (
                dn_intermediate_ref_points,
                matching_intermediate_ref_points,
            ) = keras.ops.split(
                intermediate_ref_points_all, [dn_split_point], axis=2
            )
            (
                dn_predicted_corners,
                matching_predicted_corners,
            ) = keras.ops.split(predicted_corners_all, [dn_split_point], axis=2)
            (
                dn_initial_ref_points,
                matching_initial_ref_points,
            ) = keras.ops.split(
                initial_ref_points_all, [dn_split_point], axis=2
            )
        else:
            denoising_meta_values = None
            matching_intermediate_logits = intermediate_logits_all
            matching_intermediate_ref_points = intermediate_ref_points_all
            matching_predicted_corners = predicted_corners_all
            matching_initial_ref_points = initial_ref_points_all
        matching_logits = matching_intermediate_logits[:, -1, :, :]
        matching_pred_boxes = matching_intermediate_ref_points[:, -1, :, :]
        outputs_without_aux = {
            "logits": matching_logits,
            "pred_boxes": keras.ops.clip(matching_pred_boxes, 0, 1),
        }
        indices = hungarian_matcher(
            outputs_without_aux,
            [targets],
            num_targets_per_image,
            self.use_focal_loss,
            self.matcher_alpha,
            self.matcher_gamma,
            self.matcher_bbox_cost,
            self.matcher_class_cost,
            self.matcher_ciou_cost,
            self.backbone,
        )
        num_boxes = keras.ops.shape(labels_for_item)[0]
        num_boxes = keras.ops.convert_to_tensor(num_boxes, dtype="float32")
        num_boxes = keras.ops.maximum(num_boxes, 1.0)
        losses = {}
        vfl_loss = compute_vfl_loss(
            outputs_without_aux,
            [targets],
            indices,
            num_boxes,
            self.num_classes,
            self.matcher_alpha,
            self.matcher_gamma,
        )
        losses.update(
            {
                k: v * self.weight_dict[k]
                for k, v in vfl_loss.items()
                if k in self.weight_dict
            }
        )
        box_losses = compute_box_losses(
            outputs_without_aux, [targets], indices, num_boxes
        )
        losses.update(
            {
                k: v * self.weight_dict[k]
                for k, v in box_losses.items()
                if k in self.weight_dict
            }
        )
        local_losses = compute_local_losses(
            {
                **outputs_without_aux,
                "pred_corners": matching_predicted_corners[:, -1, :, :],
                "ref_points": matching_initial_ref_points[:, -1, :, :],
                "teacher_corners": keras.ops.zeros_like(
                    matching_predicted_corners[:, -1, :, :]
                ),
                "teacher_logits": keras.ops.zeros_like(matching_logits),
            },
            [targets],
            indices,
            num_boxes,
            self.backbone,
            self.ddf_temperature,
            compute_ddf=False,
        )
        losses.update(
            {
                k: v * self.weight_dict[k]
                for k, v in local_losses.items()
                if k in self.weight_dict
            }
        )

        num_aux_layers = self.backbone.num_decoder_layers
        auxiliary_outputs_list = [
            {
                "logits": matching_intermediate_logits[:, i, :, :],
                "pred_boxes": keras.ops.clip(
                    matching_intermediate_ref_points[:, i, :, :], 0, 1
                ),
                "pred_corners": matching_predicted_corners[:, i, :, :],
                "ref_points": matching_initial_ref_points[:, i, :, :],
                "teacher_corners": matching_predicted_corners[:, -1, :, :],
                "teacher_logits": matching_intermediate_logits[:, -1, :, :],
            }
            for i in range(num_aux_layers)
        ]
        for i, aux_output in enumerate(auxiliary_outputs_list):
            aux_indices = hungarian_matcher(
                aux_output,
                [targets],
                num_targets_per_image,
                self.use_focal_loss,
                self.matcher_alpha,
                self.matcher_gamma,
                self.matcher_bbox_cost,
                self.matcher_class_cost,
                self.matcher_ciou_cost,
                self.backbone,
            )
            aux_vfl_loss = compute_vfl_loss(
                aux_output,
                [targets],
                aux_indices,
                num_boxes,
                self.num_classes,
                self.matcher_alpha,
                self.matcher_gamma,
            )
            aux_box_losses = compute_box_losses(
                aux_output, [targets], aux_indices, num_boxes
            )
            is_not_last_aux_layer = i < len(auxiliary_outputs_list) - 1
            aux_local_losses = compute_local_losses(
                aux_output,
                [targets],
                aux_indices,
                num_boxes,
                self.backbone,
                self.ddf_temperature,
                compute_ddf=is_not_last_aux_layer,
            )
            aux_losses = {**aux_vfl_loss, **aux_box_losses, **aux_local_losses}
            weighted_aux_losses = {
                k + f"_aux_{i}": v * self.weight_dict[k]
                for k, v in aux_losses.items()
                if k in self.weight_dict
            }
            losses.update(weighted_aux_losses)
        # Add encoder loss.
        enc_output = {
            "logits": enc_topk_logits,
            "pred_boxes": keras.ops.clip(enc_topk_bboxes, 0, 1),
        }
        enc_indices = hungarian_matcher(
            enc_output,
            [targets],
            num_targets_per_image,
            self.use_focal_loss,
            self.matcher_alpha,
            self.matcher_gamma,
            self.matcher_bbox_cost,
            self.matcher_class_cost,
            self.matcher_ciou_cost,
            self.backbone,
        )
        enc_vfl_loss = compute_vfl_loss(
            enc_output,
            [targets],
            enc_indices,
            num_boxes,
            self.num_classes,
            self.matcher_alpha,
            self.matcher_gamma,
        )
        enc_box_losses = compute_box_losses(
            enc_output, [targets], enc_indices, num_boxes
        )
        enc_losses = {**enc_vfl_loss, **enc_box_losses}
        weighted_enc_losses = {
            k + "_enc": v * self.weight_dict[k]
            for k, v in enc_losses.items()
            if k in self.weight_dict
        }
        losses.update(weighted_enc_losses)

        if denoising_meta_values is not None:
            dn_indices = get_cdn_matched_indices(denoising_meta_values)
            dn_num_group = denoising_meta_values["dn_num_group"][0]
            num_boxes_dn = num_boxes * keras.ops.cast(dn_num_group, "float32")
            num_dn_layers = self.backbone.num_decoder_layers + 1
            for i in range(num_dn_layers):
                is_not_last_layer = keras.ops.less(i, num_dn_layers - 1)
                teacher_idx = num_dn_layers - 1
                dn_aux_output = {
                    "logits": dn_intermediate_logits[:, i, :, :],
                    "pred_boxes": keras.ops.clip(
                        dn_intermediate_ref_points[:, i, :, :], 0, 1
                    ),
                    "pred_corners": dn_predicted_corners[:, i, :, :],
                    "ref_points": dn_initial_ref_points[:, i, :, :],
                    "teacher_corners": dn_predicted_corners[
                        :, teacher_idx, :, :
                    ],
                    "teacher_logits": dn_intermediate_logits[
                        :, teacher_idx, :, :
                    ],
                }
                vfl_loss = compute_vfl_loss(
                    dn_aux_output,
                    [targets],
                    dn_indices,
                    num_boxes_dn,
                    self.num_classes,
                    self.matcher_alpha,
                    self.matcher_gamma,
                )
                box_losses = compute_box_losses(
                    dn_aux_output, [targets], dn_indices, num_boxes_dn
                )
                local_losses = compute_local_losses(
                    dn_aux_output,
                    [targets],
                    dn_indices,
                    num_boxes_dn,
                    self.backbone,
                    self.ddf_temperature,
                    compute_ddf=is_not_last_layer,
                )
                all_losses = {**vfl_loss, **box_losses, **local_losses}
                weighted_losses = {
                    k + f"_dn_{i}": v * self.weight_dict[k]
                    for k, v in all_losses.items()
                    if k in self.weight_dict
                }
                losses.update(weighted_losses)
        total_loss = keras.ops.sum([v for v in losses.values()])
        return total_loss

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        if prediction_decoder.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "Expected `prediction_decoder` and `DFineObjectDetector` to "
                "use the same `bounding_box_format`, but got "
                "`prediction_decoder.bounding_box_format="
                f"{prediction_decoder.bounding_box_format}`, and "
                "`self.bounding_box_format="
                f"{self.bounding_box_format}`."
            )
        self._prediction_decoder = prediction_decoder
        self.make_predict_function(force=True)
        self.make_train_function(force=True)
        self.make_test_function(force=True)

    def decode_predictions(self, predictions, data):
        """Decodes raw model predictions into final bounding boxes.

        This method takes the raw output from the model (logits and normalized
        bounding boxes in center format) and converts them into the final
        detection format. The process involves:
        1.  Denormalizing the bounding box coordinates to the original image
            dimensions.
        2.  Converting boxes from center format `(cx, cy, w, h)` to corner
            format `(ymin, xmin, ymax, xmax)`.
        3.  Applying non-maximum suppression to filter out overlapping boxes
            and keep only the most confident detections.

        Args:
            predictions: dict, A dictionary of tensors from the model,
                containing `"logits"` and `"pred_boxes"`.
            data: tuple, The input data tuple, from which the original images
                are extracted to obtain their dimensions for denormalization.

        Returns:
            Dictionary: Final predictions, containing `"boxes"`, `"labels"`,
            `"confidence"`, and `"num_detections"`.
        """
        if isinstance(data, (list, tuple)):
            images, _ = data
        else:
            images = data
        logits = predictions["logits"]
        pred_boxes = predictions["pred_boxes"]
        height, width, _ = keras.ops.shape(images)[1:]
        denormalized_boxes = keras.ops.stack(
            [
                pred_boxes[..., 0] * width,  # center_x
                pred_boxes[..., 1] * height,  # center_y
                pred_boxes[..., 2] * width,  # width
                pred_boxes[..., 3] * height,  # height
            ],
            axis=-1,
        )
        pred_boxes_xyxy = keras.utils.bounding_boxes.convert_format(
            denormalized_boxes,
            source="center_xywh",
            target="xyxy",
        )
        pred_boxes_yxyx = keras.ops.stack(
            [
                pred_boxes_xyxy[..., 1],  # y_min
                pred_boxes_xyxy[..., 0],  # x_min
                pred_boxes_xyxy[..., 3],  # y_max
                pred_boxes_xyxy[..., 2],  # x_max
            ],
            axis=-1,
        )
        y_pred = self.prediction_decoder(pred_boxes_yxyx, logits, images=images)
        return y_pred

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "bounding_box_format": self.bounding_box_format,
                "matcher_class_cost": self.matcher_class_cost,
                "matcher_bbox_cost": self.matcher_bbox_cost,
                "matcher_ciou_cost": self.matcher_ciou_cost,
                "use_focal_loss": self.use_focal_loss,
                "matcher_alpha": self.matcher_alpha,
                "matcher_gamma": self.matcher_gamma,
                "weight_loss_vfl": self.weight_dict["loss_vfl"],
                "weight_loss_bbox": self.weight_dict["loss_bbox"],
                "weight_loss_ciou": self.weight_dict["loss_ciou"],
                "weight_loss_fgl": self.weight_dict["loss_fgl"],
                "weight_loss_ddf": self.weight_dict["loss_ddf"],
                "ddf_temperature": self.ddf_temperature,
                "prediction_decoder": keras.saving.serialize_keras_object(
                    self._prediction_decoder
                ),
            }
        )
        return config

    def predict_step(self, *args):
        outputs = super().predict_step(*args)
        if isinstance(outputs, tuple):
            return self.decode_predictions(outputs[0], args[-1]), outputs[1]
        return self.decode_predictions(outputs, *args)

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.saving.deserialize_keras_object(
                config["backbone"]
            )
        if "preprocessor" in config and isinstance(
            config["preprocessor"], dict
        ):
            config["preprocessor"] = keras.saving.deserialize_keras_object(
                config["preprocessor"]
            )
        if "prediction_decoder" in config and isinstance(
            config["prediction_decoder"], dict
        ):
            config["prediction_decoder"] = (
                keras.saving.deserialize_keras_object(
                    config["prediction_decoder"]
                )
            )
        return cls(**config)
