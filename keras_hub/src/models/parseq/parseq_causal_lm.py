from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_preprocessor import PARSeqPreprocessor
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.ParSeqCausalLM")
class ParSeqCausalLM(CausalLM):
    backbone_cls = PARSeqBackbone
    preprocessor_cls = PARSeqPreprocessor

    def __init__(
        self,
        preprocessor,
        backbone,
        **kwargs,
    ):
        # === Layers ===
        self.preprocessor = preprocessor
        self.backbone = backbone

        # === Functional Model ===
        # This must be "backbone.input" i.e. the full input structure,
        # rather than "backbone.inputs" which is the flattened list of inputs.
        inputs = backbone.input
        outputs = backbone(inputs=inputs)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="greedy",
        **kwargs,
    ):
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            sampler=sampler,
            **kwargs,
        )

    def generate_step(self, inputs, stop_token_ids=None):
        token_ids, padding_mask, images = (
            inputs["token_ids"],
            inputs["padding_mask"],
            inputs["images"],
        )
        images_shape = ops.shape(images)
        if len(images_shape) == 3:
            # Handle an unbatched image. Unlike `token_ids` and `padding_mask`
            # this will not automatically be upranked.
            images = ops.expand_dims(images, axis=0)

        batch_size = images_shape[0]
        num_steps = self.backbone.max_label_length + 1

        memory = self.backbone.image_encoder(images)
        pos_queries = ops.expand_dims(
            self.backbone.decode.pos_queries[:, :num_steps],
            (batch_size, -1, -1),
        )
        target_mask = query_mask = 1 - ops.triu(ops.ones((25, 25)), 1)

        def next(prompt, cache, index):
            target_out = self.backbone.decode(
                target=prompt[:, index + 1],
                memory=memory,
                target_mask=target_mask[: index + 1, : index + 1],
                target_query=pos_queries[:, index : index + 1],
                target_query_mask=query_mask[index : index + 1, : index + 1],
            )
            logits = self.backbone.head(target_out)

            return logits, None, cache

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            index=1,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            model=self,
        )

        # Compute an output padding mask with the token ids we updated.
        if stop_token_ids is not None:
            # Build a mask of `stop_token_ids` locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )

            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "images": images,
        }
