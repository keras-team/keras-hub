import keras
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone 
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import Qwen2VLVisionEncoder
from keras_hub.src.models.qwen2_vl.qwen2_vl_projector import Qwen2VLProjector

class Qwen2VLBackbone(Backbone):
    def __init__(
        self,
        vision_encoder,
        projector,
        text_backbone,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_encoder = vision_encoder
        self.projector = projector
        self.text_backbone = text_backbone # This is the standard Qwen (2/2.5) LLM

    def call(self, inputs):
        # inputs is a dict containing "images" and "token_ids"
        images = inputs["images"]
        token_ids = inputs["token_ids"]
        
        # Process Images
        image_features = self.vision_encoder(images)
        
        # Project Images to Text Space
        image_embeddings = self.projector(image_features)
        
        # Process Text (Get embeddings normally)
        text_embeddings = self.text_backbone.token_embedding(token_ids)
        
        # FUSE THEM (Placeholder concatenation)
        combined_embeddings = keras.ops.concatenate([image_embeddings, text_embeddings], axis=1)
        
        # Pass through the LLM
        x = self.text_backbone.transformer_layers(combined_embeddings)
        x = self.text_backbone.layer_norm(x)
        
        return x