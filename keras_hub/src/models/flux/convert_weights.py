def convert_mlpembedder_weights(pytorch_model, keras_model):
    """
    Convert weights from PyTorch MLPEmbedder to Keras MLPEmbedderKeras.
    """
    pytorch_in_layer_weight = (
        pytorch_model.in_layer.weight.detach().cpu().numpy()
    )
    pytorch_in_layer_bias = pytorch_model.in_layer.bias.detach().cpu().numpy()

    pytorch_out_layer_weight = (
        pytorch_model.out_layer.weight.detach().cpu().numpy()
    )
    pytorch_out_layer_bias = pytorch_model.out_layer.bias.detach().cpu().numpy()

    keras_model.in_layer.set_weights(
        [pytorch_in_layer_weight.T, pytorch_in_layer_bias]
    )
    keras_model.out_layer.set_weights(
        [pytorch_out_layer_weight.T, pytorch_out_layer_bias]
    )


def convert_selfattention_weights(pytorch_model, keras_model):
    """
    Convert weights from PyTorch SelfAttention to Keras SelfAttentionKeras.
    """

    # Extract PyTorch weights
    pytorch_qkv_weight = pytorch_model.qkv.weight.detach().cpu().numpy()
    pytorch_qkv_bias = (
        pytorch_model.qkv.bias.detach().cpu().numpy()
        if pytorch_model.qkv.bias is not None
        else None
    )

    pytorch_proj_weight = pytorch_model.proj.weight.detach().cpu().numpy()
    pytorch_proj_bias = pytorch_model.proj.bias.detach().cpu().numpy()

    # Set Keras weights (Dense layers use [weight, bias] format)
    keras_model.qkv.set_weights(
        [pytorch_qkv_weight.T]
        + ([pytorch_qkv_bias] if pytorch_qkv_bias is not None else [])
    )
    keras_model.proj.set_weights([pytorch_proj_weight.T, pytorch_proj_bias])
