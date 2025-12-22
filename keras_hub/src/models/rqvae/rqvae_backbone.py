import keras
from keras import ops
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.rqvae import rqvae_layers


class RQVAEBackbone(Backbone):
  """Residual Quantized Variational Autoencoder (RQVAE) backbone."""

  def __init__(
      self,
      input_dim,
      encoder_layer_dims,
      output_dim,
      decoder_layer_dims,
      num_embeddings,
      num_quantizers,
      decay=0.99,
      data_variance=1.0,
      commitment_cost=0.25,
      dtype=None,
      **kwargs,
  ):
    # inputs
    inputs = keras.Input(shape=(input_dim,), dtype=dtype)

    # Layers
    encoder = rqvae_layers.Encoder(
        layer_dims=encoder_layer_dims,
        output_dim=output_dim,
        dtype=dtype,
        name="encoder",
    )

    quantizers = []
    for i in range(num_quantizers):
      quantizers.append(
          rqvae_layers.VectorQuantizerEMA(
              num_embeddings=num_embeddings,
              embedding_dim=output_dim,
              decay=decay,
              dtype=dtype,
              name=f"quantizer_{i}",
          )
      )

    residual_quantizer = rqvae_layers.ResidualVectorQuantizer(
        quantizers=quantizers, dtype=dtype, name="residual_quantizer"
    )

    decoder = rqvae_layers.Decoder(
        layer_dims=decoder_layer_dims,
        output_dim=input_dim,
        dtype=dtype,
        name="decoder",
    )

    # Functional Build
    x = encoder(inputs)
    quantized, encodings, usage_ratios, quantization_loss = residual_quantizer(
        x
    )
    reconstructions = decoder(quantized)

    outputs = {
        "reconstructions": reconstructions,
        "encodings": encodings,
        "usage_ratios": usage_ratios,
        "quantization_loss": quantization_loss,
    }

    super().__init__(inputs=inputs, outputs=outputs, dtype=dtype, **kwargs)

    self.encoder = encoder
    self.residual_quantizer = residual_quantizer
    self.decoder = decoder
    self.data_variance = data_variance
    self.commitment_cost = commitment_cost

    # Save config
    self.input_dim = input_dim
    self.encoder_layer_dims = encoder_layer_dims
    self.output_dim = output_dim
    self.decoder_layer_dims = decoder_layer_dims
    self.num_embeddings = num_embeddings
    self.num_quantizers = num_quantizers
    self.decay = decay

  def compute_loss(self, x, y, y_pred, sample_weight=None):
    reconstructions = y_pred["reconstructions"]
    quantization_loss = y_pred["quantization_loss"]

    reconstruction_loss = (
        ops.mean((reconstructions - target) ** 2) / self.data_variance
    )

    loss = reconstruction_loss + self.commitment_cost * quantization_loss
    return loss

  def get_config(self):
    config = super().get_config()
    config.update({
        "input_dim": self.input_dim,
        "encoder_layer_dims": self.encoder_layer_dims,
        "output_dim": self.output_dim,
        "decoder_layer_dims": self.decoder_layer_dims,
        "num_embeddings": self.num_embeddings,
        "num_quantizers": self.num_quantizers,
        "decay": self.decay,
        "data_variance": self.data_variance,
        "commitment_cost": self.commitment_cost,
    })
    return config
