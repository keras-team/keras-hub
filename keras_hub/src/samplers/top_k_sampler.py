from keras import config
from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.samplers.sampler import Sampler


@keras_hub_export("keras_hub.samplers.TopKSampler")
class TopKSampler(Sampler):
    """Top-K Sampler class.

    This sampler implements top-k search algorithm. Briefly, top-k algorithm
    randomly selects a token from the tokens of top K probability, with
    selection chance determined by the probability.

    Args:
        k: int, the `k` value of top-k.
        seed: int. The random seed. Defaults to `None`.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Pass by name to compile.
    causal_lm.compile(sampler="top_k")
    causal_lm.generate(["Keras is a"])

    # Pass by object to compile.
    sampler = keras_hub.samplers.TopKSampler(k=5, temperature=0.7)
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
    ```
    """

    def __init__(
        self,
        k=5,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.seed = seed
        self.seed_generator = random.SeedGenerator(seed)

    def get_next_token(self, probabilities):
        # Fast path for torch backend: use native torch ops to avoid
        # ops dispatch overhead (saves ~2ms per iteration).
        if config.backend() == "torch":
            import torch

            top_k_pred, top_k_indices = torch.topk(
                probabilities, k=self.k, sorted=False
            )
            # torch.multinomial on MPS/CPU with tiny tensors (batch=1, k=5)
            # is much faster on CPU (~0.03ms vs ~1.7ms on MPS).
            # For CUDA, keep on device.
            device = top_k_pred.device
            if device.type == "mps":
                top_k_cpu = top_k_pred.to(device="cpu", dtype=torch.float32)
                sample_indices = torch.multinomial(top_k_cpu, num_samples=1).to(
                    device=device
                )
            else:
                sample_indices = torch.multinomial(
                    top_k_pred.to(dtype=torch.float32), num_samples=1
                )
            # Gather the original token indices.
            output = torch.gather(top_k_indices, 1, sample_indices)
            return output.squeeze(-1)

        # Default path for JAX/TF: use keras ops.
        # Filter out top-k tokens.
        top_k_pred, top_k_indices = ops.top_k(
            probabilities,
            k=self.k,
            sorted=False,
        )
        # Sample the next token from the probability distribution.
        sample_indices = random.categorical(
            # tf does not support half precision multinomial sampling, so make
            # sure we have full precision here.
            ops.cast(ops.log(top_k_pred), "float32"),
            1,
            seed=self.seed_generator,
            dtype="int32",
        )

        # Rearrange to get the next token idx from the original order.
        output = ops.take_along_axis(top_k_indices, sample_indices, axis=-1)
        return ops.squeeze(output, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "seed": self.seed,
            }
        )
        return config
