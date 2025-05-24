import keras
import numpy as np
import torch
from keras import ops
from torch import nn


# Define PyTorch RMSNorm layer
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Define Keras RMSNorm layer
class Qwen3LayerNorm(keras.layers.Layer):
    """A normalization layer for Qwen that implements RMS normalization."""

    def __init__(self, head_dim=None, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.epsilon = epsilon

    def build(self, input_shape):
        if self.head_dim:
            dim = self.head_dim
        else:
            dim = input_shape[-1]

        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(dim,),
            initializer="ones",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        x = ops.cast(x, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        return ops.cast(x * self.scale, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


def test_rms_norm_layers_multiple_inputs(
    hidden_size=256,
    eps=1e-6,
    input_shape=(2, 10, 256),
    num_inputs=5,
    atol=1e-4,
    random_seed=42,
):
    """
    Test to compare outputs of PyTorch Qwen3RMSNorm and Keras Qwen3LayerNorm
    for multiple inputs.

    Args:
        hidden_size (int): Dimension of the hidden state.
        eps (float): Epsilon for numerical stability.
        input_shape (tuple): Shape of each input tensor
            (batch, seq_len, hidden_size).
        num_inputs (int): Number of input tensors to test.
        atol (float): Absolute tolerance for maximum difference between outputs.
        random_seed (int): Seed for reproducibility.

    Returns:
        bool: True if all outputs are within atol, False otherwise.
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Initialize PyTorch layer
    pytorch_layer = Qwen3RMSNorm(hidden_size=hidden_size, eps=eps)

    # Initialize Keras layer
    keras_layer = Qwen3LayerNorm(head_dim=hidden_size, epsilon=eps)

    # Build Keras layer to initialize weights
    keras_layer.build(input_shape)

    # Copy PyTorch weights to Keras to ensure identical initialization
    pytorch_weight = pytorch_layer.weight.detach().cpu().numpy()
    keras_layer.scale.assign(pytorch_weight)

    # Verify weights are identical
    keras_scale = keras_layer.scale.numpy()
    weight_diff = np.max(np.abs(pytorch_weight - keras_scale))
    print(f"Max weight difference: {weight_diff}")
    assert weight_diff < 1e-8, "Weights are not identical"

    # Initialize lists to store results
    max_diffs = []
    mean_diffs = []
    all_passed = True

    print(f"\nTesting {num_inputs} inputs with atol={atol}...")
    for i in range(num_inputs):
        # Generate a unique input tensor (use different seeds for diversity)
        np.random.seed(random_seed + i)
        input_data = np.random.randn(*input_shape).astype(np.float32)
        torch_input = torch.tensor(input_data, dtype=torch.float32)
        keras_input = input_data  # Keras accepts NumPy arrays directly

        # Compute outputs
        with torch.no_grad():
            torch_output = pytorch_layer(torch_input).numpy()
        keras_output = keras_layer(keras_input).numpy()

        # Compare outputs
        max_diff = np.max(np.abs(torch_output - keras_output))
        mean_diff = np.mean(np.abs(torch_output - keras_output))
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)

        # Check if this input passes
        input_passed = max_diff < atol
        print(f"\nInput {i + 1}:")
        print(f"  Maximum absolute difference: {max_diff}")
        print(f"  Mean absolute difference: {mean_diff}")
        print(f"  Status: {'Passed' if input_passed else 'Failed'}")

        if not input_passed:
            all_passed = False

    # Summarize results
    print("\nSummary:")
    print(f"Average maximum absolute difference: {np.mean(max_diffs)}")
    print(f"Average mean absolute difference: {np.mean(mean_diffs)}")
    print(f"Maximum of maximum absolute differences: {np.max(max_diffs)}")
    print(f"Minimum of maximum absolute differences: {np.min(max_diffs)}")
    print(f"Number of inputs tested: {num_inputs}")
    print(f"Number of inputs passed: {sum(1 for d in max_diffs if d < atol)}")

    if all_passed:
        print("Overall test passed: All outputs are within atol.")
    else:
        print("Overall test failed: Some outputs differ beyond atol.")

    return all_passed


if __name__ == "__main__":
    # Run the test
    test_passed = test_rms_norm_layers_multiple_inputs(
        hidden_size=256,
        eps=1e-6,
        input_shape=(2, 10, 256),
        num_inputs=5,
        atol=1e-4,
        random_seed=42,
    )
    print("Overall test result:", "Passed" if test_passed else "Failed")
