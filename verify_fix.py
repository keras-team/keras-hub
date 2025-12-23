"""
Reproduction script for JAX __jax_array__() abstractification error.
This script reproduces the issue when using JAX master branch with Keras Hub.
"""

import os
import sys
import traceback

print("=" * 80)
print("JAX Error Reproduction Script")
print("=" * 80)
print()

# Set JAX backend
os.environ["KERAS_BACKEND"] = "jax"
print(f"KERAS_BACKEND: {os.environ.get('KERAS_BACKEND')}")
print()

try:
    print("Importing keras_hub...")
    import keras_hub

    print("✓ keras_hub imported successfully")
    print()

    print("Importing JAX...")
    import jax

    print(f"✓ JAX version: {jax.__version__}")
    print()

    print("Loading BART model (without weights for faster testing)...")
    model = keras_hub.models.BartSeq2SeqLM.from_preset(
        "bart_base_en", load_weights=False
    )
    print("✓ BART model loaded successfully")
    print()

    print("Attempting to generate text...")
    print("Command: model.generate('what is keras')")
    print()

    result = model.generate("what is keras")
    print("✓ Generation successful!")
    print(f"Generated text: {result}")

except ValueError as e:
    if "__jax_array__()" in str(e):
        print("✓ SUCCESSFULLY REPRODUCED THE JAX ERROR!")
        print()
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print()
        print("Full Traceback:")
        print("-" * 80)
        traceback.print_exc()
        print("-" * 80)
        print()
        print("This is the expected error that occurs with JAX master branch.")
        sys.exit(0)
    else:
        raise

except Exception as e:
    print("✗ UNEXPECTED ERROR:")
    print()
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    print()
    print("Full Traceback:")
    print("-" * 80)
    traceback.print_exc()
    print("-" * 80)
    sys.exit(1)

print()
print("=" * 80)
print("Script completed successfully - NO ERROR OCCURRED")
print("=" * 80)
