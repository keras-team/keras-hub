import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_hub
import numpy as np

print("Running E2E verification tests...")

# 1. Decoder Model test (generate)
print("\n--- Testing Decoder: GemmaCausalLM ---")
try:
    gemma = keras_hub.models.GemmaCausalLM.from_preset(
        "gemma_2b_en", 
        # using a tiny subset for fast testing
    )
    # Run a simple generate call
    output = gemma.generate("The capital of France is", max_length=10)
    print("Gemma generated:", output)
    print("✅ Decoder generate() successful!")
except Exception as e:
    print("⚠️ Error running Gemma:", e)

# 2. Encoder Model test (predict)
print("\n--- Testing Encoder: AlbertClassifier ---")
try:
    albert = keras_hub.models.AlbertClassifier.from_preset(
        "albert_base_en_uncased", num_classes=2
    )
    # Run a simple predict call
    preds = albert.predict(["The capital of France is Paris."])
    print("Albert predictions shape:", preds.shape)
    print("✅ Encoder predict() successful!")
except Exception as e:
    print("⚠️ Error running Albert:", e)

print("\nE2E testing script finished.")
