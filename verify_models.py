import os
import json
import keras
import keras_hub

# Load credentials from standard ~/.kaggle/kaggle.json
kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
if os.path.exists(kaggle_json_path):
    with open(kaggle_json_path, "r") as f:
        creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
        os.environ["KAGGLE_KEY"] = creds.get("key", "")
    print(" Loaded credentials from ~/.kaggle/kaggle.json")
else:
    print(" Warning: ~/.kaggle/kaggle.json not found.")

# Enable bfloat16 mixed precision to save memory on GPU
keras.mixed_precision.set_global_policy("mixed_bfloat16")


models_to_test = [
    {
        "name": "Gemma",
        "preset": "gemma_1.1_instruct_2b_en",
        "class": keras_hub.models.GemmaCausalLM,
        "prompt": "Keras is a"
    },
    {
        "name": "Mistral",
        "preset": "mistral_instruct_7b_en",
        "class": keras_hub.models.MistralCausalLM,
        "prompt": "Keras is a"
    },
    {
        "name": "Phi-3",
        "preset": "phi3_mini_128k_instruct_en",
        "class": keras_hub.models.Phi3CausalLM,
        "prompt": "<|user|>\nWhat is Keras?<|end|>\n<|assistant|>"
    }
]

for model_info in models_to_test:
    name = model_info["name"]
    preset = model_info["preset"]
    model_cls = model_info["class"]
    prompt = model_info["prompt"]
    
    print(f" Testing model: {name} (Preset: {preset})")
    
    try:
        print("Loading model and tokenizer...")
        model = model_cls.from_preset(preset)
        
        print("Compiling model...")
        model.compile(sampler="greedy")
        
        print("Generating text...")
        output = model.generate(prompt, max_length=30)
        
        print(f" {name} Succeeded! Generated output:")
        print(output)
        
    except Exception as e:
        print(f" {name} Failed with error: {type(e).__name__}: {e}")
        
print("Verification Completed")
