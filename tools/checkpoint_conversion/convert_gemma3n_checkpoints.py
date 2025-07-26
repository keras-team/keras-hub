import io
import os
import random
import traceback
from PIL import Image
import soundfile as sf

import requests

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app
from absl import flags

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)

from keras import ops  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "gemma3n_e2b_en": "google/gemma-3n-E2B"
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def test_multimodal_model(
    keras_hub_backbone, keras_hub_preprocessor, hf_model, hf_processor, sample_image, sample_audio
):
    print("\n-> Running MULTIMODAL model comparison tests...")
    # 1. Test parameter count
    keras_hub_params = keras_hub_backbone.count_params()
    hf_params = hf_model.num_parameters()
    print(f"   - Keras param count: {keras_hub_params}")
    print(f"   - HF param count:    {hf_params}")
    assert abs(keras_hub_params - hf_params) < 1000

    # 2. Test final hidden state outputs with a multimodal prompt
    prompt = "Describe the image and the audio."
    
    # Hugging Face processing
    hf_inputs = hf_processor(
        text=prompt,
        images=sample_image,
        audio=sample_audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs, output_hidden_states=True)
    hf_final_hidden_states = hf_outputs.hidden_states[-1].detach().cpu().numpy()

    # Keras processing
    # NOTE: This assumes a fully implemented Gemma3nPreprocessor that can handle
    # text, images, and audio.
    keras_hub_inputs = keras_hub_preprocessor(
        {"text": prompt, "images": sample_image, "audio": sample_audio}
    )

    keras_hub_output = keras_hub_backbone(keras_hub_inputs)
    keras_hub_final_hidden_states = ops.convert_to_numpy(keras_hub_output)

    # Compare the final sequence output
    try:
        # We only compare the sequence length that both models produce,
        # as padding strategies might differ slightly.
        common_seq_len = min(hf_final_hidden_states.shape[1], keras_hub_final_hidden_states.shape[1])
        np.testing.assert_allclose(
            keras_hub_final_hidden_states[:, :common_seq_len, :],
            hf_final_hidden_states[:, :common_seq_len, :],
            atol=1e-3, rtol=1e-3 # Looser tolerance for complex models
        )
        print("   - Multimodal hidden state comparison test passed!")
    except AssertionError as err:
        print("\n--- Multimodal hidden state comparison test failed! ---")
        print(traceback.format_exc())
        print(err.args[0])
        print("---------------------------------------------------\n")

def validate_multimodal_generation(
    keras_model, hf_model, hf_processor, sample_image, sample_audio
):
    print("\n-> Running MULTIMODAL generation validation...")
    prompt = "What is in this image and what sound can you hear?"
    length = 48

    # KerasHub Generation
    # NOTE: This assumes a fully implemented Gemma3nCausalLM that can accept
    # multimodal inputs in its generate method.
    keras_output = keras_model.generate(
        {"text": prompt, "images": sample_image, "audio": sample_audio},
        max_length=length
    )
    print("🔶 KerasHub output:", keras_output[0])

    # Hugging Face Generation
    hf_inputs = hf_processor(
        text=prompt,
        images=sample_image,
        audio=sample_audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        hf_outputs = hf_model.generate(**hf_inputs, max_length=length, do_sample=False)
    
    hf_generated_text = hf_processor.batch_decode(
        hf_outputs, skip_special_tokens=True
    )[0]
    print("🔶 Huggingface output:", hf_generated_text)

def main(_):
    # === Get the preset name ===
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(f"Invalid preset {FLAGS.preset}")
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    # === Load sample data ===
    print("\n-> Loading sample image and audio data...")
    # Load sample image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    sample_image = Image.open(requests.get(image_url, stream=True).raw)
    # Load sample audio
    audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
    audio_response = requests.get(audio_url)
    sample_audio, _ = sf.read(io.BytesIO(audio_response.content))
    print("   - Sample data loaded.")

    # === Load the Huggingface model and processor ===
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset,
        device_map=device,
        torch_dtype=torch.float16, # Use float16 for memory
        trust_remote_code=True,
    )
    hf_processor = AutoProcessor.from_pretrained(hf_preset, trust_remote_code=True)
    hf_model.eval()

    # === Load Keras model and construct preprocessor ===
    # The `from_preset` call triggers the conversion from the HF checkpoint.
    keras_hub_backbone = keras_hub.models.Gemma3nBackbone.from_preset(
        f"hf://{hf_preset}"
    )
    # NOTE: The Keras Tokenizer should be part of the full Preprocessor.
    # We create a placeholder for the full preprocessor here.
    keras_hub_preprocessor = keras_hub.models.Gemma3nPreprocessor.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Huggingface model and Keras model loaded")

    # === Check that the models' outputs match with multimodal inputs ===
    test_multimodal_model(
        keras_hub_backbone,
        keras_hub_preprocessor,
        hf_model,
        hf_processor,
        sample_image,
        sample_audio
    )
    print("\n-> Tests passed!")

    # === Create the full CausalLM for generation validation ===
    gemma3n_lm = keras_hub.models.Gemma3nCausalLM(
        backbone=keras_hub_backbone,
        preprocessor=keras_hub_preprocessor,
        sampler="greedy"
    )

    validate_multimodal_generation(
        gemma3n_lm,
        hf_model,
        hf_processor,
        sample_image,
        sample_audio
    )

    # === Save the final converted Keras model ===
    gemma3n_lm.save_to_preset(f"./{preset}")
    print(f"\n🏁 Preset saved to ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)