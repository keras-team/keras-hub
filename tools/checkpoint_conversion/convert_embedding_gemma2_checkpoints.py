import json
import os
import sys
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from keras import ops
from PIL import Image
from transformers import AutoModel
from transformers import AutoProcessor
from transformers import AutoTokenizer

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_text_embedder import (  # noqa: E501
    EmbeddingGemma2TextEmbedder,
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, "HF repo id or local directory", required=True
)
flags.DEFINE_string("upload_uri", None, "Optional kaggle/hf upload URI")


def _get_hf_tokenizer(preset):
    try:
        return AutoProcessor.from_pretrained(preset).tokenizer
    except Exception:
        return AutoTokenizer.from_pretrained(preset)


def _get_hf_pool(hf_model, hf_inputs):
    with torch.no_grad():
        out = hf_model(**hf_inputs)
        mask = hf_inputs["attention_mask"]
        emb = out[0]
        mask_exp = mask.unsqueeze(-1).expand(emb.size()).float()
        pool = torch.sum(emb * mask_exp, 1) / torch.clamp(
            mask_exp.sum(1), min=1e-9
        )
        return torch.nn.functional.normalize(pool, p=2, dim=1).numpy()


def run_validators(uri):
    print("Loading Keras model...")
    embedder = EmbeddingGemma2TextEmbedder.from_preset(uri, dtype="float32")
    preprocessor = embedder.preprocessor

    print("Loading HF model...")
    hf_model = AutoModel.from_pretrained(FLAGS.preset, dtype=torch.float32)
    hf_model.eval()

    try:
        hf_processor = AutoProcessor.from_pretrained(FLAGS.preset)
    except Exception:
        hf_processor = None
    hf_tokenizer = _get_hf_tokenizer(FLAGS.preset)

    results = []
    failed = False

    def report(name, is_pass, diff=None):
        nonlocal failed
        if not is_pass:
            failed = True
        status = "PASS" if is_pass else "FAIL"
        diff_str = f"{diff:.6e}" if diff is not None else "N/A"
        results.append((name, status, diff_str))
        print(f"[{status}] {name} (diff: {diff_str})")

    # (2) Text embeddings (query, document, + config_sentence_transformers.json)
    try:
        print("Running text embeddings validator...")
        tasks = ["query", "document"]
        try:
            st_name = "config_sentence_transformers.json"
            p = os.path.join(FLAGS.preset, st_name)
            if not os.path.exists(p):
                p = hf_hub_download(FLAGS.preset, st_name)
            with open(p) as f:
                config_st = json.load(f)
                prompts = config_st.get("prompts", {})
                for k in prompts:
                    if k not in tasks:
                        tasks.append(k)
                        break
        except Exception:
            prompts = {}
            print("Skipping third prompt name: file not found or parse failed.")

        for task in tasks:
            base_texts = ["Hello world!", "This is a longer test sentence."]
            prompt_str = prompts.get(task, "")
            texts = [prompt_str + t for t in base_texts]

            prep = preprocessor({"texts": texts})
            keras_out = ops.convert_to_numpy(embedder(prep))

            hf_inputs = hf_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            hf_out = _get_hf_pool(hf_model, hf_inputs)

            diff = np.max(np.abs(keras_out - hf_out))
            mean_diff = np.mean(np.abs(keras_out - hf_out))
            print(f"Text ({task}) max: {diff:.6e}, mean: {mean_diff:.6e}")
            report(f"Text_{task}", diff < 1e-4, diff)

    except Exception:
        traceback.print_exc()
        report("Text", False)

    # (3) Vision
    try:
        if getattr(hf_model, "vision_tower", None) is not None and hf_processor:
            print("Running Vision validator...")
            # One image per batch: batching mixed aspect ratios trips a
            # pre-existing bug in `Gemma4VisionAveragePooling` (not part of
            # this port), and HF treats a flat image list as one sample.
            diff, enc_diff = 0.0, 0.0
            for shape in [(224, 224, 3), (448, 256, 3)]:
                img = np.random.randint(0, 256, shape, dtype=np.uint8)
                prep = preprocessor({"images": img[None]})
                keras_out = ops.convert_to_numpy(embedder(prep))
                hf_inputs = hf_processor(
                    images=[[Image.fromarray(img)]], return_tensors="pt"
                )
                hf_out = _get_hf_pool(hf_model, hf_inputs)
                diff = max(diff, float(np.max(np.abs(keras_out - hf_out))))
                # Encoder-only: HF pixels into the Keras vision encoder,
                # which separates the bicubic-resize difference (documented
                # in `Gemma4ImageConverter`) from the weight mapping.
                with torch.no_grad():
                    vo = hf_model.vision_tower(
                        pixel_values=hf_inputs["pixel_values"],
                        pixel_position_ids=hf_inputs["image_position_ids"],
                    )
                    hf_tokens = hf_model.embed_vision(
                        inputs_embeds=vo.last_hidden_state
                    ).numpy()
                k_tokens = ops.convert_to_numpy(
                    embedder.backbone.vision_encoder(
                        {
                            "pixel_values": hf_inputs["pixel_values"]
                            .numpy()[None]
                            .astype("float32"),
                            "pixel_position_ids": hf_inputs[
                                "image_position_ids"
                            ]
                            .numpy()[None]
                            .astype("int32"),
                        }
                    )
                )[0, 0, : hf_tokens.shape[0]]
                enc_diff = max(
                    enc_diff, float(np.max(np.abs(k_tokens - hf_tokens)))
                )
            print(f"Vision max_abs: {diff:.6e} (end-to-end, incl. resize)")
            print(f"Vision_Encoder max_abs: {enc_diff:.6e} (HF pixels)")
            # Weight mapping is gated at 1e-4 on `Vision_Encoder` (HF pixels).
            # The end-to-end row additionally includes `ops.image.resize`
            # bicubic vs. HF's PIL/torchvision bicubic (see the note in
            # `Gemma4ImageConverter`); measured 6e-4 on google/embeddinggemma-2
            # with the encoder at 3.5e-5, so it is bounded at 1e-3 and labelled.
            report("Vision_resize_limited", diff < 1e-3, diff)
            report("Vision_Encoder", enc_diff < 1e-4, enc_diff)
        else:
            print("Skipping Vision check: no vision tower found.")
    except Exception:
        traceback.print_exc()
        report("Vision", False)

    # (4) Audio
    try:
        if getattr(hf_model, "audio_tower", None) is not None and hf_processor:
            print("Running Audio validator...")
            aud1 = np.random.uniform(-1, 1, (int(0.5 * 16000),))
            aud2 = np.random.uniform(-1, 1, (int(1.3 * 16000),))
            aud1, aud2 = aud1.astype("float32"), aud2.astype("float32")

            prep = preprocessor({"audio": tf.ragged.constant([aud1, aud2])})
            keras_out = ops.convert_to_numpy(embedder(prep))

            hf_inputs = hf_processor(
                audio=[aud1, aud2], sampling_rate=16000, return_tensors="pt"
            )
            hf_out = _get_hf_pool(hf_model, hf_inputs)

            diff = np.max(np.abs(keras_out - hf_out))
            print(f"Audio max_abs: {diff:.6e}")
            report("Audio", diff < 1e-4, diff)
        else:
            print("Skipping Audio check: no audio tower found.")
    except Exception:
        traceback.print_exc()
        report("Audio", False)

    # (5) Video
    try:
        if getattr(hf_model, "vision_tower", None) is not None and hf_processor:
            print("Running Video validator...")
            vid1 = [
                np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                for _ in range(3)
            ]
            vid2 = [
                np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                for _ in range(8)
            ]
            # 8 frames x (soft tokens + 2 markers) must fit; the default 512
            # would truncate the Keras side and compare unequal sequences.
            soft = embedder.backbone.vision_encoder.num_vision_tokens_per_image
            old_seq_len = preprocessor.sequence_length
            preprocessor.sequence_length = 8 * (soft + 2) + 16

            prep = preprocessor({"videos": tf.ragged.constant([vid1, vid2])})
            keras_out = ops.convert_to_numpy(embedder(prep))
            preprocessor.sequence_length = old_seq_len

            vid1_pil = [Image.fromarray(f) for f in vid1]
            vid2_pil = [Image.fromarray(f) for f in vid2]
            hf_inputs = hf_processor(
                videos=[vid1_pil, vid2_pil], return_tensors="pt"
            )
            hf_out = _get_hf_pool(hf_model, hf_inputs)

            diff = np.max(np.abs(keras_out - hf_out))
            print(f"Video max_abs: {diff:.6e} (end-to-end, incl. resize)")
            # Frames pass through the same resize as images; see the
            # `Vision_resize_limited` note above. Measured 4.5e-4 and 1.2e-3
            # on google/embeddinggemma-2 with unseeded random frames, so the
            # labelled bound is 2e-3.
            report("Video_resize_limited", diff < 2e-3, diff)
        else:
            print("Skipping Video check: no vision tower found.")
    except Exception:
        traceback.print_exc()
        report("Video", False)

    # (6) Cosine similarity retrieval pair
    try:
        print("Running Cosine similarity validator...")
        q = "How to bake a cake?"
        doc_match = "Here is a recipe for a chocolate cake."
        doc_unrelated = "The capital of France is Paris."

        prep_q = preprocessor({"texts": [q]})
        prep_d_m = preprocessor({"texts": [doc_match]})
        prep_d_u = preprocessor({"texts": [doc_unrelated]})

        keras_q = ops.convert_to_numpy(embedder(prep_q))[0]
        keras_d_m = ops.convert_to_numpy(embedder(prep_d_m))[0]
        keras_d_u = ops.convert_to_numpy(embedder(prep_d_u))[0]

        k_sim_m = np.dot(keras_q, keras_d_m)
        k_sim_u = np.dot(keras_q, keras_d_u)

        def get_hf_emb(text):
            inp = hf_tokenizer([text], return_tensors="pt")
            return _get_hf_pool(hf_model, inp)[0]

        hf_q = get_hf_emb(q)
        hf_d_m = get_hf_emb(doc_match)
        hf_d_u = get_hf_emb(doc_unrelated)

        h_sim_m = np.dot(hf_q, hf_d_m)
        h_sim_u = np.dot(hf_q, hf_d_u)

        is_pass = (k_sim_m > k_sim_u) == (h_sim_m > h_sim_u)
        report("Cosine_Similarity", is_pass)
    except Exception:
        traceback.print_exc()
        report("Cosine_Similarity", False)

    # (8) Sliding window text
    try:
        print("Running Sliding Window Text validator...")
        cfg = getattr(hf_model.config, "text_config", hf_model.config)
        sw = getattr(cfg, "sliding_window", 4)
        long_text = "word " * (sw * 2)

        preprocessor.sequence_length = max(2 * sw + 16, 64)
        prep = preprocessor({"texts": [long_text]})
        keras_out = ops.convert_to_numpy(embedder(prep))
        print(f"Used Keras seq length: {preprocessor.sequence_length}")

        hf_inputs = hf_tokenizer([long_text], return_tensors="pt")
        hf_out = _get_hf_pool(hf_model, hf_inputs)
        print(f"Used HF seq length: {hf_inputs['input_ids'].shape[1]}")

        diff = np.max(np.abs(keras_out - hf_out))
        print(f"SlidingWindow max_abs: {diff:.6e}")
        report("SlidingWindow", diff < 1e-4, diff)
    except Exception:
        traceback.print_exc()
        report("SlidingWindow", False)

    # (1) Tokenizer ids
    try:
        print("Running Tokenizer IDs validator...")
        strings = ["Hello world!", "This is a longer string.", "query: Hello"]
        is_pass = True

        for s in strings:
            prep_dict = preprocessor(s)
            k_ids = ops.convert_to_numpy(prep_dict["token_ids"]).tolist()
            mask = ops.convert_to_numpy(prep_dict["padding_mask"]).tolist()
            if isinstance(mask[0], list):
                k_ids, mask = k_ids[0], mask[0]
            k_ids = [t for t, m in zip(k_ids, mask) if m]

            hf_ids = hf_tokenizer(s)["input_ids"]

            if k_ids != hf_ids:
                print(f"Mismatch for {s!r}:\nKeras: {k_ids}\nHF:    {hf_ids}")
                is_pass = False
        report("Tokenizer_IDs", is_pass)
    except Exception:
        traceback.print_exc()
        report("Tokenizer_IDs", False)

    # (7) Exact parameter count
    try:
        print("Running Parameter Count validator...")
        hf_params = sum(p.numel() for p in hf_model.parameters())
        # state_dict() == the tensors in the checkpoint: parameters plus
        # persistent buffers (per-layer scalars, clipped-linear bounds).
        hf_total = sum(v.numel() for v in hf_model.state_dict().values())
        keras_params = embedder.backbone.count_params()
        print(
            f"HF params: {hf_params}, HF state_dict total: {hf_total}, "
            f"Keras params: {keras_params}"
        )
        report("Parameter_Count", hf_total == keras_params)
    except Exception:
        traceback.print_exc()
        report("Parameter_Count", False)

    print("\n--- Validator Results ---")
    for name, status, diff in results:
        print(f"{name:25s} | {status:4s} | Diff: {diff}")

    if failed:
        sys.exit(1)
    else:
        print("All validators passed.")


def main(_):
    # Validator inputs are random; seed so the resize-limited rows are
    # reproducible run to run.
    np.random.seed(0)
    preset_name = FLAGS.preset
    uri = preset_name if os.path.exists(preset_name) else "hf://" + preset_name

    print(f"Loading and converting {preset_name}...")
    embedder = EmbeddingGemma2TextEmbedder.from_preset(uri, dtype="float32")

    output_dir = os.path.abspath(f"converted_{preset_name.replace('/', '_')}")
    embedder.save_to_preset(output_dir)
    print(f"Saved to {output_dir}")

    run_validators(output_dir)


if __name__ == "__main__":
    app.run(main)
