"""Gemma3 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gemma3_1b": {
        "metadata": {
            "description": (
                "1 billion parameter, 26-layer, text-only pretrained "
                "Gemma3 model."
            ),
            "params": 999885952,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_1b/3",
    },
    "gemma3_instruct_1b": {
        "metadata": {
            "description": (
                "1 billion parameter, 26-layer, text-only instruction-tuned "
                "Gemma3 model."
            ),
            "params": 999885952,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_1b/3",
    },
    "gemma3_4b_text": {
        "metadata": {
            "description": (
                "4 billion parameter, 34-layer, text-only pretrained "
                "Gemma3 model."
            ),
            "params": 3880099328,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_4b_text/2",
    },
    "gemma3_instruct_4b_text": {
        "metadata": {
            "description": (
                "4 billion parameter, 34-layer, text-only instruction-tuned "
                "Gemma3 model."
            ),
            "params": 3880099328,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_4b_text/3",
    },
    "gemma3_12b_text": {
        "metadata": {
            "description": (
                "12 billion parameter, 48-layer, text-only pretrained "
                "Gemma3 model."
            ),
            "params": 11765788416,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_12b_text/3",
    },
    "gemma3_instruct_12b_text": {
        "metadata": {
            "description": (
                "12 billion parameter, 48-layer, text-only instruction-tuned "
                "Gemma3 model."
            ),
            "params": 11765788416,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_12b_text/3",
    },
    "gemma3_27b_text": {
        "metadata": {
            "description": (
                "27 billion parameter, 62-layer, text-only pretrained "
                "Gemma3 model."
            ),
            "params": 27009002240,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_27b_text/4",
    },
    "gemma3_instruct_27b_text": {
        "metadata": {
            "description": (
                "27 billion parameter, 62-layer, text-only instruction-tuned "
                "Gemma3 model."
            ),
            "params": 27009002240,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_27b_text/3",
    },
    "gemma3_4b": {
        "metadata": {
            "description": (
                "4 billion parameter, 34-layer, vision+text pretrained "
                "Gemma3 model."
            ),
            "params": 4299915632,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_4b/1",
    },
    "gemma3_instruct_4b": {
        "metadata": {
            "description": (
                "4 billion parameter, 34-layer, vision+text instruction-tuned "
                "Gemma3 model."
            ),
            "params": 4299915632,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_4b/1",
    },
    "gemma3_12b": {
        "metadata": {
            "description": (
                "12 billion parameter, 48-layer, vision+text pretrained "
                "Gemma3 model."
            ),
            "params": 12187079280,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_12b/2",
    },
    "gemma3_instruct_12b": {
        "metadata": {
            "description": (
                "12 billion parameter, 48-layer, vision+text instruction-tuned "
                "Gemma3 model."
            ),
            "params": 12187079280,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_12b/2",
    },
    "gemma3_27b": {
        "metadata": {
            "description": (
                "27 billion parameter, 62-layer, vision+text pretrained "
                "Gemma3 model."
            ),
            "params": 27432062576,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_27b/2",
    },
    "gemma3_instruct_27b": {
        "metadata": {
            "description": (
                "27 billion parameter, 62-layer, vision+text instruction-tuned "
                "Gemma3 model."
            ),
            "params": 27432062576,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_27b/2",
    },
    "gemma3_270m": {
        "metadata": {
            "description": (
                "270-million parameter(170m embedding,100m transformer params) "
                "model, 18-layer, text-only designed for hyper-efficient AI, "
                "particularly for task-specific fine-tuning."
            ),
            "params": 268098176,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_270m/3",
    },
    "gemma3_instruct_270m": {
        "metadata": {
            "description": (
                "270-million parameter(170m embedding,100m transformer params) "
                "model, 18-layer, text-only,instruction-tuned model designed "
                "for hyper-efficient AI, particularly for task-specific "
                "fine-tuning."
            ),
            "params": 268098176,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/gemma3/keras/gemma3_instruct_270m/4",
    },
    "medgemma_4b": {
        "metadata": {
            "description": (
                "A 4 billion parameter model based on Gemma 3. "
                "This model is pre-trained for performance on medical text "
                "and image comprehension and is optimized for medical "
                "applications that involve a text generation component."
            ),
            "params": 4300079472,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/medgemma/keras/medgemma_4b/1",
    },
    "medgemma_instruct_4b": {
        "metadata": {
            "description": (
                "A 4 billion parameter model based on Gemma 3. "
                "This model is instruction-tuned for performance on medical "
                "text and image comprehension and is optimized for medical "
                "applications that involve a text generation component."
            ),
            "params": 4300079472,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/medgemma/keras/medgemma_instruct_4b/1",
    },
    "medgemma_instruct_27b": {
        "metadata": {
            "description": (
                "A 27 billion parameter model based on Gemma 3. "
                "This model is instruction-tuned for performance on medical "
                " text and image comprehension and is optimized for medical "
                "applications that involve a text generation component."
            ),
            "params": 27432406640,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/medgemma/keras/medgemma_instruct_27b/1",
    },
    "medgemma_instruct_27b_text": {
        "metadata": {
            "description": (
                "A 27 billion parameter text-only model based on Gemma 3. "
                "This model is instruction-tuned (No images) for performance "
                "on medical text comprehension and is optimized for medical "
                "applications that involve a text generation component."
            ),
            "params": 27009002240,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/medgemma/keras/medgemma_instruct_27b_text/1",
    },
    "medgemma_1.5_instruct_4b": {
        "metadata": {
            "description": (
                "A 4 billion parameter,Instruct-tuned MedGemma 1.5 4B is an "
                "updated version of the Instruction-tuned MedGemma 4B model."
            ),
            "params": 4300079472,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/medgemma/keras/medgemma_1.5_instruct_4b/1",
    },
    "function_gemma_instruct_270m": {
        "metadata": {
            "description": (
                "A 270M Million parameter text-only model based on Gemma 3. "
                "This model is trained specifically for function calling "
                "improvements."
            ),
            "params": 268098176,
            "path": "gemma3",
        },
        "kaggle_handle": "kaggle://keras/function-gemma/keras/function_gemma_instruct_270m/1",
    },
}
