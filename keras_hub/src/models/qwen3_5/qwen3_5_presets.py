"""Qwen3.5 model preset configurations."""

backbone_presets = {
    "qwen3_5_0.8b_base": {
        "metadata": {
            "description": (
                "Ultra-lightweight foundation model. Ideal for edge "
                "devices and efficient, task-specific fine-tuning. "
                "Supports Text, Multimodal, video processing tasks."
            ),
            "params": 852985920,
            "path": "qwen3_5",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5/keras/qwen3_5_0.8b_base/1",
    },
    "qwen3_5_0.8b": {
        "metadata": {
            "description": (
                "Instruction-tuned ultra-lightweight model. "
                "Best for simple chat and basic NLP tasks on "
                "resource-constrained devices. Supports Text, "
                "Multimodal,video processing tasks."
            ),
            "params": 852985920,
            "path": "qwen3_5",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5/keras/qwen3_5_0.8b/1",
    },
    "qwen3_5_2b_base": {
        "metadata": {
            "description": (
                "Lightweight foundation model. Balances speed "
                "and capability; great for mobile deployment "
                "and domain-specific fine-tuning. Supports "
                "Text, Multimodal, video processing tasks."
            ),
            "params": 2213241664,
            "path": "qwen3_5",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5/keras/qwen3_5_2b_base/1",
    },
    "qwen3_5_2b": {
        "metadata": {
            "description": (
                "Instruction-tuned lightweight model. Optimized "
                "for fast chat applications and general "
                "assistance on consumer hardware. Supports "
                "Text, Multimodal, video processing tasks."
            ),
            "params": 2213241664,
            "path": "qwen3_5",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5/keras/qwen3_5_2b/1",
    },
    "qwen3_5_4b_base": {
        "metadata": {
            "description": (
                "Mid-small foundation model. Offers improved "
                "reasoning and context understanding for "
                "custom fine-tuning tasks."
            ),
            "params": 4539265536,
            "path": "qwen3_5",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5/keras/qwen3_5_4b_base/1",
    },
    "qwen3_5_4b": {
        "metadata": {
            "description": (
                "Instruction-tuned mid-small model. A capable "
                "assistant for general text generation and "
                "conversational tasks on standard GPUs. Supports"
                "Multimodal, video processing tasks."
            ),
            "params": 4539265536,
            "path": "qwen3_5",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5/keras/qwen3_5_4b/1",
    },
    "qwen3_5_9b_base": {
        "metadata": {
            "description": (
                "Mid-sized foundation model. Delivers strong "
                "reasoning, coding, and math baseline "
                "capabilities for advanced fine-tuning. "
                "Supports Multimodal , video processing tasks."
            ),
            "params": 9409813744,
            "path": "qwen3_5",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5/keras/qwen3_5_9b_base/1",
    },
    "qwen3_5_9b": {
        "metadata": {
            "description": (
                "Instruction-tuned mid-sized model. Highly "
                "capable chatbot offering strong logic, coding "
                "assistance, and multi-lingual support. "
                "Supports Multimodal , video processing tasks."
            ),
            "params": 9409813744,
            "path": "qwen3_5",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5/keras/qwen3_5_9b/1",
    },
    "qwen3_5_27b": {
        "metadata": {
            "description": (
                "Instruction-tuned large model. Delivers "
                "high-tier performance for complex reasoning, "
                "coding, and extensive contextual tasks. "
                "Supports Multimodal , video processing tasks."
            ),
            "params": 27356728560,
            "path": "qwen3_5",
        },
        "kaggle_handle": "kaggle://keras/qwen3-5/keras/qwen3_5_27b/1",
    },
}
