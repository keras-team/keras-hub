# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""StableDiffusion3 preset configurations."""

backbone_presets = {
    "stable_diffusion_3_medium": {
        "metadata": {
            "description": (
                "3 billion parameter, including CLIP L and CLIP G text "
                "encoders, MMDiT generative model, and VAE decoder. "
                "Developed by Stability AI."
            ),
            "params": 2952806723,
            "official_name": "StableDiffusion3",
            "path": "stablediffusion3",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/stablediffusion3/keras/stable_diffusion_3_medium/1",
    }
}
