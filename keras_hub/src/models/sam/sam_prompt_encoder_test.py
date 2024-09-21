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
import itertools

import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.sam.sam_prompt_encoder import SAMPromptEncoder
from keras_hub.src.tests.test_case import TestCase


class SAMPromptEncoderTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 128
        self.init_kwargs = {
            "hidden_size": 32,
            "image_embedding_size": (8, 8),
            "input_image_size": (self.image_size, self.image_size),
            "mask_in_channels": 16,
        }
        self.prompt_encoder = SAMPromptEncoder(**self.init_kwargs)

    def get_prompts(self, prompts="all"):
        rng = np.random.default_rng(0)

        prompts_dict = {}

        if "all" in prompts or "points" in prompts:
            prompts_dict["points"] = ops.convert_to_tensor(
                rng.integers(0, 1023, (self.batch_size, 10, 2)), dtype="float32"
            )
            prompts_dict["labels"] = ops.convert_to_tensor(
                1 * (rng.random((self.batch_size, 10)) > 0.5), dtype="int32"
            )

        if "all" in prompts or "boxes" in prompts:
            x1y1 = rng.integers(0, 1022, (self.batch_size, 2))
            x2y2 = rng.integers(x1y1, 1023, (self.batch_size, 2))
            box = np.stack([x1y1, x2y2], axis=1)
            prompts_dict["boxes"] = ops.convert_to_tensor(
                box[:, None, ...], dtype="float32"
            )
        if "all" in prompts or "masks" in prompts:
            prompts_dict["masks"] = ops.convert_to_tensor(
                1.0 * (rng.random((self.batch_size, 1, 32, 32, 1)) > 0.5),
                dtype="float32",
            )

        return prompts_dict

    def test_layer_basics(self):
        inputs = self.get_prompts()
        self.run_layer_test(
            cls=SAMPromptEncoder,
            init_kwargs={
                "hidden_size": 32,
                "image_embedding_size": (8, 8),
                "input_image_size": (self.image_size, self.image_size),
                "mask_in_channels": 16,
            },
            input_data=inputs,
            expected_output_shape={
                "prompt_sparse_embeddings": (2, 12, 32),
                "prompt_dense_embeddings": (2, 8, 8, 32),
                "prompt_dense_positional_embeddings": (
                    2,
                    8,
                    8,
                    32,
                ),
            },
            expected_num_trainable_weights=16,
            expected_num_non_trainable_weights=1,
            expected_num_non_trainable_variables=1,
        )

    def test_prompt_encoder_simple(self):
        outputs = self.prompt_encoder(**self.get_prompts())
        (
            sparse_embeddings,
            dense_embeddings,
            prompt_dense_positional_embeddings,
        ) = (
            outputs["prompt_sparse_embeddings"],
            outputs["prompt_dense_embeddings"],
            outputs["prompt_dense_positional_embeddings"],
        )

        sparse_embeddings = ops.convert_to_numpy(sparse_embeddings)
        dense_embeddings = ops.convert_to_numpy(dense_embeddings)
        prompt_dense_positional_embeddings = ops.convert_to_numpy(
            prompt_dense_positional_embeddings
        )

        self.assertEqual(sparse_embeddings.shape, (self.batch_size, 12, 32))
        self.assertEqual(dense_embeddings.shape, (self.batch_size, 8, 8, 32))
        self.assertEqual(
            prompt_dense_positional_embeddings.shape, (1, 8, 8, 32)
        )

    @parameterized.named_parameters(
        [
            ("_".join(x), x)
            for x in itertools.chain(
                itertools.combinations(["points", "boxes", "masks"], 1),
                itertools.combinations(["points", "boxes", "masks"], 2),
            )
        ]
    )
    def test_prompt_encoder_partial_prompts(self, prompts):
        prompts_dict = self.get_prompts(prompts)
        outputs = self.prompt_encoder(**prompts_dict)
        sparse_embeddings, dense_embeddings = (
            outputs["prompt_sparse_embeddings"],
            outputs["prompt_dense_embeddings"],
        )

        sparse_embeddings_dim = 0
        if "points" in prompts:
            sparse_embeddings_dim += prompts_dict["points"].shape[1]
        if "boxes" in prompts:
            sparse_embeddings_dim += prompts_dict["boxes"].shape[1] * 2
        self.assertAllEqual(
            sparse_embeddings.shape,
            (self.batch_size, sparse_embeddings_dim, 32),
        )
        if "masks" not in prompts:
            no_mask_embed = ops.broadcast_to(
                self.prompt_encoder.no_mask_embed(ops.arange(1)),
                (self.batch_size, 8, 8, 32),
            )
            self.assertAllClose(dense_embeddings, no_mask_embed)
