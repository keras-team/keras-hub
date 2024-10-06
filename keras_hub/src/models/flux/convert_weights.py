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


def convert_mlpembedder_weights(pytorch_model, keras_model):
    pytorch_in_layer_weight = (
        pytorch_model.in_layer.weight.detach().cpu().numpy()
    )
    pytorch_in_layer_bias = pytorch_model.in_layer.bias.detach().cpu().numpy()

    pytorch_out_layer_weight = (
        pytorch_model.out_layer.weight.detach().cpu().numpy()
    )
    pytorch_out_layer_bias = pytorch_model.out_layer.bias.detach().cpu().numpy()

    keras_model.in_layer.set_weights(
        [pytorch_in_layer_weight.T, pytorch_in_layer_bias]
    )
    keras_model.out_layer.set_weights(
        [pytorch_out_layer_weight.T, pytorch_out_layer_bias]
    )


def convert_selfattention_weights(pytorch_model, keras_model):

    pytorch_qkv_weight = pytorch_model.qkv.weight.detach().cpu().numpy()
    pytorch_qkv_bias = (
        pytorch_model.qkv.bias.detach().cpu().numpy()
        if pytorch_model.qkv.bias is not None
        else None
    )

    pytorch_proj_weight = pytorch_model.proj.weight.detach().cpu().numpy()
    pytorch_proj_bias = pytorch_model.proj.bias.detach().cpu().numpy()

    keras_model.qkv.set_weights(
        [pytorch_qkv_weight.T]
        + ([pytorch_qkv_bias] if pytorch_qkv_bias is not None else [])
    )
    keras_model.proj.set_weights([pytorch_proj_weight.T, pytorch_proj_bias])


def convert_modulation_weights(pytorch_model, keras_model):
    pytorch_weight = pytorch_model.lin.weight.detach().cpu().numpy()
    pytorch_bias = pytorch_model.lin.bias.detach().cpu().numpy()

    keras_model.lin.set_weights([pytorch_weight.T, pytorch_bias])


def convert_doublestreamblock_weights(pytorch_model, keras_model):
    # Convert img_mod weights
    convert_modulation_weights(pytorch_model.img_mod, keras_model.img_mod)

    # Convert txt_mod weights
    convert_modulation_weights(pytorch_model.txt_mod, keras_model.txt_mod)

    # Convert img_attn weights
    convert_selfattention_weights(pytorch_model.img_attn, keras_model.img_attn)

    # Convert txt_attn weights
    convert_selfattention_weights(pytorch_model.txt_attn, keras_model.txt_attn)

    # Convert img_mlp weights (2 Linear layers in PyTorch -> 2 Dense layers in Keras)
    keras_model.img_mlp.layers[0].set_weights(
        [
            pytorch_model.img_mlp[0].weight.detach().cpu().numpy().T,
            pytorch_model.img_mlp[0].bias.detach().cpu().numpy(),
        ]
    )
    keras_model.img_mlp.layers[2].set_weights(
        [
            pytorch_model.img_mlp[2].weight.detach().cpu().numpy().T,
            pytorch_model.img_mlp[2].bias.detach().cpu().numpy(),
        ]
    )

    # Convert txt_mlp weights (2 Linear layers in PyTorch -> 2 Dense layers in Keras)
    keras_model.txt_mlp.layers[0].set_weights(
        [
            pytorch_model.txt_mlp[0].weight.detach().cpu().numpy().T,
            pytorch_model.txt_mlp[0].bias.detach().cpu().numpy(),
        ]
    )
    keras_model.txt_mlp.layers[2].set_weights(
        [
            pytorch_model.txt_mlp[2].weight.detach().cpu().numpy().T,
            pytorch_model.txt_mlp[2].bias.detach().cpu().numpy(),
        ]
    )


def convert_singlestreamblock_weights(pytorch_model, keras_model):
    convert_modulation_weights(pytorch_model.modulation, keras_model.modulation)

    # Convert linear1 (Dense) weights
    keras_model.linear1.set_weights(
        [
            pytorch_model.linear1.weight.detach().cpu().numpy().T,
            pytorch_model.linear1.bias.detach().cpu().numpy(),
        ]
    )

    # Convert linear2 (Dense) weights
    keras_model.linear2.set_weights(
        [
            pytorch_model.linear2.weight.detach().cpu().numpy().T,
            pytorch_model.linear2.bias.detach().cpu().numpy(),
        ]
    )


def convert_lastlayer_weights(pytorch_model, keras_model):

    # Convert linear (Dense) weights
    keras_model.linear.set_weights(
        [
            pytorch_model.linear.weight.detach().cpu().numpy().T,
            pytorch_model.linear.bias.detach().cpu().numpy(),
        ]
    )

    # Convert adaLN_modulation (Sequential) weights
    keras_model.adaLN_modulation.layers[1].set_weights(
        [
            pytorch_model.adaLN_modulation[1].weight.detach().cpu().numpy().T,
            pytorch_model.adaLN_modulation[1].bias.detach().cpu().numpy(),
        ]
    )


def convert_flux_weights(pytorch_model, keras_model):
    # Convert img_in (Dense) weights
    keras_model.img_in.set_weights(
        [
            pytorch_model.img_in.weight.detach().cpu().numpy().T,
            pytorch_model.img_in.bias.detach().cpu().numpy(),
        ]
    )

    # Convert time_in (MLPEmbedder) weights
    convert_mlpembedder_weights(pytorch_model.time_in, keras_model.time_in)

    # Convert vector_in (MLPEmbedder) weights
    convert_mlpembedder_weights(pytorch_model.vector_in, keras_model.vector_in)

    # Convert guidance_in (if present)
    if keras_model.guidance_embed:
        convert_mlpembedder_weights(
            pytorch_model.guidance_in, keras_model.guidance_in
        )

    # Convert txt_in (Dense) weights
    keras_model.txt_in.set_weights(
        [
            pytorch_model.txt_in.weight.detach().cpu().numpy().T,
            pytorch_model.txt_in.bias.detach().cpu().numpy(),
        ]
    )

    # Convert double_blocks (DoubleStreamBlock) weights
    for pt_block, keras_block in zip(
        pytorch_model.double_blocks, keras_model.double_blocks
    ):
        convert_doublestreamblock_weights(pt_block, keras_block)

    # Convert single_blocks (SingleStreamBlock) weights
    for pt_block, keras_block in zip(
        pytorch_model.single_blocks, keras_model.single_blocks
    ):
        convert_singlestreamblock_weights(pt_block, keras_block)

    # Convert final_layer (LastLayer) weights
    convert_lastlayer_weights(
        pytorch_model.final_layer, keras_model.final_layer
    )
