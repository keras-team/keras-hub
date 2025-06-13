import gc
import os
import random
import sys
import warnings

import numpy as np
from jax import Array
from rich import print
from utils import keras_print

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

# TensorFlow module


# More suppressed warnings from TensorFlow
# tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(0)
# tf.get_logger().setLevel(logging.ERROR)

### Keras module
import keras

### Pytorch (for converting pytorch weights)
import torch as torch

## Text encoder
from clipEncoder import CLIPTextTransformer  # SD 1.4/1.5

## Tokenizer
from clipTokenizer import LegacySimpleTokenizer
from clipTokenizer import SimpleTokenizer

## ControlNet
from controlNetDiffusionModels import (
    ControlNetDiffusionModel as ControlNetModel,
)
from controlNetDiffusionModels import DiffusionModel as ControlDiffusionModel

### Models from Modules
## VAE, encode and decode
from EncodeDecode import Decoder
from EncodeDecode import ImageEncoder

# from autoencoderKl import Decoder, Encoder
## Diffusion
from kerasCVDiffusionModels import DiffusionModel
from kerasCVDiffusionModels import DiffusionModelV2
from openClipEncoder import OpenCLIPTextTransformer  # SD 2.x

### Modules for image building
from PIL import Image

### Safetensors (for converting safetensor weights)
from safetensors.torch import load_file

### Sampler modules
from samplers import DPMSolverKerasCV as DPMSolver
from samplers.basicSampler import BasicSampler

### Tools
from tools import textEmbeddings as textEmbeddingTools

# import cv2  # OpenCV


### Global Variables
MAX_TEXT_LEN = 77
from constants import _ALPHAS_CUMPROD
from constants import PYTORCH_CKPT_MAPPING

### Main Class


class StableDiffusion:
    ### Base class/object for Stable Diffusion
    def __init__(
        self,
        imageHeight=512,
        imageWidth=512,
        jit_compile=False,
        weights=None,
        legacy=True,
        VAE="Original",
        textEmbeddings=None,
        mixedPrecision=False,
        optimizer="nadam",
        device=None,
        controlNet=[
            False,
            None,
        ],  # [0] = Use ControlNet? [1] = ControlNet Weights [2] = Input [3] = Strength
    ):
        self.device = device

        with keras.device(self.device):
            ### Step 1: Establish image dimensions for UNet ###
            ## requires multiples of 2**7, 2 to the power of 7
            self.imageHeight = round(imageHeight / 128) * 128
            self.imageWidth = round(imageWidth / 128) * 128

            # Global policy
            self.dtype = keras.config.floatx()  # Default

            # Maaaybe float16 will result in faster images?
            if mixedPrecision is True:
                self.change_policy("mixed_float16")

            ### Step 2: Load Text Embeddings ###
            textEmbeddingTokens = []
            if textEmbeddings == None:
                keras_print("\nIgnoring Text Embeddings")
                self.textEmbeddings = None
                self.textEmbeddingsTokens = None
            else:
                keras_print("\nUsing Text Embeddings")
                self.textEmbeddings, self.textEmbeddingsTokens = (
                    textEmbeddingTools.load_text_embedding(textEmbeddings)
                )

            ### Step 3: Which version of Stable Diffusion ###

            self.legacy = legacy

            ### Step 4: Create Tokenizer ###
            if self.legacy is True:
                if self.textEmbeddings is None:
                    # If no textEmbeddings were given, we're not adding to the special tokens list in the tokenizer
                    self.tokenizer = LegacySimpleTokenizer()
                else:
                    self.tokenizer = LegacySimpleTokenizer(
                        specialTokens=self.textEmbeddingsTokens
                    )
            else:
                if self.textEmbeddings is None:
                    self.tokenizer = SimpleTokenizer()
                else:
                    self.tokenizer = SimpleTokenizer(
                        specialTokens=self.textEmbeddingsTokens
                    )

            ### Step 5: Create Models ###
            """
            We need to create empty models before we can compile them with
            the weights of the trained models.
            First, let's check for pytorch weights. If given, we will load them later.
            If not, then we're loading in a pre-compiled model OR weights made for TensorFlow
            """

            ## Step 5.1: Create weightless models ##
            if controlNet[0] == True:
                keras_print("\nUsing ControlNet", controlNet[1])

            text_encoder, diffusion_model, decoder, encoder, control_net = (
                CreateModels(
                    self.imageHeight,
                    self.imageWidth,
                    preCompiled=None,  # If not None, then we're passing on Keras weights ".h5"
                    legacy=legacy,
                    addedTokens=self.textEmbeddings,
                    useControlNet=[controlNet[0]],
                    device=self.device,
                )
            )

            ## Step 5.2 Create object/class variables that point to the compiled models
            self.text_encoder = text_encoder
            self.diffusion_model = diffusion_model
            self.decoder = decoder
            self.encoder = encoder
            self.controlNet = control_net

            ## Step 5.4: Load Weights
            # NOTE: must be done after creating models
            self.weights = weights

            self.set_weights(weights, VAE)

            ### Step 6: Load Text Embedding Weights ###
            if self.textEmbeddings is not None:
                if legacy is True:
                    CLIP = CLIPTextTransformer
                else:
                    CLIP = OpenCLIPTextTransformer
                self.text_encoder = textEmbeddingTools.load_text_embedding_weight(
                    text_encoder=text_encoder,
                    CLIP=CLIP,
                    max_text_length=MAX_TEXT_LEN,
                    embeddings=self.textEmbeddings,
                    legacy=legacy,
                )

            ### Step 7: Load ControlNet Weights ###
            if controlNet[0] == True:
                if ".safetensors" in controlNet[1]:
                    loadWeightsFromSafeTensor(
                        self,
                        controlNet[
                            1
                        ],  # Which weights to load, in this case maybe all four models
                        legacy,  # Which version of Stable Diffusion
                        ["controlNet"],  # Which specific Models to load
                    )
                elif ".pth" in controlNet[1]:
                    loadWeightsFromPytorchCKPT(
                        self,
                        controlNet[
                            1
                        ],  # Which weights to load, in this case maybe all four models
                        legacy,  # Which version of Stable Diffusion
                        ["controlNet"],  # Which specific Models to load
                    )

            ### Step 8: Compile Models ###
            self.jitCompile = jit_compile
            self.compile_models(optimizer, self.jitCompile)

            ## Cache
            self.prompt = None
            self.negativePrompt = None
            self.encodedPrompt = None
            self.encodedNegativePrompt = None
            self.batch_size = None
            self.controlNetCache = None

    def compile_models(self, optimizer="nadam", jit_compile=False):
        modules = ["text_encoder", "diffusion_model", "decoder", "encoder"]

        if jit_compile is True:
            keras_print(
                "\nCompiling models with XLA (Accelerated Linear Algebra):"
            )
        else:
            keras_print("\nCompiling models")

        with keras.device(self.device):
            for module in modules:
                getattr(self, module).compile(
                    optimizer=keras.optimizers.Adam(), jit_compile=jit_compile
                )
                print(module, "compiled.")

    """
    Generate and image, the key function
    """

    def generate(
        self,
        prompt,
        negative_prompt=None,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=7.5,
        temperature=1,
        seed=None,
        input_image=None,  # expecting file path as a string or np.ndarray
        input_image_strength=0.5,
        input_mask=None,  # expecting a file path as a string
        sampler=None,
        control_net_strength=1,
        control_net_image=None,
        control_net_cache=False,
        v_prediction=False,
    ):
        with keras.device(self.device):
            ## Memory Efficiency
            # Clear up tensorflow memory
            keras_print("\n...cleaning memory...")
            keras.backend.clear_session()
            gc.collect()

            keras_print("...getting to work...")

            ### Step 1: Cache Prompts
            if self.prompt != prompt:  # New prompt?
                # Create prompt cache
                self.prompt = prompt
                self.encodedPrompt = None

            if self.negativePrompt != negative_prompt:  # New negative prompt?
                # Create negative prompt cache
                self.negativePrompt = negative_prompt
                self.encodedNegativePrompt = None

            if self.batch_size != batch_size:  # New batch size?
                # clear prompt caches if batch_size has changed
                self.encodedPrompt = None
                self.encodedNegativePrompt = None
                self.batch_size = batch_size

            ### Step 2: Tokenize prompts
            # the tokenized prompts are AKA "starting context"
            # we'll also tokenize the negative prompt, the "unconditional context"

            if self.encodedPrompt is None:
                # No cached encoded prompt exists
                keras_print("\n...tokenizing prompt...")

                if self.textEmbeddings is not None:
                    keras_print("...checking for text embeddings...")
                    prompt = textEmbeddingTools.inject_tokens(
                        prompt=prompt, embeddings=self.textEmbeddings
                    )

                phrase, pos_ids = self.encode_text(
                    prompt, batch_size, self.legacy
                )

                keras_print("...encoding the tokenized prompt...")
                context = self.text_encoder([phrase, pos_ids], training=False)

                # Cache encoded prompt
                self.encodedPrompt = context
            else:
                # Load cached encoded prompt
                keras_print("...using cached encoded prompt...")
                context = self.encodedPrompt

            if self.encodedNegativePrompt is None:
                keras_print("...tokenizing negative prompt...")
                if negative_prompt is None:
                    # Encoding text requires a string variable
                    negative_prompt = ""

                if self.textEmbeddings is not None:
                    keras_print("...checking for text embeddings...")
                    negative_prompt = textEmbeddingTools.inject_tokens(
                        prompt=negative_prompt, embeddings=self.textEmbeddings
                    )

                unconditional_tokens, pos_ids = self.encode_text(
                    negative_prompt, batch_size, self.legacy
                )

                keras_print("...encoding the tokenized negative prompt...")
                unconditional_context = self.text_encoder(
                    [unconditional_tokens, pos_ids], training=False
                )

                # Cache encoded negative prompt
                self.encodedNegativePrompt = unconditional_context
            else:
                keras_print("...using cached encoded negative prompt...")
                unconditional_context = self.encodedNegativePrompt

            ### Step 3: Prepare the input image, if it was given
            ## If given, we're expecting an np.ndarry
            input_image_tensor = None
            if input_image is not None:
                if isinstance(input_image, np.ndarray):
                    print("...received NumPy Array...")
                    print(input_image.shape)

                    input_image = keras.ops.convert_to_tensor(
                        input_image, dtype=keras.config.floatx()
                    )

                    # Resize the image to self.imageHeight x self.imageWidth
                    input_image = keras.ops.image.resize(
                        input_image, [self.imageHeight, self.imageWidth]
                    )

                    inputImageArray = keras.initializers.Constant(
                        input_image, dtype=keras.config.floatx()
                    )
                    inputImageArray = keras.ops.expand_dims(
                        input_image[..., :3], axis=0
                    )
                    input_image_tensor = keras.ops.cast(
                        (inputImageArray / 255.0) * 2 - 1, self.dtype
                    )

                    print(input_image_tensor.shape)
                    # displayImage(input_image_tensor, name = "1preppedImage")
                elif isinstance(input_image, Array):
                    print("...received jax.Array (JAX Array)...")
                    input_image_tensor = input_image
                    # displayImage(input_image_tensor, name = "1preppedImage")

            ### Step 4: Prepare the image mask, if it was given
            if isinstance(input_mask, str):
                print("...preparing input mask...")
                input_mask = Image.open(input_mask)
                input_mask = input_mask.resize(
                    (self.imageWidth, self.imageHeight)
                )
                input_mask_array = np.array(input_mask, dtype=np.float32)[
                    None, ..., None
                ]
                input_mask_array = input_mask_array / 255.0

                latent_mask = input_mask.resize(
                    (self.imageWidth // 8, self.imageHeight // 8)
                )
                latent_mask = np.array(latent_mask, dtype=np.float32)[
                    None, ..., None
                ]
                latent_mask = 1 - (latent_mask.astype("float") / 255.0)
                latent_mask_tensor = keras.ops.cast(
                    keras.ops.repeat(latent_mask, batch_size, axis=0),
                    self.dtype,
                )
            else:
                latent_mask_tensor = None

            ### Step 5: Create a random seed if one is not provided
            if seed is None:
                keras_print("...generating random seed...")
                seed = random.randint(1000, sys.maxsize)
                seed = int(seed)
            else:
                seed = int(seed)

            ### Step 6: Create time steps
            keras_print("...creating time steps...")
            timesteps = keras.ops.arange(1, 1000, 1000 // num_steps)

            ### Step 7: Load Sampler and:
            ### Step 8: Start Diffusion
            if sampler == "DPMSolver":
                keras_print("...using DPM Solver...\n...starting sampler...")

                alphasCumprod = keras.initializers.Constant(_ALPHAS_CUMPROD)

                noiseScheduler = DPMSolver.NoiseScheduler(
                    beta_schedule="scaled_linear"
                )

                print(
                    "...starting diffusion...\n...this solver not supported yet!\nDividing by zero now:\n"
                )

                x = 5 / 0
            else:
                if sampler is None:
                    keras_print("...no sampler given...")

                # ControlNet
                # Parameters: [0]Use ControlNet, [1] Input Image, [2]Strength, [3] Cache Input
                if self.controlNet is not None:
                    control_net_image = [
                        keras.initializers.Constant(
                            control_net_image[0].copy(),
                            dtype=keras.config.floatx(),
                        )
                        / 255.0
                    ]
                    if control_net_cache is False:
                        self.controlNetCache = None
                    if isinstance(self.controlNetCache, dict):
                        if (
                            len(self.controlNetCache["unconditional"])
                            != timesteps
                        ):
                            keras_print("Incompatible cache!")
                            self.controlNetCache = None
                    control_net_paramters = [
                        True,
                        control_net_image,
                        control_net_strength,
                        self.controlNetCache,
                    ]
                else:
                    control_net_paramters = [False, None, 1, None]

                # Create Sampler
                sampler = BasicSampler(
                    model=self,
                    timesteps=timesteps,
                    batch_size=batch_size,
                    seed=seed,
                    input_image=input_image_tensor,
                    input_mask=latent_mask_tensor,
                    input_image_strength=input_image_strength,
                    temperature=temperature,
                    alphas_cumprod=_ALPHAS_CUMPROD,
                    controlNetInput=control_net_paramters[
                        1
                    ],  # Input Image, assuming pre-processed
                )

                if v_prediction is True:
                    keras_print("...using v-prediction...")

                # Sample, create image essentially
                latentImage, self.controlNetCache = sampler.sample(
                    context,
                    unconditional_context,
                    unconditional_guidance_scale,
                    control_net=[
                        control_net_paramters[0],
                        control_net_paramters[2],
                        control_net_paramters[3],
                    ],  # [0]Use Control Net, [2]Strength, [3]Cache
                    vPrediction=v_prediction,
                    device=self.device,
                )

            ### Step 9: Decoding stage
            keras_print("\n...decoding latent image...")
            decoded = self.decoder(latentImage, training=False)
            decoded = ((decoded + 1) / 2) * 255

            ### Step 10: Merge inpainting result of input mask with original image
            if input_mask is not None:
                decoded = (
                    inputImageArray * (1 - input_mask_array)
                    + np.array(decoded) * input_mask_array
                )

            ### Memory cleanup
            gc.collect()

            ### Step 11: return final image as an array
            return np.clip(decoded, 0, 255).astype("uint8")

    def change_policy(self, policy):
        if policy == "mixed_float16":
            # self.dtype = tf.float16
            if keras.mixed_precision.global_policy().name != "mixed_float16":
                print("\n...using mixed precision...")
                keras.mixed_precision.set_global_policy("mixed_float16")
                # self.dtype = tf.float16

        if policy == "float32":
            # self.dtype = keras.config.floatx()
            if keras.mixed_precision.global_policy().name != "float32":
                print("\n...using regular precision...")
                keras.mixed_precision.set_global_policy("float32")
                # self.dtype = keras.config.floatx()

    def encode_text(self, prompt, batch_size, legacy):
        text_limit = MAX_TEXT_LEN - 1
        with keras.device(self.device):
            if legacy is True:
                # First, encode the prompt
                inputs = self.tokenizer.encode(prompt)
                # Then check the inputs length and truncate if too long
                if len(inputs) > text_limit:
                    keras_print(
                        "Prompt is too long (should be less than 77 words). Truncating down to 77 words..."
                    )
                    inputs = inputs[:text_limit]

                """## Create numpy array with the inputs
                # Phrase - aka the prompt
                phrase = [49406] + inputs + [49407] * (text_limit - len(inputs))
                phrase = np.array(phrase)[None].astype("int32")
                phrase = np.repeat(phrase, batch_size, axis = 0)

                # Position ID
                pos_ids = np.array(list(range(77)))[None].astype("int32")
                pos_ids = np.repeat(pos_ids, batch_size, axis = 0)"""

                # Phrase - aka the prompt
                phrase = keras.ops.concatenate(
                    [[49406], inputs, [49407] * (text_limit - len(inputs))],
                    axis=0,
                )
                phrase = keras.ops.expand_dims(phrase, axis=0)
                phrase = keras.ops.repeat(phrase, batch_size, axis=0)
                phrase = keras.ops.cast(phrase, dtype="int32")

                # Position ID
                pos_ids = keras.ops.expand_dims(keras.ops.arange(77), axis=0)
                pos_ids = keras.ops.repeat(pos_ids, batch_size, axis=0)
                pos_ids = keras.ops.cast(pos_ids, dtype="int32")
            else:
                # First, encode the prompt
                text_limit += 1
                if isinstance(prompt, str):
                    inputs = [prompt]
                # Then tokenize the prompt
                startOfToken = self.tokenizer.encoder["<start_of_text>"]
                endOfToken = self.tokenizer.encoder["<end_of_text>"]
                allTokens = [
                    [startOfToken] + self.tokenizer.encode(input) + [endOfToken]
                    for input in inputs
                ]
                # Create the empty tensor/numpy array to load the tokens into
                phrase = np.zeros((len(allTokens), text_limit), dtype=np.int32)

                for i, tokens in enumerate(allTokens):
                    if len(tokens) > text_limit:
                        tokens = tokens[:text_limit]  # Truncate
                        tokens[-1] = endOfToken
                    phrase[i, : len(tokens)] = np.array(tokens)

                phrase = np.repeat(phrase, batch_size, axis=0)

                pos_ids = np.array(list(range(text_limit)))[None].astype("int32")
                pos_ids = np.repeat(pos_ids, batch_size, axis=0)

            return phrase, pos_ids

    def set_weights(self, weights, VAE="Original"):
        self.weights = weights
        # Load weights for VAE models, if given
        if VAE != "Original":
            if ".ckpt" in VAE:
                loadWeightsFromPytorchCKPT(
                    self,
                    VAE,  # Which weights to load, in this case weights for VAE
                    self.legacy,  # Which version of Stable Diffusion
                    ["decoder", "encoder"],  # Models to load
                    True,
                )
            elif ".safetensors" in VAE:
                loadWeightsFromSafeTensor(
                    self,
                    VAE,  # Which weights to load, in this case weights for VAE
                    self.legacy,  # Which version of Stable Diffusion
                    ["decoder", "encoder"],  # Models to load
                    True,
                )
            else:
                loadWeightsFromKeras(self, VAE, VAEOnly=True)

        # Load all weights
        if ".ckpt" in self.weights:
            if VAE == "Original":  # Load all weights from PyTorch .ckpt
                modules = [
                    "text_encoder",
                    "diffusion_model",
                    "decoder",
                    "encoder",
                ]
            else:  # only load weights for the text encoder and diffusion model if VAE was given
                modules = ["text_encoder", "diffusion_model"]
            loadWeightsFromPytorchCKPT(
                self,
                self.weights,  # Which weights to load, in this case maybe all four models
                self.legacy,  # Which version of Stable Diffusion
                modules,  # Which specific Models to load
            )
        elif ".safetensors" in self.weights:
            if VAE == "Original":  # Load all weights from PyTorch .ckpt
                modules = [
                    "text_encoder",
                    "diffusion_model",
                    "decoder",
                    "encoder",
                ]
            else:  # only load weights for the text encoder and diffusion model if VAE was given
                modules = ["text_encoder", "diffusion_model"]
            loadWeightsFromSafeTensor(
                self,
                self.weights,  # Which weights to load, in this case maybe all four models
                self.legacy,  # Which version of Stable Diffusion
                modules,  # Which specific Models to load
            )
        else:
            if VAE == "Original":
                loadWeightsFromKeras(self, self.weights, VAEOnly=False)
            else:
                loadWeightsFromKeras(self, self.weights, VAEOnly=VAE)


### Functions ###


def CreateModels(
    imageHeight=512,
    imageWidth=512,
    preCompiled=None,
    legacy=True,
    addedTokens=0,
    useControlNet=[False],
    device=None,
):
    with keras.device(device):
        # Memory Clean up
        keras.backend.clear_session()
        gc.collect()

        controlNet = None

        if legacy is True:
            # Are we using Pre-Stable Diffusion 2.0?

            keras_print("\nCreating models in legacy mode...")

            # Create Text Encoder model
            input_word_ids = keras.layers.Input(
                shape=(MAX_TEXT_LEN,), dtype="int32"
            )
            input_pos_ids = keras.layers.Input(
                shape=(MAX_TEXT_LEN,), dtype="int32"
            )
            embeds = CLIPTextTransformer()([input_word_ids, input_pos_ids])
            text_encoder = keras.models.Model(
                [input_word_ids, input_pos_ids], embeds
            )
            keras_print("Created text encoder model")

            if useControlNet[0] is False:
                # Create Diffusion model
                diffusion_model = DiffusionModel(
                    imageHeight, imageWidth, MAX_TEXT_LEN
                )
                keras_print("Created diffusion model")
            else:
                # Create seperate control net model
                controlNet = ControlNetModel(
                    imageHeight, imageWidth, MAX_TEXT_LEN
                )

                keras_print("Created ControlNet Model")

                # Create Diffusion model
                diffusion_model = ControlDiffusionModel(
                    imageHeight, imageWidth, MAX_TEXT_LEN
                )
                keras_print("Created diffusion model")

            # Create Decoder model
            decoder = Decoder(
                img_height=imageHeight,
                img_width=imageWidth,
            )
            keras_print("Created decoder model")

            # Create Image Encoder model
            encoder = ImageEncoder(img_height=imageHeight, img_width=imageWidth)
            keras_print("Created encoder model")

        else:
            # We're using SD 2.0 and newer

            print("\nCreating models in contemporary mode...")

            # Create Text Encoder model
            input_word_ids = keras.layers.Input(
                shape=(MAX_TEXT_LEN,), dtype="int32"
            )
            input_pos_ids = keras.layers.Input(
                shape=(MAX_TEXT_LEN,), dtype="int32"
            )
            embeds = OpenCLIPTextTransformer()([input_word_ids, input_pos_ids])
            text_encoder = keras.models.Model(
                [input_word_ids, input_pos_ids], embeds
            )
            print("Created text encoder model")

            # Create Diffusion model
            diffusion_model = DiffusionModelV2(
                imageHeight, imageWidth, MAX_TEXT_LEN
            )
            print("Created diffusion model")

            # Create Decoder model
            decoder = Decoder(
                img_height=imageHeight,
                img_width=imageWidth,
            )
            print("Created decoder model")

            # Create Image Encoder model
            encoder = ImageEncoder(img_height=imageHeight, img_width=imageWidth)
            print("Created encoder model")

        # return created models
        return text_encoder, diffusion_model, decoder, encoder, controlNet


def loadWeightsFromKeras(models, weightsPath, VAEOnly=False):
    keras_print("\nLoading Keras weights for:", weightsPath)
    textEncoderWeights = weightsPath + "/text_encoder.h5"
    diffusionModelWeights = weightsPath + "/diffusion_model.h5"
    imageEncoderWeights = weightsPath + "/encoder.h5"
    decoderWeights = weightsPath + "/decoder.h5"

    if VAEOnly is False:
        models.text_encoder.load_weights(textEncoderWeights)
        keras_print("...Text Encoder weights loaded!")
        models.diffusion_model.load_weights(diffusionModelWeights)
        keras_print("...diffusion model weights loaded")
    models.encoder.load_weights(imageEncoderWeights)
    keras_print("...Image Encoder weights loaded!")
    models.decoder.load_weights(decoderWeights)
    keras_print("...Decoder weights loaded!")
    keras_print("All weights loaded!")


def loadWeightsFromPytorchCKPT(
    model,
    pytorch_ckpt_path,
    legacy=True,
    moduleName=["text_encoder", "diffusion_model", "decoder", "encoder"],
    VAEoverride=False,
):
    print("\nLoading pytorch checkpoint " + pytorch_ckpt_path)
    pytorchWeights = torch.load(pytorch_ckpt_path, map_location="mps")
    if legacy is True:
        ## Legacy Mode
        print("...loading pytroch weights in legacy mode...")
        for module in moduleName:
            module_weights = []
            if module == "text_encoder":
                module = "text_encoder_legacy"
            for i, (key, perm) in enumerate(PYTORCH_CKPT_MAPPING[module]):
                if VAEoverride is True:
                    key = key.replace("first_stage_model.", "")
                if "state_dict" in pytorchWeights:
                    weight = pytorchWeights["state_dict"][key].detach().numpy()
                else:
                    weight = pytorchWeights[key].detach().numpy()
                if perm is not None:
                    weight = np.transpose(weight, perm)
                module_weights.append(weight)
            if module == "text_encoder_legacy":
                module = "text_encoder"

            getattr(model, module).set_weights(module_weights)

            print(
                "Loaded %d pytorch weights for %s"
                % (len(module_weights), module)
            )
    else:
        ## Contemporary Mode
        print("...loading pytorch weights in contemporary mode...")
        for module in moduleName:
            module_weights = []
            in_projWeightConversion = []
            in_projBiasConversion = []
            for i, (key, perm) in enumerate(PYTORCH_CKPT_MAPPING[module]):
                if "in_proj" not in key:
                    if VAEoverride is True:
                        key = key.replace("first_stage_model.", "")
                    weight = pytorchWeights["state_dict"][key].detach().numpy()

                    if module == "diffusion_model":
                        if "proj_in.weight" in key or "proj_out.weight" in key:
                            # print(i+1,
                            # " Overriding premuation from constants:\n",key)
                            # This is so the constants.py "diffusion_model"
                            # dictionary keeps its legacy state
                            perm = (1, 0)

                    if perm is not None:
                        weight = np.transpose(weight, perm)
                    module_weights.append(weight)
                else:
                    if module == "text_encoder":
                        # "in_proj" layer of SD2.x is a matrix multiplication
                        # of the query, key, and value layers of SD1.4/5
                        # We will slice this layer into the the three vectors
                        if "weight" in key:
                            # Get the in_proj.weight
                            originalWeight = (
                                pytorchWeights["state_dict"][key]
                                .float()
                                .numpy()
                            )

                            queryWeight = originalWeight[:1024, ...]
                            queryWeight = np.transpose(queryWeight, (1, 0))

                            keyWeight = originalWeight[1024:2048, ...]
                            keyWeight = np.transpose(keyWeight, (1, 0))

                            valueWeight = originalWeight[2048:, ...]
                            valueWeight = np.transpose(valueWeight, (1, 0))

                            # Clear local variable to carry forward for bias
                            in_projWeightConversion = []

                            in_projWeightConversion.append(
                                queryWeight
                            )  # Query states
                            in_projWeightConversion.append(
                                keyWeight
                            )  # Key states
                            in_projWeightConversion.append(
                                valueWeight
                            )  # Value states
                        elif "bias" in key:
                            originalBias = (
                                pytorchWeights["state_dict"][key]
                                .float()
                                .numpy()
                            )

                            queryBias = originalBias[:1024]

                            keyBias = originalBias[1024:2048]

                            valueBias = originalBias[2048:]

                            # Clear local variable to carry forward for bias
                            in_projBiasConversion = []

                            in_projBiasConversion.append(
                                queryBias
                            )  # Query states
                            in_projBiasConversion.append(keyBias)  # Key states
                            in_projBiasConversion.append(
                                valueBias
                            )  # Value states

                            # add converted weights/biases in the correct order
                            # Query
                            module_weights.append(in_projWeightConversion[0])
                            module_weights.append(in_projBiasConversion[0])
                            # Key
                            module_weights.append(in_projWeightConversion[1])
                            module_weights.append(in_projBiasConversion[1])
                            # Value
                            module_weights.append(in_projWeightConversion[2])
                            module_weights.append(in_projBiasConversion[2])

            print("Loading weights for ", module)

            getattr(model, module).set_weights(module_weights)
            print(
                "Loaded %d pytorch weights for %s"
                % (len(module_weights), module)
            )

    ## Memory Clean up
    del pytorchWeights


def loadWeightsFromSafeTensor(
    model,
    safetensor_path,
    legacy=True,
    moduleName=["text_encoder", "diffusion_model", "decoder", "encoder"],
    VAEoverride=False,
):
    print("\nLoading safetensor " + safetensor_path)
    safeTensorWeights = load_file(safetensor_path)
    if legacy is True:
        ## Legacy Mode
        print("...loading safetensors weights in legacy mode...")
        for module in moduleName:
            module_weights = []
            if module == "text_encoder":
                module = "text_encoder_legacy"
            for i, (key, perm) in enumerate(PYTORCH_CKPT_MAPPING[module]):
                if VAEoverride is True:
                    key = key.replace("first_stage_model.", "")
                if "state_dict" in safeTensorWeights:
                    weight = (
                        safeTensorWeights["state_dict"][key].detach().numpy()
                    )
                else:
                    if module == "controlNet":
                        # Replace "control_model." in case the safetensor
                        # doesn't have that key
                        key = key.replace("control_model.", "")
                    weight = safeTensorWeights[key].detach().numpy()
                if perm is not None:
                    weight = np.transpose(weight, perm)
                module_weights.append(weight)
            if module == "text_encoder_legacy":
                module = "text_encoder"

            getattr(model, module).set_weights(module_weights)

            print(
                "Loaded %d safetensors weights for %s"
                % (len(module_weights), module)
            )
    else:
        ## Contemporary Mode
        print("...loading safetensors weights in contemporary mode...")
        for module in moduleName:
            module_weights = []
            in_projWeightConversion = []
            in_projBiasConversion = []
            for i, (key, perm) in enumerate(PYTORCH_CKPT_MAPPING[module]):
                if "in_proj" not in key:
                    if VAEoverride is True:
                        key = key.replace("first_stage_model.", "")
                    weight = (
                        safeTensorWeights["state_dict"][key].detach().numpy()
                    )

                    if module == "diffusion_model":
                        if "proj_in.weight" in key or "proj_out.weight" in key:
                            # print(i+1,
                            # " Overriding premuation from constants:\n",key)
                            # This, so the constants.py "diffusion_model" dict
                            # keeps its legacy state
                            perm = (1, 0)

                    if perm is not None:
                        weight = np.transpose(weight, perm)
                    module_weights.append(weight)
                else:
                    if module == "text_encoder":
                        # "in_proj" layer of SD2.x is a matrix multiplcation
                        # of the query, key, and value layers of SD1.4/5
                        # We will slice this layer into the the three vectors
                        if "weight" in key:
                            # Get the in_proj.weight
                            originalWeight = (
                                safeTensorWeights["state_dict"][key]
                                .float()
                                .numpy()
                            )

                            queryWeight = originalWeight[:1024, ...]
                            queryWeight = np.transpose(queryWeight, (1, 0))

                            keyWeight = originalWeight[1024:2048, ...]
                            keyWeight = np.transpose(keyWeight, (1, 0))

                            valueWeight = originalWeight[2048:, ...]
                            valueWeight = np.transpose(valueWeight, (1, 0))

                            # Clear local variable to carry forward for bias
                            in_projWeightConversion = []

                            in_projWeightConversion.append(
                                queryWeight
                            )  # Query states
                            in_projWeightConversion.append(
                                keyWeight
                            )  # Key states
                            in_projWeightConversion.append(
                                valueWeight
                            )  # Value states
                        elif "bias" in key:
                            originalBias = (
                                safeTensorWeights["state_dict"][key]
                                .float()
                                .numpy()
                            )

                            queryBias = originalBias[:1024]

                            keyBias = originalBias[1024:2048]

                            valueBias = originalBias[2048:]

                            # Clear local variable to carry forward for bias
                            in_projBiasConversion = []

                            in_projBiasConversion.append(
                                queryBias
                            )  # Query states
                            in_projBiasConversion.append(keyBias)  # Key states
                            in_projBiasConversion.append(
                                valueBias
                            )  # Value states

                            # add converted weights/biases in the correct order
                            # Query
                            module_weights.append(in_projWeightConversion[0])
                            module_weights.append(in_projBiasConversion[0])
                            # Key
                            module_weights.append(in_projWeightConversion[1])
                            module_weights.append(in_projBiasConversion[1])
                            # Value
                            module_weights.append(in_projWeightConversion[2])
                            module_weights.append(in_projBiasConversion[2])

            print("Loading weights for ", module)

            getattr(model, module).set_weights(module_weights)
            print(
                "Loaded %d safetensors weights for %s"
                % (len(module_weights), module)
            )

    ## Memory Clean up
    del safeTensorWeights


def displayImage(input_image_tensor, name="image"):
    # Assuming input_image_tensor is a TensorFlow tensor representing the image
    # Remove the batch dimension
    input_image_tensor = keras.ops.squeeze(input_image_tensor, axis=0)

    # Convert the tensor to a NumPy array
    input_image_array = input_image_tensor.numpy()

    # Rescale the array to the range [0, 255]
    input_image_array = ((input_image_array + 1) / 2.0) * 255.0

    # Convert the array to uint8 data type
    input_image_array = input_image_array.astype("uint8")

    # Display the image using Matplotlib
    imageFromBatch = Image.fromarray(input_image_array)
    imageFromBatch.save("debug/" + name + ".png")
