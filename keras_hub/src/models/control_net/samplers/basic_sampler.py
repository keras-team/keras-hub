### Basic Modules
import math
import random

import keras
from keras_hub.src.models.control_net.utils import keras_print
### Modules for image building
from PIL import Image

### TensorFlow Modules


# import cv2  # OpenCV


class BasicSampler:

    def __init__(
        self,
        model=None,
        timesteps=keras.ops.arange(1, 1000, 1000 // 50),
        batchSize=1,
        seed=1990,
        inputImage=None,  # Expecting a tensor
        inputMask=None,  # Expecting a tensor
        inputImageStrength=0.5,
        temperature=1,
        AlphasCumprod=None,
        controlNetInput=None,
    ):
        print("...starting Basic Sampler...")
        self.model = model
        self.timesteps = timesteps
        self.batchSize = batchSize
        self.seed = seed
        self.inputImage = inputImage
        self.inputMask = inputMask
        self.inputImageStrength = inputImageStrength
        self.inputImageNoise_T = self.timesteps[
            int(len(self.timesteps) * self.inputImageStrength)
        ]
        self.temperature = temperature
        self.AlphasCumprod = AlphasCumprod  # Length = 1000

        self.latent, self.alphas, self.alphas_prev, self.controlNetInput = (
            self.getStartingParameters(
                self.timesteps,
                self.batchSize,
                seed,
                inputImage=self.inputImage,
                inputImageNoise_T=self.inputImageNoise_T,
                controlNetInput=controlNetInput,
            )
        )

        if self.inputImage is not None:
            self.timesteps = self.timesteps[
                : int(len(self.timesteps) * self.inputImageStrength)
            ]

        print("...sampler ready...")

    def addNoise(self, x, t, noise=None, DType=keras.config.floatx()):
        batch_size, w, h = x.shape[0], x.shape[1], x.shape[2]
        if noise is None:
            # Post-Encode version:
            noise = keras.random.normal((batch_size, w, h, 4), dtype=DType)
            # Pre-Encode version:
            # noise = keras.random.normal((batch_size,w,h,3), dtype = DType)
        sqrt_alpha_prod = self.AlphasCumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.AlphasCumprod[t]) ** 0.5

        return sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

    def getStartingParameters(
        self,
        timesteps,
        batchSize,
        seed,
        inputImage=None,
        inputImageNoise_T=None,
        controlNetInput=None,
    ):
        # Use floor division to get minimum height/width of image size
        # for the Diffusion and Decoder models
        floorDividedImageHeight = self.model.imageHeight // 8
        floorDividedImageWidth = self.model.imageWidth // 8

        alphas = [self.AlphasCumprod[t] for t in timesteps]  # sample steps length
        alphas_prev = [1.0] + alphas[:-1]

        if inputImage is None:
            # Create a random input image from noise
            latent = keras.random.stateless_normal(
                (batchSize, floorDividedImageHeight, floorDividedImageWidth, 4),
                seed=[seed, seed],
            )
        else:
            ## Debug Variables
            randomNumber = str(random.randint(0, 2**31))

            # Noise the input image before encoding
            # latent = self.addNoise(inputImage, inputImageNoise_T)

            # Encode the given image
            print(inputImage.shape)
            latent = self.model.encoder(inputImage, training=False)
            print(latent.shape)
            # self.displayImage(latent,("encoded" + randomNumber))
            # Repeat it within the tensor for the given batch size
            latent = keras.ops.repeat(latent, batchSize, axis=0)
            # Noise the image after encode
            latent = self.addNoise(latent, inputImageNoise_T)

        if controlNetInput is None:
            # Create a random input image from noise
            controlNetLatent = keras.random.normal(
                (batchSize, floorDividedImageHeight, floorDividedImageWidth, 3),
                seed=seed,
            )
        else:
            controlNetLatent = keras.ops.repeat(controlNetInput, batchSize, axis=0)

        return latent, alphas, alphas_prev, controlNetLatent

    def get_x_prev_and_pred_x0(self, x, e_t, index, a_t, a_prev, temperature, seed):
        sigma_t = keras.initializers.Constant(0.0)
        sqrt_one_minus_at = keras.ops.sqrt(keras.initializers.Constant(1.0) - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / keras.ops.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = (
            keras.ops.sqrt(
                keras.initializers.Constant(1.0) - a_prev - keras.ops.square(sigma_t)
            )
            * e_t
        )
        noise = sigma_t * keras.random.normal(x.shape, seed=seed) * temperature
        x_prev = keras.ops.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0

    # Keras Version
    def sample(
        self,
        context,
        unconditionalContext,
        unconditionalGuidanceScale,
        controlNet=[None, 1, None],  # [0]Use ControlNet, [1]Strength, [2] Cache Input
        vPrediction=False,
        device=None,
    ):
        with keras.device(device):
            # Progress Bar set-up
            progbar = keras.utils.Progbar(len(self.timesteps))
            iteration = 0

            # ControlNet Cache
            if controlNet[2] is not None:
                keras_print("...using controlNet cache...")
                controlNetCache = controlNet[2]
            else:
                if controlNet[0] is True:
                    keras_print("...creating controlNet cache...")
                controlNetCache = []

            if controlNet[2] is not None and len(controlNet[2]) != len(
                list(enumerate(self.timesteps))[::-1]
            ):
                keras_print("...updating controlNet cache...")
                controlNetCache = []
                controlNet[2] = None

            keras_print("...sampling:")

            # Iteration loop
            for index, timestep in list(enumerate(self.timesteps))[::-1]:

                latentPrevious = self.latent

                # Establish timestep embedding
                # t_emb = self.timestepEmbedding(float(timestep))
                t_emb = self.timestepEmbedding(int(timestep))
                t_emb = keras.ops.repeat(
                    t_emb, self.batchSize, axis=0
                )  # shape is (1, 320)

                inputsConditional = [self.latent, t_emb, context]
                inputsUnconditional = [self.latent, t_emb, unconditionalContext]

                if controlNet[0] is True:

                    if controlNet[2] is None:
                        # No cache was given, so we're starting from scratch

                        # Get unconditional and conditional tensors(arrays)
                        controlNetUnconditionalArray = self.model.controlNet(
                            [
                                self.latent,
                                t_emb,
                                unconditionalContext,
                                keras.ops.concatenate(self.controlNetInput, axis=3),
                            ],
                            training=False,
                        )
                        controlNetConditionalArray = self.model.controlNet(
                            [
                                self.latent,
                                t_emb,
                                context,
                                keras.ops.concatenate(self.controlNetInput, axis=3),
                            ],
                            training=False,
                        )

                        # Apply strength
                        controlNetUnconditionalArray = [
                            result * scale
                            for result, scale in zip(
                                controlNetUnconditionalArray, controlNet[1]
                            )
                        ]
                        controlNetConditionalArray = [
                            result * scale
                            for result, scale in zip(
                                controlNetConditionalArray, controlNet[1]
                            )
                        ]

                        # Update Cache
                        controlNetCacheData = {
                            "unconditional": controlNetUnconditionalArray,
                            "conditional": controlNetConditionalArray,
                        }
                        controlNetCache.insert(0, controlNetCacheData)

                        # Add the resulting tensors from the contorlNet models to the list of inputs for the diffusion models
                        inputsUnconditional.append(controlNetUnconditionalArray)
                        inputsConditional.append(controlNetConditionalArray)
                    else:
                        # Use ControlNet Cache
                        inputsUnconditional.extend(
                            controlNetCache[index]["unconditional"]
                        )
                        inputsConditional.extend(controlNetCache[index]["conditional"])

                # Get unconditional (negative prompt) latent image
                unconditionalLatent = self.model.diffusion_model(
                    inputsUnconditional, training=False
                )

                # Get conditional (positive prompt) latent image
                self.latent = self.model.diffusion_model(
                    inputsConditional, training=False
                )

                # Combine the two latent images
                self.latent = unconditionalLatent + unconditionalGuidanceScale * (
                    self.latent - unconditionalLatent
                )

                # Alphas
                a_t, a_prev = self.alphas[index], self.alphas_prev[index]

                # Predictions
                if vPrediction is False:
                    # Debug Info
                    if iteration == 0:
                        print("Latent Previous dtype:", latentPrevious.dtype)
                        print("Latent dtype:", self.latent.dtype)

                    # Make the data types (dtypes) match
                    if latentPrevious.dtype != self.latent.dtype:
                        latentPrevious = keras.ops.cast(
                            latentPrevious, dtype=self.latent.dtype
                        )

                    pred_x0 = (
                        latentPrevious - math.sqrt(1.0 - a_t) * self.latent
                    ) / math.sqrt(a_t)

                    self.latent = (
                        self.latent * math.sqrt(1.0 - a_prev)
                        + math.sqrt(a_prev) * pred_x0
                    )
                else:
                    # v-Prediction for SD 2.1-V models
                    self.latent = self.predictEpsFromZandV(
                        latentPrevious, index, self.latent
                    )

                # Keras Progress Bar Update
                iteration += 1
                progbar.update(iteration)

            keras_print("...finished! Returning latent image...")

            return self.latent, controlNetCache

    def predictEpsFromZandV(self, latent, timestep, velocity):

        # sqrt_alphas_cumprod = keras.ops.sqrt(keras.ops.cumprod([1 - alpha for alpha in self.alphas], axis = 0, exclusive = True))
        sqrt_alphas_cumprod = keras.ops.sqrt(self.alphas)
        # keras_print("\nSquare Root Alphas Cumprod:\n",len(sqrt_alphas_cumprod))
        tensorShape = sqrt_alphas_cumprod.shape[0]
        # sqrt_alphas_cumprod = sqrt_alphas_cumprod[timestep]
        # sqrt_alphas_cumprod = keras.ops.reshape(sqrt_alphas_cumprod, (tensorShape,) + (1,) * (len(latent.shape) - 1))

        sqrt_one_minus_alphas_cumprod = keras.ops.sqrt(
            [1 - alpha for alpha in self.alphas]
        )
        # keras_print("\nSquare Root Alphas Cumprod Minus One:\n",len(sqrt_one_minus_alphas_cumprod))
        tensorShape = sqrt_one_minus_alphas_cumprod.shape[0]
        # sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timestep]
        # sqrt_one_minus_alphas_cumprod = keras.ops.reshape(sqrt_one_minus_alphas_cumprod, (tensorShape,) + (1,) * (len(latent.shape) - 1))

        return (
            sqrt_alphas_cumprod[timestep] * latent
            - sqrt_one_minus_alphas_cumprod[timestep] * velocity
        )

    def predictStartFromZandV(self, latent, timestep, velocity):
        # sqrt_alphas_cumprod = keras.ops.sqrt(keras.ops.cumprod([1 - alpha for alpha in self.alphas], axis = 0, exclusive = True))
        sqrt_alphas_cumprod = keras.ops.sqrt(self.alphas)
        tensorShape = sqrt_alphas_cumprod.shape[0]
        # sqrt_alphas_cumprod = sqrt_alphas_cumprod[timestep]
        # sqrt_alphas_cumprod = keras.ops.reshape(sqrt_alphas_cumprod, (tensorShape,) + (1,) * (len(latent.shape) - 1))

        # sqrt_one_minus_alphas_cumprod = keras.ops.sqrt(1 - keras.ops.cumprod(self.alphas, axis = 0, exclusive = True))
        sqrt_one_minus_alphas_cumprod = keras.ops.sqrt(
            [1 - alpha for alpha in self.alphas]
        )
        tensorShape = sqrt_one_minus_alphas_cumprod.shape[0]
        # sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timestep]
        # sqrt_one_minus_alphas_cumprod = keras.ops.reshape(sqrt_one_minus_alphas_cumprod, (tensorShape,) + (1,) * (len(latent.shape) - 1))

        """sqrt_alphas_cumprod_t = extractIntoTensor(sqrt_alphas_cumprod, timestep, latent.shape)
        sqrt_one_minus_alphas_cumprod_t = extractIntoTensor(sqrt_one_minus_alphas_cumprod, timestep, latent.shape)

        return sqrt_alphas_cumprod_t * latent - sqrt_one_minus_alphas_cumprod_t * velocity"""

        """print(sqrt_alphas_cumprod.shape)
        print(timestep)
        print(sqrt_alphas_cumprod[timestep])"""

        return (
            sqrt_alphas_cumprod[timestep] * velocity
            + sqrt_one_minus_alphas_cumprod[timestep] * latent
        )

    def timestepEmbedding(self, timesteps, dimensions=320, max_period=10000.0):
        half = dimensions // 2
        freqs = keras.ops.exp(
            -keras.ops.log(max_period)
            * keras.ops.arange(0, half, dtype=keras.config.floatx())
            / half
        )
        args = (
            keras.ops.convert_to_tensor([timesteps], dtype=keras.config.floatx())
            * freqs
        )
        embedding = keras.ops.concatenate([keras.ops.cos(args), keras.ops.sin(args)], 0)
        embedding = keras.ops.reshape(embedding, [1, -1])
        return embedding

    def displayImage(self, image, name="sampler"):
        # Assuming input_image_tensor is a TensorFlow tensor representing the image

        try:
            input_image_tensor = self.model.decoder(image, training=False)
        except Exception as e:
            print(e)
            input_image_tensor = image

        # Assuming input_image_tensor is a TensorFlow tensor representing the image
        # Remove the batch dimension
        input_image_tensor = keras.ops.squeeze(input_image_tensor, axis=0)

        # keras.ops.image.resize(input_image_tensor, [self.model.imageWidth, self.model.imageHeight])

        # Convert the tensor to a NumPy array
        input_image_array = input_image_tensor.numpy()

        # Rescale the array to the range [0, 255]
        input_image_array = ((input_image_array + 1) / 2.0) * 255.0

        # Convert the array to uint8 data type
        input_image_array = input_image_array.astype("uint8")

        # Display the image using Matplotlib
        imageFromBatch = Image.fromarray(input_image_array)
        imageFromBatch.save("debug/" + name + ".png")


"""
Utilities
"""


def extractIntoTensor(a, t, x_shape):
    b, *_ = keras.ops.shape(t)
    out = keras.ops.take(a, t, axis=-1)
    return keras.ops.reshape(out, (b,) + (1,) * (len(x_shape) - 1))
