import keras
import numpy as np
import torch as torch
from keras_hub.src.models.control_net.utils import keras_print


class Embedding:
    """
    This is an object class that stores the loaded Text Embedding
    It only needs two key variables to exist:
        Name: unique name of embedding
        Vector: vector(s) of the embedding
    """

    def __init__(self, vector, name, step=None):
        self.vector = vector
        self.name = name

        # Adjust the vector shape to (x, 768)
        # This is for single vector text embeddings, which may come as a (768,) instead of (1,768)
        if self.vector.ndim < 2:
            if self.vector.shape[0] == 768:  # Stable Diffusion 1.4/1.5
                self.vector = self.vector.reshape((1, 768))
            elif self.vector.shape[0] == 1024:  # Stable Diffusion 2.x
                self.vector = self.vector.reshape((1, 1024))

        # Create the unique tokens
        if self.vector.shape[0] > 1:
            # If we have a multidimensional vector, then we'll split up the token per dimension
            self.token = []
            for dimension in range(self.vector.shape[0]):
                self.token.append("<" + self.name + "_" + str(dimension) + ">")
            self.name = "<" + self.name + ">"
        else:
            # Single dimension vector, so the token is the name
            self.name = "<" + self.name + ">"
            self.token = self.name

        # Extra info
        self.step = step
        self.shape = self.vector.shape
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = self.name + ".pt"

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        print("If I could save, I'd save this:\n", embedding_data)


def injectTokens(prompt, embeddings):
    """
    This code searches the given prompt for any of the given embeddings and replaces it with the proper embedding for the tokenizer

    Only necessary for multi-vector embeddings because we've split the token up per vector.

    For example, if we have a multi-vector embedding like this:
        Token: <multi-vector>
        Vectors: (3,768)

    Then, in the creation of the embedding class, we've automatically created the actual token for the tokenizer:
        Token: <multi-vector_0> <multi-vector_1> <multi-vector_3>
        Vecors: (3,768)

    So, if we find a multi-vector token, then we replace it's user-friendly token name with the actual token name. For example:

        Prompt: A picture of <multi-vector>, painted by Caravaggio

        becomes

        Prompt: A picture of <multi-vector_0> <multi-vector_1> <multi-vector_3>, painted by Caravaggio
    """
    prompt = prompt.lower()
    foundTokens = 0

    for embedding in embeddings:
        if embedding.name in prompt:
            # First, let's prepare the replacement tokens
            replacementToken = ""
            if type(embedding.token) is str:
                replacementToken = embedding.name
            elif type(embedding.token) is list:
                for token in embedding.token:
                    replacementToken = replacementToken + " " + token
            foundTokens += 1
            prompt = prompt.replace(embedding.name, replacementToken)

    keras_print("...found", foundTokens, "text embedding token(s)...")

    return prompt


def loadTextEmbedding(textEmbeddings):
    """
    Using pytorch, we load in the text embedding weights as numpy arrays and store them in the Embeddings object class

    textEmbeddings REQUIRES a list expecting the first index to have the file path. For example:

    ['models/embeddings/','myEmbedding.pt','myOtherEmbedding.bin','etc.pt']

    The code then seperates the file path as a variable and uses it to find the embeddings
    """
    finalTextEmbeddings = []
    tokensToAdd = []
    # save file path into seperate location
    embeddingsPath = textEmbeddings[0]
    # delete file path from list
    del textEmbeddings[0]

    for textEmbedding in textEmbeddings:
        print("\nLoading text embedding " + textEmbedding)
        # Load the text embedding file
        textEmbeddingFile = torch.load(
            embeddingsPath + textEmbedding, map_location="cpu"
        )

        # Debug Info
        # print("Data for",textEmbedding,"\n",textEmbeddingFile)
        # print(textEmbeddingFile.keys()) # Shows the entire file data, which should be a dictionary

        if "pt" in textEmbedding:
            # load the necessary values
            stringToToken = textEmbeddingFile[
                "string_to_token"
            ]  # Token assigned to vector
            stringToParam = textEmbeddingFile["string_to_param"]  # The vector(s)
            textEmbeddingName = textEmbedding.replace(".pt", "")
        elif "bin" in textEmbedding:
            # load the necessary values
            for key, value in textEmbeddingFile.items():
                stringToToken = key  # Token assigned to vector
                stringToParam = value  # The vector
            textEmbeddingName = textEmbedding.replace(".bin", "")

        # Save the token for finding the vector
        if type(stringToToken) is dict:
            token = list(stringToToken.keys())[
                0
            ]  # Convert dictionary to a list and then pull the first value
        else:
            token = stringToToken

        # Save the vector by finding it with the token
        if type(stringToToken) is dict:
            textEmbeddingVector = stringToParam[token]
        else:
            textEmbeddingVector = stringToParam

        # Debug info
        # print("Weight type:\n",type(textEmbeddingVector))
        # print("Vector shape:\n", textEmbeddingVector.shape)

        # Make the token lowercase
        token = textEmbeddingName.lower()
        print("Unique Token: ", "<" + token + ">")

        embedding = Embedding(name=token, vector=textEmbeddingVector.detach().numpy())
        try:
            embedding.step = textEmbeddingFile["step"]
            embedding.sd_checkpoint_name = textEmbeddingFile["sd_checkpoint_name"]
        except Exception as e:
            embedding.step = 0
            embedding.sd_checkpoint_name = "N/A"

        finalTextEmbeddings.append(embedding)

        if type(embedding.token) is str:
            tokensToAdd.append(embedding.token)
        elif type(embedding.token) is list:
            tokensToAdd.extend(embedding.token)

        # Memory Clean up
        del textEmbeddingFile

    # add file path back to list for re-compiling later, if needed
    textEmbeddings.insert(0, embeddingsPath)

    return finalTextEmbeddings, tokensToAdd


def loadTextEmbeddingWeight(textEncoder, CLIP, maxTextLength, embeddings, legacy):
    """
    This code is where the magic happens with Text Embeddings.
    We're going to add our text embeddings to the Text Encoder Model
    """
    keras_print("\nLoading Text Embedding weights...")

    if legacy == True:
        columnLength = 768
    else:
        columnLength = 1024

    # First get the current weights of the text encoder
    originalWeights = textEncoder.get_weights()

    # Find the "token_embedding" weights
    updatedWeights = originalWeights[0]
    successfulTokenCount = 0

    # Add our token vectors to the "token_embedding" weights
    for embedding in embeddings:
        if np.size(embedding.vector[0]) != columnLength:
            # if our vector column length doesn't match our version of stable diffusion, then skip this embedding
            print(
                embedding.name,
                "not compatible with current version of Stable Diffusion",
            )
            continue

        # Add our vectors to the weights for the "token_embeddings"
        updatedWeights = np.vstack((updatedWeights, embedding.vector))

        # Update our token count, taking multidimensional vectors into account
        if type(embedding.token) is list:
            successfulTokenCount += len(embedding.token)
        else:
            successfulTokenCount += 1

    keras_print(
        "...found all compatible embeddings, total:", successfulTokenCount, "..."
    )

    # Create new Text Encoder model, increasing the size of tokens for the CLIP model
    keras_print("...creating new text encoder model with embeddings")
    input_word_ids = keras.layers.Input(shape=(maxTextLength,), dtype="int32")
    input_pos_ids = keras.layers.Input(shape=(maxTextLength,), dtype="int32")
    embeds = CLIP(vocabularySize=49408 + successfulTokenCount)(
        [input_word_ids, input_pos_ids]
    )
    textEncoder = keras.models.Model([input_word_ids, input_pos_ids], embeds)
    keras_print(
        "...created text encoder model with", successfulTokenCount, "token(s) added"
    )

    # Update the weights for "token_embedding" and then set the weights of the model
    keras_print("...setting updated weights for token_embedding...")
    originalWeights[0] = updatedWeights
    textEncoder.set_weights(originalWeights)
    keras_print("...weights loaded!")

    return textEncoder
