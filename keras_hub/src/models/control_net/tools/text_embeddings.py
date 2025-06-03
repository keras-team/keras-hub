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
        # This is for single vector text embeddings, which may come as
        # a (768,) instead of (1,768)
        if self.vector.ndim < 2:
            if self.vector.shape[0] == 768:  # Stable Diffusion 1.4/1.5
                self.vector = self.vector.reshape((1, 768))
            elif self.vector.shape[0] == 1024:  # Stable Diffusion 2.x
                self.vector = self.vector.reshape((1, 1024))

        # Create the unique tokens
        if self.vector.shape[0] > 1:
            # If we have a multidimensional vector, then we'll split up the
            # token per dimension
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


def inject_tokens(prompt, embeddings):
    """
    This code searches the given prompt for any of the given embeddings and
     replaces it with the proper embedding for the tokenizer

    Only necessary for multi-vector embeddings because we've split the token
     up per vector.

    For example, if we have a multi-vector embedding like this:
        Token: <multi-vector>
        Vectors: (3,768)

    Then, in the creation of the embedding class, we've automatically created
     the actual token for the tokenizer:
        Token: <multi-vector_0> <multi-vector_1> <multi-vector_3>
        Vecors: (3,768)

    So, if we find a multi-vector token, then we replace it's user-friendly
    token name with the actual token name. For example:

        Prompt: A picture of <multi-vector>, painted by Caravaggio

        becomes

        Prompt:
        A picture of <multi-vector_0> <multi-vector_1> <multi-vector_3>
        , painted by Caravaggio
    """
    prompt = prompt.lower()
    found_tokens = 0

    for embedding in embeddings:
        if embedding.name in prompt:
            # First, let's prepare the replacement tokens
            replacement_token = ""
            if isinstance(embedding.token, str):
                replacement_token = embedding.name
            elif isinstance(embedding.token, list):
                for token in embedding.token:
                    replacement_token = replacement_token + " " + token
            found_tokens += 1
            prompt = prompt.replace(embedding.name, replacement_token)

    keras_print("...found", found_tokens, "text embedding token(s)...")

    return prompt


def load_text_embedding(text_embeddings):
    """
    Using pytorch, we load in the text embedding weights as numpy arrays and
    store them in the Embeddings object class

    textEmbeddings REQUIRES a list expecting the first index to have the file
    path. For example:

    ['models/embeddings/','myEmbedding.pt','myOtherEmbedding.bin','etc.pt']

    The code then seperates the file path as a variable and uses it to find the
    embeddings
    """
    final_text_embeddings = []
    tokens_to_add = []
    # save file path into seperate location
    embeddings_path = text_embeddings[0]
    # delete file path from list
    del text_embeddings[0]

    for text_embedding in text_embeddings:
        print("\nLoading text embedding " + text_embedding)
        # Load the text embedding file
        text_embedding_file = torch.load(
            embeddings_path + text_embedding, map_location="cpu"
        )

        # Debug Info
        # print("Data for",text_embedding,"\n",text_embedding_file)
        # print(text_embedding_file.keys())
        # ^Shows the entire file data, which should be a dictionary

        if "pt" in text_embedding:
            # load the necessary values
            string_to_token = text_embedding_file[
                "string_to_token"
            ]  # Token assigned to vector
            string_to_param = text_embedding_file[
                "string_to_param"
            ]  # The vector(s)
            text_embedding_name = text_embedding.replace(".pt", "")
        elif "bin" in text_embedding:
            # load the necessary values
            for key, value in text_embedding_file.items():
                string_to_token = key  # Token assigned to vector
                string_to_param = value  # The vector
            text_embedding_name = text_embedding.replace(".bin", "")

        # Save the token for finding the vector
        if isinstance(string_to_token, dict):
            token = list(string_to_token.keys())[
                0
            ]  # Convert dictionary to a list and then pull the first value
        else:
            token = string_to_token

        # Save the vector by finding it with the token
        if isinstance(string_to_token, dict):
            text_embedding_vector = string_to_param[token]
        else:
            text_embedding_vector = string_to_param

        # Debug info
        # print("Weight type:\n",type(text_embedding_vector))
        # print("Vector shape:\n", text_embedding_vector.shape)

        # Make the token lowercase
        token = text_embedding_name.lower()
        print("Unique Token: ", "<" + token + ">")

        embedding = Embedding(
            name=token, vector=text_embedding_vector.detach().numpy()
        )
        try:
            embedding.step = text_embedding_file["step"]
            embedding.sd_checkpoint_name = text_embedding_file[
                "sd_checkpoint_name"
            ]
        except Exception:
            embedding.step = 0
            embedding.sd_checkpoint_name = "N/A"

        final_text_embeddings.append(embedding)

        if isinstance(embedding.token, str):
            tokens_to_add.append(embedding.token)
        elif isinstance(embedding.token, list):
            tokens_to_add.extend(embedding.token)

        # Memory Clean up
        del text_embedding_file

    # add file path back to list for re-compiling later, if needed
    text_embeddings.insert(0, embeddings_path)

    return final_text_embeddings, tokens_to_add


def load_text_embedding_weight(
    text_encoder, CLIP, max_text_length, embeddings, legacy
):
    """
    This code is where the magic happens with Text Embeddings.
    We're going to add our text embeddings to the Text Encoder Model
    """
    keras_print("\nLoading Text Embedding weights...")

    if legacy is True:
        column_length = 768
    else:
        column_length = 1024

    # First get the current weights of the text encoder
    original_weights = text_encoder.get_weights()

    # Find the "token_embedding" weights
    updated_weights = original_weights[0]
    successful_token_count = 0

    # Add our token vectors to the "token_embedding" weights
    for embedding in embeddings:
        if np.size(embedding.vector[0]) != column_length:
            # skip if our vector column length doesn't match our version of
            # stable diffusion, then skip this embedding
            print(
                embedding.name,
                "not compatible with current version of Stable Diffusion",
            )
            continue

        # Add our vectors to the weights for the "token_embeddings"
        updated_weights = np.vstack((updated_weights, embedding.vector))

        # Update our token count, taking multidimensional vectors into account
        if isinstance(embedding.token, list):
            successful_token_count += len(embedding.token)
        else:
            successful_token_count += 1

    keras_print(
        "...found all compatible embeddings, total:",
        successful_token_count,
        "...",
    )

    # Create new Text Encoder model, incr. the size of tokens for CLIP model
    keras_print("...creating new text encoder model with embeddings")
    input_word_ids = keras.layers.Input(shape=(max_text_length,), dtype="int32")
    input_pos_ids = keras.layers.Input(shape=(max_text_length,), dtype="int32")
    embeds = CLIP(vocabularySize=49408 + successful_token_count)(
        [input_word_ids, input_pos_ids]
    )
    text_encoder = keras.models.Model([input_word_ids, input_pos_ids], embeds)
    keras_print(
        "...created text encoder model with",
        successful_token_count,
        "token(s) added",
    )

    # Update weights for "token_embedding" & then set the weights of the model
    keras_print("...setting updated weights for token_embedding...")
    original_weights[0] = updated_weights
    text_encoder.set_weights(original_weights)
    keras_print("...weights loaded!")

    return text_encoder
