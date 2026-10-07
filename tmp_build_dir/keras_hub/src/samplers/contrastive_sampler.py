from keras import ops
from keras import tree

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.samplers.sampler import Sampler
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.samplers.ContrastiveSampler")
class ContrastiveSampler(Sampler):
    """Contrastive Sampler class.

    This sampler implements contrastive search algorithm. In short, the sampler
    chooses the token having the max "score" as the next token. The "score" is
    a weighted sum between token's probability and max similarity against
    previous tokens. By using this joint score, contrastive sampler reduces the
    behavior of duplicating seen tokens.

    Args:
        k: int, the `k` value of top-k. Next token will be chosen from k tokens.
        alpha: float, the weight of minus max similarity in joint score
            computation. The larger the value of `alpha`, the score relies more
            on the similarity than the token probability.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Pass by name to compile.
    causal_lm.compile(sampler="contrastive")
    causal_lm.generate(["Keras is a"])

    # Pass by object to compile.
    sampler = keras_hub.samplers.ContrastiveSampler(k=5)
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
    ```
    """

    def __init__(
        self,
        k=5,
        alpha=0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.alpha = alpha

    def __call__(
        self,
        next,
        prompt,
        cache=None,
        index=0,
        mask=None,
        stop_token_ids=None,
        hidden_states=None,
        model=None,
    ):
        if hidden_states is None:
            raise ValueError(
                "`ContrastiveSampler` requires passing a `hidden_states`, but"
                "received `None`."
            )
        batch_size, max_length = ops.shape(prompt)[0], ops.shape(prompt)[1]
        index = ops.cast(index, "int32")

        def create_beams(x):
            """Add initial beam state."""
            x = ops.repeat(x, self.k, axis=0)
            flat_shape = (batch_size * self.k,) + ops.shape(x)[1:]
            return ops.reshape(x, flat_shape)

        def flatten_beams(x):
            """Combine the beam dim and batch dim."""
            flat_shape = (batch_size * self.k,) + ops.shape(x)[2:]
            return ops.reshape(x, flat_shape)

        def unflatten_beams(x):
            """Separate the beam dim and batch dim."""
            unflat_shape = (batch_size, self.k) + ops.shape(x)[1:]
            return ops.reshape(x, unflat_shape)

        mask = ops.zeros_like(prompt, dtype="bool") if mask is None else mask
        # Compute initial logits.
        logits, _, cache = next(prompt, cache, index)
        # `ops.while_loop` will not accept `None` as a value for `loop_vars`.
        has_cache = cache is not None
        cache = cache if has_cache else ()

        def cond(prompt, cache, index, logits, hidden_states):
            if stop_token_ids is None:
                return True
            # Stop if all sequences have produced a *new* stop token.
            end_tokens = any_equal(prompt, stop_token_ids, ~mask)
            prompt_done = ops.any(end_tokens, axis=-1)
            return ops.logical_not(ops.all(prompt_done))

        def body(prompt, cache, index, logits, hidden_states):
            # Compute the softmax distribution for the next token.
            probabilities = self.compute_probabilities(logits)

            # Replicate for `self.k` times to find the best token in top-k
            # candidates.
            prompt_beams = create_beams(prompt)
            mask_beams = create_beams(mask)
            hidden_states_beams = create_beams(hidden_states)
            cache_beams = None
            if has_cache:
                cache_beams = tree.map_structure(create_beams, cache)

            # Get top-k candidate tokens and their probabilities.
            top_k_probabilities, top_k_indices = ops.top_k(
                probabilities, k=self.k, sorted=False
            )
            next_token_probabilities = flatten_beams(top_k_probabilities)
            next_token = flatten_beams(top_k_indices)
            next_token = ops.cast(next_token, prompt.dtype)
            next_token = ops.where(
                mask_beams[:, index], prompt_beams[:, index], next_token
            )

            # Update the prompt with the next token.
            next_token = ops.expand_dims(next_token, -1)
            prompt_beams = ops.slice_update(
                prompt_beams, [0, index], next_token
            )

            # Compute the logits and hidden states for top-k candidate tokens.
            next_logits, next_hidden_states_beams, cache_beams = next(
                prompt_beams, cache_beams, index + 1
            )

            # Compute the max similarity score for top-k candidate tokens
            # against previous tokens.
            similarity_scores = self.similarity(
                hidden_states_beams, next_hidden_states_beams
            )
            # Replace all future indices with -1, the lowest similarity score.
            score_mask = ops.expand_dims(ops.arange(max_length) < index, 0)
            similarity_scores = ops.where(score_mask, similarity_scores, -1)
            max_similarity_scores = ops.cast(
                ops.max(similarity_scores, axis=1),
                dtype=next_token_probabilities.dtype,
            )
            # The final score of each candidate token is weighted sum of
            # probability and similarity against previous tokens.
            accumulated_scores = (
                1 - self.alpha
            ) * next_token_probabilities - self.alpha * max_similarity_scores
            # Unflatten variables to shape [batch_size, self.k, ...] for
            # gather purpose.
            unflat_score = unflatten_beams(accumulated_scores)
            unflat_prompt = unflatten_beams(prompt_beams)
            unflat_next_logits = unflatten_beams(next_logits)
            unflat_next_hidden_states = unflatten_beams(
                next_hidden_states_beams
            )
            best_token_indices = ops.argmax(unflat_score, axis=1)

            def gather_best_token(beams):
                indices = best_token_indices
                for axis in range(1, len(beams.shape)):
                    indices = ops.expand_dims(indices, axis=axis)
                best = ops.take_along_axis(
                    beams,
                    indices,
                    axis=1,
                )
                return ops.squeeze(best, axis=1)

            prompt = gather_best_token(unflat_prompt)
            # We avoid recomputing forward pass for each token by updating the
            # cache/hidden_states using the output, and pass the logits to
            # next iteration step.
            logits = gather_best_token(unflat_next_logits)
            next_hidden_states = gather_best_token(unflat_next_hidden_states)
            if has_cache:
                cache = tree.map_structure(unflatten_beams, cache_beams)
                cache = tree.map_structure(gather_best_token, cache)

            hidden_states = ops.slice_update(
                hidden_states,
                [0, index, 0],
                next_hidden_states[:, None, :],
            )
            return (prompt, cache, index + 1, logits, hidden_states)

        prompt, _, _, _, _ = self.run_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, cache, index, logits, hidden_states),
            maximum_iterations=(max_length - index),
            model=model,
        )
        return prompt

    def similarity(self, h1, h2):
        h2 = ops.expand_dims(h2, -1)
        h1_norm = ops.sqrt(ops.sum(h1 * h1, axis=-1))
        h2_norm = ops.sqrt(ops.sum(h2 * h2, axis=-2))
        return ops.squeeze(ops.matmul(h1, h2), axis=-1) / (h1_norm * h2_norm)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "alpha": self.alpha,
            }
        )
        return config
