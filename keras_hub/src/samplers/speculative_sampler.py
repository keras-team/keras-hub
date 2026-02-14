from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.samplers.sampler import Sampler
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.samplers.SpeculativeSampler")
class SpeculativeSampler(Sampler):
    """Speculative Sampler class.

    This sampler implements speculative decoding for accelerated inference.
    A smaller draft model generates candidate tokens that are verified by
    the target model in parallel. When draft tokens match target predictions,
    multiple tokens can be accepted per iteration, providing speedup.

    The algorithm proceeds in three phases per iteration:
    1. Draft phase: Generate K tokens using the draft model (serial, cheap)
    2. Verify phase: Check all K tokens with target model (parallel, expensive)
    3. Accept phase: Accept matching prefix and add one bonus token

    Args:
        num_speculative_tokens: int. Number of tokens to speculatively generate
            per iteration. Defaults to `5`.
        draft_temperature: float. Temperature for draft model sampling.
            Defaults to `1.0`.
        seed: int. The random seed. Defaults to `None`.

    Call arguments:
        {{call_args}}
        draft_next: callable. Draft model's next token function with signature
            `(prompt, cache, index) -> (logits, hidden_states, cache)`.
        draft_cache: Initial cache for the draft model.
        verify_next: callable. Optional batch verification function with
            signature `(prompt, cache, index, k) -> (logits, cache)` where
            logits has shape `(batch, k+1, vocab)`. Required for speedup.

    Examples:
    ```python
    causal_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")

    # Pass by object to compile.
    sampler = keras_hub.samplers.SpeculativeSampler(num_speculative_tokens=5)
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
    ```
    """

    def __init__(
        self,
        num_speculative_tokens=5,
        draft_temperature=1.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_speculative_tokens = num_speculative_tokens
        self.draft_temperature = draft_temperature
        self.seed = seed
        self.seed_generator = random.SeedGenerator(seed)

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
        draft_next=None,
        draft_cache=None,
        verify_next=None,
    ):
        if draft_next is None:
            raise ValueError(
                "`SpeculativeSampler` requires `draft_next` but received "
                "`None`."
            )

        batch_size = ops.shape(prompt)[0]
        max_length = ops.shape(prompt)[-1]
        index = ops.cast(index, "int32")
        max_length = ops.cast(max_length, "int32")
        k = self.num_speculative_tokens

        mask = ops.zeros_like(prompt, dtype="bool") if mask is None else mask
        mask = ops.cast(mask, dtype="bool")

        # `ops.while_loop` will not accept `None` as a value for `loop_vars`.
        has_cache = cache is not None
        has_draft_cache = draft_cache is not None
        cache = cache if has_cache else ()
        draft_cache = draft_cache if has_draft_cache else ()

        def cond(prompt, cache, draft_cache, index):
            if stop_token_ids is None:
                return index < max_length
            end_tokens = any_equal(prompt, stop_token_ids, ~mask)
            prompt_done = ops.any(end_tokens, axis=-1)
            still_going = ops.logical_not(ops.all(prompt_done))
            return ops.logical_and(still_going, index < max_length)

        def body(prompt, cache, draft_cache, index):
            # Phase 1: Draft K tokens serially.
            draft_ids = []
            current_prompt = prompt
            current_draft_cache = draft_cache if has_draft_cache else None

            for i in range(k):
                actual_idx = index + i
                safe_idx = ops.minimum(actual_idx, max_length - 1)
                logits, _, current_draft_cache = draft_next(
                    current_prompt, current_draft_cache, safe_idx
                )
                probs = self.compute_probabilities(logits)
                probs = ops.cast(probs, "float32") / self.draft_temperature
                next_token = ops.argmax(probs, axis=-1)
                next_token = ops.cast(next_token, current_prompt.dtype)

                # Handle masked positions.
                in_bounds = actual_idx < max_length
                mask_val = ops.take(mask, [safe_idx], axis=1)[:, 0]
                prompt_val = ops.take(current_prompt, [safe_idx], axis=1)[:, 0]
                next_token = ops.where(mask_val, prompt_val, next_token)
                next_token = ops.where(in_bounds, next_token, prompt_val)

                current_prompt = ops.slice_update(
                    current_prompt, [0, safe_idx], next_token[:, None]
                )
                draft_ids.append(next_token)

            draft_ids = ops.stack(draft_ids, axis=1)
            prompt_with_drafts = current_prompt

            # Phase 2: Verify with target model.
            if verify_next is not None:
                target_logits, updated_cache = verify_next(
                    prompt_with_drafts, cache if has_cache else None, index, k
                )
            else:
                # Fallback to serial verification (no speedup).
                logits_list = []
                current_cache = cache if has_cache else None
                for i in range(k + 1):
                    safe_idx = ops.minimum(index + i, max_length - 1)
                    logits, _, current_cache = next(
                        prompt_with_drafts, current_cache, safe_idx
                    )
                    logits_list.append(logits)
                target_logits = ops.stack(logits_list, axis=1)
                updated_cache = current_cache

            # Phase 3: Compute matching prefix.
            target_probs = self.compute_probabilities(target_logits)
            target_tokens = ops.argmax(target_probs[:, :k, :], axis=-1)
            target_tokens = ops.cast(target_tokens, prompt.dtype)

            matches = ops.equal(draft_ids, target_tokens)
            matches_int = ops.cast(matches, "int32")
            prefix_mask = ops.cumprod(matches_int, axis=1)
            num_accepted = ops.sum(prefix_mask, axis=1)

            # Update prompt with accepted tokens.
            final_prompt = prompt
            for i in range(k):
                safe_idx = ops.minimum(index + i, max_length - 1)
                in_bounds = (index + i) < max_length
                position = ops.full((batch_size,), i, dtype="int32")
                is_accepted = ops.logical_and(
                    position < num_accepted, in_bounds
                )

                draft_token = draft_ids[:, i]
                mask_val = ops.take(mask, [safe_idx], axis=1)[:, 0]
                existing = ops.take(final_prompt, [safe_idx], axis=1)[:, 0]

                token = ops.where(is_accepted, draft_token, existing)
                token = ops.where(mask_val, existing, token)
                token = ops.cast(token, prompt.dtype)
                final_prompt = ops.slice_update(
                    final_prompt, [0, safe_idx], token[:, None]
                )

            # Add bonus token from target model.
            safe_num = ops.minimum(num_accepted, k)
            gather_idx = safe_num[:, None, None]
            gather_idx = ops.broadcast_to(
                gather_idx, (batch_size, 1, ops.shape(target_probs)[-1])
            )
            bonus_probs = ops.take_along_axis(target_probs, gather_idx, axis=1)[
                :, 0, :
            ]
            bonus_token = ops.argmax(bonus_probs, axis=-1)
            bonus_token = ops.cast(bonus_token, prompt.dtype)

            min_accepted = ops.min(num_accepted)
            bonus_idx = index + min_accepted
            safe_bonus_idx = ops.minimum(bonus_idx, max_length - 1)
            in_bounds = bonus_idx < max_length

            mask_val = ops.take(mask, [safe_bonus_idx], axis=1)[:, 0]
            existing = ops.take(final_prompt, [safe_bonus_idx], axis=1)[:, 0]
            bonus_token = ops.where(in_bounds, bonus_token, existing)
            bonus_token = ops.where(mask_val, existing, bonus_token)
            final_prompt = ops.slice_update(
                final_prompt, [0, safe_bonus_idx], bonus_token[:, None]
            )

            new_index = ops.minimum(index + min_accepted + 1, max_length)

            return (
                final_prompt,
                updated_cache if has_cache else cache,
                current_draft_cache if has_draft_cache else draft_cache,
                new_index,
            )

        prompt, _, _, _ = self.run_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, cache, draft_cache, index),
            maximum_iterations=(max_length - index),
            model=model,
        )
        return prompt

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_speculative_tokens": self.num_speculative_tokens,
                "draft_temperature": self.draft_temperature,
                "seed": self.seed,
            }
        )
        return config
