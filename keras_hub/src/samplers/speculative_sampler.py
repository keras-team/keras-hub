from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.samplers.sampler import Sampler
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.samplers.SpeculativeSampler")
class SpeculativeSampler(Sampler):
    """Speculative decoding sampler.

    Accelerates autoregressive inference by running a small draft model to
    propose K candidate tokens, then verifying all K+1 positions with the
    large target model in a single parallel forward pass.

    References:
    - [Fast Inference from Transformers via Speculative Decoding]
        (https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022)

    The algorithm follows three phases per outer iteration:
      1. **Draft phase**: The draft model auto-regressively generates K tokens.
      2. **Verify phase**: The target model scores all K+1 positions in one
         parallel forward pass via `verify_next`.
      3. **Accept phase**: Tokens are accepted/rejected via rejection sampling
         (stochastic) or greedy matching. A bonus token is always appended.

    When `base_sampler` is provided, stochastic rejection sampling is used.
    Otherwise greedy acceptance is used.

    Args:
        num_speculative_tokens: int. Number of draft tokens per outer
            iteration. Defaults to `5`.
        base_sampler: optional `keras_hub.samplers.Sampler`. When set,
            enables stochastic rejection sampling using this sampler's
            temperature, top-k, and top-p parameters. Defaults to `None`
            (greedy acceptance).
        seed: optional int. Random seed for stochastic sampling.

    Call arguments:
        next: callable `(prompt, cache, index) → (logits, hidden, cache)`.
            Target model single-token forward pass (used as fallback when
            `verify_next` is not provided).
        prompt: int tensor `(batch, max_length)`. Running token sequence.
        cache: optional cache tensor.
        index: int. Current generation position.
        mask: bool tensor `(batch, max_length)`. True at prompt positions.
        stop_token_ids: optional sequence of int stop-token ids.
        model: optional Keras model (needed for JAX stateless scopes).
        draft_next: **required** callable
            `(prompt, cache, index) → (logits, hidden, cache)`.
            Draft model single-token forward pass.
        draft_cache: optional draft model cache.
        verify_next: optional callable
            `(prompt, cache, index, k) → (logits, cache)` or
            `(prompt, cache, index, k) → (logits, hidden_states, cache)`.
            If provided, the target model verifies K+1 positions in a single
            parallel forward pass (strongly recommended for performance).
    """

    def __init__(
        self,
        num_speculative_tokens=5,
        base_sampler=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_speculative_tokens = num_speculative_tokens
        self.base_sampler = base_sampler
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
                "`SpeculativeSampler` requires a `draft_next` callable."
            )

        batch_size = ops.shape(prompt)[0]
        max_length = ops.shape(prompt)[-1]
        index = ops.cast(index, "int32")
        k = self.num_speculative_tokens

        if mask is None:
            mask = ops.zeros_like(prompt, dtype="bool")
        mask = ops.cast(mask, "bool")

        has_cache = cache is not None
        has_draft_cache = draft_cache is not None
        cache = cache if has_cache else ()
        draft_cache = draft_cache if has_draft_cache else ()

        def cond(prompt, cache, draft_cache, index):
            if stop_token_ids is None:
                return index < max_length
            end_tokens = any_equal(prompt, stop_token_ids, ~mask)
            prompt_done = ops.any(end_tokens, axis=-1)
            return ops.logical_and(
                ops.logical_not(ops.all(prompt_done)),
                index < max_length,
            )

        def body(prompt, cache, draft_cache, index):
            current_draft_cache = draft_cache if has_draft_cache else None

            # ── Phase 1: Draft K tokens ───────────────────────────────────
            # Run the draft model K times autoregressively. Collect draft
            # token ids and probabilities for the accept step.
            current_prompt = prompt
            draft_ids_list = []
            draft_probs_list = []

            for i in range(k):
                # Clamp the index to [0, max_length - 1] for safety.
                safe_idx = ops.minimum(
                    ops.cast(index + i, "int32"), max_length - 1
                )
                logits, _, current_draft_cache = draft_next(
                    current_prompt, current_draft_cache, safe_idx
                )
                # Apply temperature to get q_probs for the acceptance ratio
                # p(x)/q(x).  Temperature comes from base_sampler when set.
                if self.base_sampler is not None:
                    dt = getattr(self.base_sampler, "temperature", 1.0)
                    probs = ops.softmax(
                        ops.cast(logits, "float32") / ops.cast(dt, "float32")
                    )
                else:
                    probs = self.compute_probabilities(logits)

                # Draft token: always greedy (argmax). This is correct for
                # both sparse-vocab (MTP) and dense-vocab assistants.
                next_token = ops.argmax(probs, axis=-1)

                next_token = ops.cast(next_token, prompt.dtype)

                # Respect existing prompt tokens (mask).
                mask_val = mask[:, safe_idx]
                existing = current_prompt[:, safe_idx]
                in_bounds = ops.cast((index + i) < max_length, next_token.dtype)
                next_token = ops.where(mask_val, existing, next_token)
                next_token = ops.where(
                    ops.cast(in_bounds, "bool"), next_token, existing
                )

                current_prompt = ops.slice_update(
                    current_prompt, [0, safe_idx], next_token[:, None]
                )
                draft_ids_list.append(next_token)
                draft_probs_list.append(probs)

            # Stack: (batch, k), (batch, k, vocab)
            draft_ids = ops.stack(draft_ids_list, axis=1)
            q_probs = ops.stack(draft_probs_list, axis=1)

            # ── Phase 2: Verify with target model ─────────────────────────
            verify_hidden_states = None
            if verify_next is not None:
                # Single parallel forward pass covering positions
                # [index-1 .. index+k-1] (k+1 positions total).
                # verify_next may return (logits, cache) or
                # (logits, hidden_states, cache) — handle both.
                _verify_result = verify_next(
                    current_prompt,
                    cache if has_cache else None,
                    index,
                    k,
                )
                if len(_verify_result) == 3:
                    target_logits, verify_hidden_states, updated_cache = (
                        _verify_result
                    )
                else:
                    target_logits, updated_cache = _verify_result
            else:
                # Fallback: serial verification (slower).
                logits_list = []
                current_cache = cache if has_cache else None
                for i in range(k + 1):
                    safe_idx = ops.minimum(
                        ops.cast(index + i, "int32"), max_length - 1
                    )
                    logits, _, current_cache = next(
                        current_prompt, current_cache, safe_idx
                    )
                    logits_list.append(logits)
                target_logits = ops.stack(logits_list, axis=1)
                updated_cache = current_cache

            # target_logits: (batch, k+1, vocab)
            target_probs = self.compute_probabilities(target_logits)
            # p_probs: (batch, k, vocab)  — the K verification distributions
            p_probs = target_probs[:, :k, :]

            # Refresh the target cache in draft_cache so the next draft cycle
            # uses the updated K/V entries. Supports 2-tuple and 3-tuple forms.
            if (
                has_draft_cache
                and has_cache
                and isinstance(draft_cache, tuple)
                and len(draft_cache) == 2
            ):
                current_draft_cache = (current_draft_cache[0], updated_cache)
            elif (
                has_draft_cache
                and has_cache
                and isinstance(draft_cache, tuple)
                and len(draft_cache) == 3
            ):
                current_draft_cache = (
                    current_draft_cache[0],
                    updated_cache,
                    current_draft_cache[2],
                )

            # ── Phase 3: Accept / reject via rejection sampling ───────────
            if self.base_sampler is not None:
                # Stochastic acceptance (Leviathan et al., 2022 Algorithm 1).
                # For each draft position i:
                #   Accept token t_i if Uniform(0,1) < p(t_i) / q(t_i).
                gather_idx = ops.expand_dims(
                    ops.cast(draft_ids, "int32"), axis=-1
                )
                p_target = ops.take_along_axis(p_probs, gather_idx, axis=-1)[
                    :, :, 0
                ]
                q_draft = ops.take_along_axis(q_probs, gather_idx, axis=-1)[
                    :, :, 0
                ]

                r = random.uniform(
                    shape=ops.shape(p_target),
                    seed=self.seed_generator,
                    dtype="float32",
                )
                matches = (r * q_draft) < p_target
            else:
                # Greedy acceptance: token matches if argmax(p) == draft_id.
                target_tokens = ops.cast(
                    ops.argmax(p_probs, axis=-1), prompt.dtype
                )
                matches = ops.equal(draft_ids, target_tokens)

            # Compute the prefix of accepted tokens (all must match up to i).
            # num_accepted[b] = length of the accepted prefix for item b.
            matches_int = ops.cast(matches, "int32")
            prefix_mask = ops.cumprod(matches_int, axis=1)
            num_accepted = ops.sum(prefix_mask, axis=1)  # (batch,)

            # Write accepted draft tokens back into the output prompt.
            final_prompt = prompt
            for i in range(k):
                safe_idx = ops.minimum(
                    ops.cast(index + i, "int32"), max_length - 1
                )
                is_accepted = ops.logical_and(
                    ops.cast(i, "int32") < num_accepted,
                    (index + i) < max_length,
                )
                token = ops.where(
                    is_accepted,
                    draft_ids[:, i],
                    final_prompt[:, safe_idx],
                )
                final_prompt = ops.slice_update(
                    final_prompt, [0, safe_idx], token[:, None]
                )

            # ── Bonus token (always appended) ─────────────────────────────
            # Use the minimum accepted count across the batch so we advance
            # the index by a common amount.
            min_accepted = ops.min(num_accepted)  # scalar int
            bonus_idx = ops.cast(index + min_accepted, "int32")
            safe_bonus_idx = ops.minimum(bonus_idx, max_length - 1)

            if self.base_sampler is not None:
                # Residual distribution for rejected positions:
                #   p_bonus = max(0, p_target - q_draft) / Z
                # When all K tokens were accepted, sample from p_target
                # directly (no residual needed).
                # Use ops.gather with scalar min_accepted to stay in-graph.
                gather_bonus = ops.expand_dims(
                    ops.broadcast_to(
                        ops.expand_dims(min_accepted, axis=0),
                        (batch_size,),
                    ),
                    axis=-1,
                )  # (batch, 1)
                gather_bonus = ops.expand_dims(
                    gather_bonus, axis=-1
                )  # (batch, 1, 1)
                vocab = ops.shape(target_probs)[-1]
                gather_bonus_b = ops.broadcast_to(
                    gather_bonus, (batch_size, 1, vocab)
                )

                p_b = ops.take_along_axis(target_probs, gather_bonus_b, axis=1)[
                    :, 0, :
                ]

                # Clamp to avoid negative probabilities from floating-point
                # error.
                safe_q_idx = ops.minimum(
                    ops.cast(min_accepted, "int32"),
                    ops.cast(k - 1, "int32"),
                )
                gather_q = ops.expand_dims(
                    ops.broadcast_to(
                        ops.expand_dims(safe_q_idx, axis=0), (batch_size,)
                    ),
                    axis=-1,
                )
                gather_q = ops.expand_dims(gather_q, axis=-1)
                gather_q_b = ops.broadcast_to(gather_q, (batch_size, 1, vocab))
                q_b = ops.take_along_axis(q_probs, gather_q_b, axis=1)[:, 0, :]

                all_accepted = ops.equal(min_accepted, ops.cast(k, "int32"))
                res_probs = ops.maximum(
                    ops.cast(0.0, p_b.dtype),
                    ops.cast(p_b, p_b.dtype) - ops.cast(q_b, p_b.dtype),
                )
                res_sum = ops.sum(res_probs, axis=-1, keepdims=True) + ops.cast(
                    1e-7, res_probs.dtype
                )
                res_probs = res_probs / res_sum

                # When all K tokens accepted, fall through to p_b directly.
                # all_accepted is a scalar bool; ops.where broadcasts it
                # over (batch, vocab) without needing explicit indexing.
                bonus_probs = ops.where(all_accepted, p_b, res_probs)

                # Gumbel-max trick for differentiable argmax sampling.
                uniform_noise = random.uniform(
                    shape=ops.shape(bonus_probs),
                    seed=self.seed_generator,
                    dtype="float32",
                )
                gumbel = -ops.log(
                    -ops.log(uniform_noise + ops.cast(1e-7, "float32"))
                )
                bonus_token = ops.argmax(
                    ops.log(
                        ops.cast(bonus_probs, "float32")
                        + ops.cast(1e-7, "float32")
                    )
                    + gumbel,
                    axis=-1,
                )
            else:
                # Greedy bonus: argmax of the target distribution at position
                # min_accepted.
                gather_bonus = ops.expand_dims(
                    ops.broadcast_to(
                        ops.expand_dims(min_accepted, axis=0),
                        (batch_size,),
                    ),
                    axis=-1,
                )
                gather_bonus = ops.expand_dims(gather_bonus, axis=-1)
                vocab = ops.shape(target_probs)[-1]
                gather_bonus_b = ops.broadcast_to(
                    gather_bonus, (batch_size, 1, vocab)
                )
                bonus_probs = ops.take_along_axis(
                    target_probs, gather_bonus_b, axis=1
                )[:, 0, :]
                bonus_token = ops.argmax(bonus_probs, axis=-1)

            bonus_token = ops.cast(bonus_token, prompt.dtype)
            existing_bonus = final_prompt[:, safe_bonus_idx]
            in_bounds = ops.cast(bonus_idx < max_length, "bool")
            bonus_token = ops.where(in_bounds, bonus_token, existing_bonus)
            final_prompt = ops.slice_update(
                final_prompt, [0, safe_bonus_idx], bonus_token[:, None]
            )

            # Advance by min_accepted + 1 (the bonus token).
            new_index = ops.minimum(
                bonus_idx + ops.cast(1, "int32"),
                ops.cast(max_length, "int32"),
            )

            # Update fixed_pos to the new cycle-start position (new_index - 1)
            # so the next cycle's draft steps share the correct RoPE anchor.
            if (
                has_draft_cache
                and isinstance(current_draft_cache, tuple)
                and len(current_draft_cache) == 3
            ):
                # Seed the next draft cycle with the target's hidden state
                # at the accepted position:
                # verify_hidden_states[:, min_accepted, :].
                if verify_hidden_states is not None:
                    h_dim = ops.shape(verify_hidden_states)[2]
                    new_seed_hidden = ops.slice(
                        verify_hidden_states,
                        [0, ops.cast(min_accepted, "int32"), 0],
                        [batch_size, 1, h_dim],
                    )
                else:
                    new_seed_hidden = current_draft_cache[0]
                current_draft_cache = (
                    new_seed_hidden,
                    current_draft_cache[1],
                    new_index - ops.cast(1, "int32"),
                )

            return (
                final_prompt,
                updated_cache if has_cache else cache,
                (current_draft_cache if has_draft_cache else draft_cache),
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
        base_sampler_config = None
        if self.base_sampler is not None:
            from keras_hub.src.samplers.serialization import serialize

            base_sampler_config = serialize(self.base_sampler)
        config = super().get_config()
        config.update(
            {
                "num_speculative_tokens": self.num_speculative_tokens,
                "base_sampler": base_sampler_config,
                "seed": self.seed,
            }
        )
        return config
