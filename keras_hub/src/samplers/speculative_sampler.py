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
      1. **Draft phase**: `draft_next` is called K times in sequence to
         produce K candidate tokens. Nothing about this phase requires the
         draft model itself to be autoregressive: a one-shot block drafter
         (e.g. a bidirectional block-diffusion assistant) can compute all K
         candidates in a single forward pass on its first call and simply
         return the i-th cached candidate on each subsequent call.
      2. **Verify phase**: The target model scores all K+1 positions in one
         parallel forward pass via `verify_next`.
      3. **Accept phase**: Tokens are accepted/rejected via rejection sampling
         (stochastic) or greedy matching. A bonus token is always appended.

    When `base_sampler` is provided, stochastic rejection sampling is used.
    Otherwise greedy acceptance is used.

    `draft_cache` is a tuple. The sampler reads its layout from the length:
    a 2-tuple `(draft_state, target_cache)`, a 3-tuple
    `(draft_state, target_cache, anchor_pos)`, or a 4-tuple
    `(draft_state, target_cache, anchor_pos, drafter_cache)`.
    After each cycle the sampler replaces `target_cache` with the verified
    target cache. For the 3-tuple and 4-tuple forms, it also sets `anchor_pos`
    to the new cycle-start position. It replaces `draft_state` with the entry
    at position `min_accepted` of the hidden states that `verify_next`
    returns, if `verify_next` returns hidden states. The sampler never reads
    or changes `drafter_cache`. The `draft_next` and `verify_next` closures
    of any model can use this layout.

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
        if self.base_sampler is not None and hasattr(
            self.base_sampler, "temperature"
        ):
            self.base_sampler.temperature = self.temperature
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
            # token ids for the accept step.
            current_prompt = prompt
            draft_ids_list = []

            for i in range(k):
                # Clamp the index to [0, max_length - 1] for safety.
                safe_idx = ops.minimum(
                    ops.cast(index + i, "int32"), max_length - 1
                )
                logits, _, current_draft_cache = draft_next(
                    current_prompt, current_draft_cache, safe_idx
                )
                # Apply temperature before the draft argmax.
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

            # Stack: (batch, k)
            draft_ids = ops.stack(draft_ids_list, axis=1)

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
            if self.base_sampler is not None:
                target_probs = self._filter_probabilities(target_probs)
            # p_probs: (batch, k, vocab)  — the K verification distributions
            p_probs = target_probs[:, :k, :]

            # Refresh the target cache in draft_cache so the next draft cycle
            # uses the updated K/V entries. Supports any model whose
            # draft_cache is a 2-tuple `(draft_state, target_cache)`, a
            # 3-tuple `(draft_state, target_cache, anchor_pos)`, or a
            # 4-tuple `(draft_state, target_cache, anchor_pos, drafter_cache)`
            # — the tuple shape alone identifies the convention, so no
            # per-model check is needed. The optional 4th element is
            # opaque to this sampler (e.g. a drafter's own persistent
            # cache, entirely managed by that model's own `draft_next`);
            # it is carried through unchanged here.
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
            elif (
                has_draft_cache
                and has_cache
                and isinstance(draft_cache, tuple)
                and len(draft_cache) == 4
            ):
                current_draft_cache = (
                    current_draft_cache[0],
                    updated_cache,
                    current_draft_cache[2],
                    current_draft_cache[3],
                )

            # ── Phase 3: Accept / reject via rejection sampling ───────────
            if self.base_sampler is not None:
                # Stochastic acceptance (Leviathan et al., 2022 Algorithm 1).
                # The draft token is deterministic, so q is a point mass.
                # Accept token t_i if Uniform(0,1) < p(t_i) / q(t_i) = p(t_i).
                gather_idx = ops.expand_dims(
                    ops.cast(draft_ids, "int32"), axis=-1
                )
                p_target = ops.take_along_axis(p_probs, gather_idx, axis=-1)[
                    :, :, 0
                ]

                r = random.uniform(
                    shape=ops.shape(p_target),
                    seed=self.seed_generator,
                    dtype="float32",
                )
                matches = r < p_target
            else:
                # Greedy acceptance: token matches if argmax(p) == draft_id.
                target_tokens = ops.cast(
                    ops.argmax(p_probs, axis=-1), prompt.dtype
                )
                matches = ops.equal(draft_ids, target_tokens)

            # A prompt position of a longer row is forced. The draft keeps the
            # prompt token there, so the position counts as matched.
            draft_positions = ops.cast(index, "int32") + ops.arange(
                k, dtype="int32"
            )
            forced = ops.logical_and(
                ops.take(
                    ops.cast(mask, "bool"),
                    ops.minimum(draft_positions, max_length - 1),
                    axis=1,
                ),
                (draft_positions < max_length)[None, :],
            )
            matches = ops.logical_or(matches, forced)

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
                # Residual distribution for a rejected position. The draft
                # is a point mass, so the residual is p with the rejected
                # draft token removed. When all K tokens are accepted, sample
                # from p directly.
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

                safe_q_idx = ops.minimum(
                    ops.cast(min_accepted, "int32"),
                    ops.cast(k - 1, "int32"),
                )
                rejected_ids = ops.take(draft_ids, safe_q_idx, axis=1)
                rejected_onehot = ops.one_hot(
                    ops.cast(rejected_ids, "int32"), vocab, dtype=p_b.dtype
                )
                res_probs = p_b * (1.0 - rejected_onehot)
                res_sum = ops.sum(res_probs, axis=-1, keepdims=True)
                res_probs = ops.where(
                    res_sum > 0.0,
                    res_probs / ops.maximum(res_sum, 1e-30),
                    p_b,
                )

                all_accepted = ops.equal(min_accepted, ops.cast(k, "int32"))
                bonus_probs = ops.where(all_accepted, p_b, res_probs)

                # Gumbel-max trick. Zero-probability tokens never win.
                uniform_noise = random.uniform(
                    shape=ops.shape(bonus_probs),
                    seed=self.seed_generator,
                    dtype="float32",
                )
                gumbel = -ops.log(
                    -ops.log(uniform_noise + ops.cast(1e-7, "float32"))
                )
                bonus_probs = ops.cast(bonus_probs, "float32")
                log_probs = ops.where(
                    bonus_probs > 0.0,
                    ops.log(ops.maximum(bonus_probs, 1e-30)),
                    -1e30,
                )
                bonus_token = ops.argmax(log_probs + gumbel, axis=-1)
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
            # Keep prompt tokens of rows that are still inside their prompt.
            keep_existing = ops.logical_or(
                ops.logical_not(in_bounds), mask[:, safe_bonus_idx]
            )
            # Keep the accepted draft of rows that accepted more than the
            # batch minimum.
            keep_existing = ops.logical_or(
                keep_existing, num_accepted > min_accepted
            )
            bonus_token = ops.where(keep_existing, existing_bonus, bonus_token)
            final_prompt = ops.slice_update(
                final_prompt, [0, safe_bonus_idx], bonus_token[:, None]
            )

            # Advance by min_accepted + 1 (the bonus token).
            new_index = ops.minimum(
                bonus_idx + ops.cast(1, "int32"),
                ops.cast(max_length, "int32"),
            )

            # Update anchor_pos to the new cycle-start position
            # (new_index - 1) so the next cycle's draft steps share the
            # correct positional anchor. Applies to both the 3-tuple and
            # 4-tuple draft_cache forms; a 4th element (e.g. a drafter's
            # own persistent cache) is carried through unchanged.
            if (
                has_draft_cache
                and isinstance(current_draft_cache, tuple)
                and len(current_draft_cache) in (3, 4)
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
                ) + current_draft_cache[3:]

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

    def _filter_probabilities(self, probs):
        """Apply the top-k and top-p of `base_sampler` to `probs`."""
        k = getattr(self.base_sampler, "k", None)
        p = getattr(self.base_sampler, "p", None)
        if k is None and p is None:
            return probs
        vocab = probs.shape[-1]
        cutoff = vocab if k is None else min(k, vocab)
        sorted_probs, _ = ops.top_k(probs, k=cutoff, sorted=True)
        if p is None:
            keep = ops.ones_like(sorted_probs, dtype="bool")
        else:
            cumulative = ops.cumsum(sorted_probs, axis=-1)
            keep = cumulative <= p
            # Include the first token that exceeds `p`.
            keep = ops.concatenate(
                [ops.ones_like(keep[..., :1]), keep[..., :-1]], axis=-1
            )
        threshold = ops.min(
            ops.where(keep, sorted_probs, 2.0), axis=-1, keepdims=True
        )
        probs = ops.where(probs >= threshold, probs, 0.0)
        return probs / ops.sum(probs, axis=-1, keepdims=True)

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
