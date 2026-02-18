import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# PaliGemma EOS token ID (same as in pi0_fast.py)
PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.discrete_state_input = config.discrete_state_input # 是否使用离散状态输入
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))   # 第一个“动作序列”的起点为True，其他都是False（双向），意味着将后续的49个Action Token视为与第一个Token同一级的“并列块”
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def generate_subtask(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_gen_steps: int = 50,
        temperature: float = 0.0,
    ) -> jnp.ndarray:
        """Stage 1 of π0.5 inference: generate subtask text via VLM autoregressive decoding.

        Uses only the PaliGemma VLM (expert 0) to autoregressively generate subtask tokens.
        The Action Expert (expert 1) is NOT used in this stage.

        The observation.tokenized_prompt should already be formatted as the subtask generation
        prefix, e.g. "Task: clean the bedroom. Subtask: " via tokenize_subtask_prompt().

        Args:
            rng: JAX random key (used when temperature > 0).
            observation: Observation with images and tokenized subtask prompt.
            max_gen_steps: Maximum number of tokens to generate.
            temperature: Sampling temperature. 0 = greedy (argmax).

        Returns:
            generated_tokens: int32[batch, max_gen_steps] — generated token IDs.
                Contains actual tokens up to EOS, then zeros (padding).

        Data flow:
            1. embed_prefix() → image_tokens + prompt_tokens → prefix_tokens [batch, ~968, 2048]
            2. VLM forward pass [prefix_tokens, None] → prefix_out [batch, ~968, 2048] + kv_cache
            3. decode_to_logits(prefix_out[:, -1:, :]) → logits [batch, 1, 257152]
            4. Sample token from logits → int32[batch]
            5. Embed token → [batch, 1, 2048]
            6. VLM forward with kv_cache → new hidden [batch, 1, 2048] + updated kv_cache
            7. Repeat 3-6 until EOS or max_gen_steps
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        # Step 1-2: Forward pass the prefix through VLM only
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # Run only VLM (expert 0), skip Action Expert (expert 1 = None)
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        # prefix_out: [batch, prefix_len_padded, 2048] — VLM hidden states after final norm

        # Track actual valid prefix length per batch item
        prefix_len = jnp.sum(prefix_mask, axis=-1)  # [batch] — count of True values

        # Step 3: Get logits from last VALID position (not absolute last, which may be padding)
        # For each batch item, the last valid position index is prefix_len[b] - 1
        last_valid_idx = prefix_len - 1  # [batch]
        last_hidden = prefix_out[jnp.arange(batch_size), last_valid_idx, :]  # [batch, 2048]
        last_hidden = last_hidden[:, None, :]  # [batch, 1, 2048]
        logits = self.PaliGemma.llm(last_hidden, method="decode_to_logits")  # [batch, 1, vocab_size]
        last_logit = logits[:, 0, :]  # [batch, vocab_size]

        # Track generated tokens
        generated_tokens = jnp.zeros((batch_size, max_gen_steps), dtype=jnp.int32)

        # Autoregressive generation loop (Python for-loop, since KV cache shape changes)
        for step_i in range(max_gen_steps):
            # Step 4: Sample next token
            if temperature > 0.0:
                rng, step_rng = jax.random.split(rng)
                token = jax.random.categorical(step_rng, last_logit / temperature, axis=-1)  # [batch]
            else:
                token = jnp.argmax(last_logit, axis=-1)  # [batch]

            # Store generated token
            generated_tokens = generated_tokens.at[:, step_i].set(token)

            # Check for EOS (all samples in batch)
            all_eos = jnp.all(token == PALIGEMMA_EOS_TOKEN)
            if all_eos:
                break

            # Step 5: Embed the sampled token
            token_2d = token[:, None]  # [batch, 1]
            token_emb = self.PaliGemma.llm(token_2d, method="embed")  # [batch, 1, 2048]

            # Step 6: Forward pass with KV cache
            # Token at step_i has position = prefix_len + step_i
            # (last valid prefix position was prefix_len - 1, so first gen token is at prefix_len)
            next_pos = (prefix_len + step_i)[:, None]  # [batch, 1]
            # Mask: the new token can attend to all valid previous tokens + itself
            # kv_cache is (k_stacked, v_stacked) with shape [layers, batch, seq_len, kv_heads, head_dim]
            kv_seq_len = kv_cache[0].shape[2]
            gen_mask = jnp.ones((batch_size, 1, kv_seq_len + 1), dtype=jnp.bool_)
            # Zero out attention to padding positions in original prefix
            prefix_pad_mask = jnp.concatenate(
                [prefix_mask, jnp.ones((batch_size, step_i + 1), dtype=jnp.bool_)], axis=-1
            )
            gen_mask = gen_mask * prefix_pad_mask[:, None, :]

            (vlm_out, _), kv_cache = self.PaliGemma.llm(
                [token_emb, None], mask=gen_mask, positions=next_pos, kv_cache=kv_cache
            )
            # vlm_out: [batch, 1, 2048]

            # Get logits for next step
            logits = self.PaliGemma.llm(vlm_out, method="decode_to_logits")  # [batch, 1, vocab_size]
            last_logit = logits[:, -1, :]  # [batch, vocab_size]

        return generated_tokens

    def sample_actions_with_subtask(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        high_level_prompt: str,
        tokenizer: _tokenizer.PaligemmaTokenizer,
        max_gen_steps: int = 50,
        temperature: float = 0.0,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, str]:
        """Two-stage π0.5 inference: generate subtask text, then generate actions.

        Stage 1: Use VLM to autoregressively generate a subtask from the high-level prompt.
        Stage 2: Use the generated subtask as the prompt for standard action generation.

        Args:
            rng: JAX random key.
            observation: Observation with images and state.
            high_level_prompt: High-level task instruction, e.g. "clean the bedroom".
            tokenizer: PaligemmaTokenizer instance for tokenization/detokenization.
            max_gen_steps: Max tokens to generate for subtask.
            temperature: Sampling temperature for subtask generation.
            num_steps: Number of flow matching denoising steps.
            noise: Optional initial noise for action generation.

        Returns:
            (actions, subtask_text):
                actions: float32[batch, action_horizon, action_dim] — generated actions.
                subtask_text: str — the generated subtask text, e.g. "pick up the pillow".
        """
        batch_size = observation.state.shape[0]
        rng, subtask_rng, action_rng = jax.random.split(rng, 3)

        # ====== Stage 1: Generate subtask ======
        # Tokenize the high-level prompt for subtask generation
        subtask_prompt_tokens, subtask_prompt_mask = tokenizer.tokenize_subtask_prompt(high_level_prompt)
        # Add batch dimension: [max_token_len] → [batch, max_token_len]
        subtask_prompt_tokens = jnp.broadcast_to(
            jnp.asarray(subtask_prompt_tokens)[None, :], (batch_size, subtask_prompt_tokens.shape[0])
        )
        subtask_prompt_mask = jnp.broadcast_to(
            jnp.asarray(subtask_prompt_mask)[None, :], (batch_size, subtask_prompt_mask.shape[0])
        )

        # Create a temporary observation with the subtask generation prompt
        subtask_obs = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=subtask_prompt_tokens,
            tokenized_prompt_mask=subtask_prompt_mask,
        )

        # Generate subtask tokens
        generated_tokens = self.generate_subtask(
            subtask_rng, subtask_obs, max_gen_steps=max_gen_steps, temperature=temperature
        )
        # generated_tokens: int32[batch, max_gen_steps]

        # Decode the first sample's tokens to text
        subtask_text = tokenizer.detokenize(np.asarray(generated_tokens[0]))
        logger.info(f"Generated subtask: '{subtask_text}'")

        # ====== Stage 2: Generate actions using subtask as prompt ======
        # Re-tokenize with the format matching training config:
        # - discrete_state_input=True  → "Task: ..., State: 128 130 ...;\nAction: " (pi05_droid)
        # - discrete_state_input=False → "subtask_text\n"                            (pi05_libero)
        if self.discrete_state_input:
            state_np = np.asarray(observation.state[0])  # [action_dim]
        else:
            state_np = None
        action_tokens, action_mask = tokenizer.tokenize(subtask_text, state=state_np)
        # Add batch dimension
        action_tokens = jnp.broadcast_to(
            jnp.asarray(action_tokens)[None, :], (batch_size, action_tokens.shape[0])
        )
        action_mask = jnp.broadcast_to(
            jnp.asarray(action_mask)[None, :], (batch_size, action_mask.shape[0])
        )

        # Create observation with subtask prompt for action generation
        action_obs = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=action_tokens,
            tokenized_prompt_mask=action_mask,
        )

        # Use standard sample_actions for flow matching action generation
        actions = self.sample_actions(action_rng, action_obs, num_steps=num_steps, noise=noise)

        return actions, subtask_text
