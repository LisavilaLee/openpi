from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger("openpi")

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_with_subtask(
        self,
        obs: dict,
        *,
        max_decoding_steps: int = 128,
        temperature: float = 0.0,
        noise: np.ndarray | None = None,
    ) -> dict:
        """Two-stage inference for Pi0.5: generate subtask, then generate actions.

        Stage 1: The VLM autoregressively generates a subtask from the high-level prompt.
        Stage 2: The generated subtask is used as the language condition for action generation.

        This method handles the re-tokenization between stages, which cannot be done inside
        a jitted function due to the dynamic nature of text processing.

        Args:
            obs: Observation dict. Must contain a "prompt" key with the high-level task
                 instruction (e.g., "clean the bedroom") and a "state" key.
            max_decoding_steps: Maximum number of subtask tokens to generate.
            temperature: Sampling temperature for subtask generation (0.0 = greedy).
            noise: Optional initial noise for Flow Matching action generation.

        Returns:
            Dict with keys:
            - "state": The robot state.
            - "actions": Generated continuous actions.
            - "subtask": The generated subtask text string.
            - "policy_timing": Timing information.
        """
        from openpi.models.pi0 import Pi0

        if self._is_pytorch_model:
            raise NotImplementedError("infer_with_subtask is only supported for JAX models")

        if not isinstance(self._model, Pi0) or not self._model.pi05:
            raise RuntimeError("infer_with_subtask requires a Pi0.5 model (pi05=True)")

        # --- Stage 1: Generate subtask ---
        # Build a subtask-prediction observation: replace "Action: " with "Subtask: "
        # in the tokenized prompt.
        inputs_stage1 = jax.tree.map(lambda x: x, obs)

        # Save the original prompt and state for re-tokenization
        original_prompt = inputs_stage1.get("prompt")
        if original_prompt is not None and not isinstance(original_prompt, str):
            original_prompt = original_prompt.item()
        if original_prompt is None:
            raise ValueError("A 'prompt' key is required in obs for subtask generation")

        # Apply the standard input transforms (which will tokenize with "Action: " suffix).
        # We need to re-tokenize for subtask, so we do it manually.
        inputs_stage1 = self._input_transform(inputs_stage1)

        # Re-tokenize with "Subtask: " format instead of "Action: "
        tok = _tokenizer.PaligemmaTokenizer(self._model.max_token_len)
        state_for_tokenize = np.asarray(inputs_stage1["state"])
        subtask_tokens, subtask_mask = tok.tokenize_for_subtask(original_prompt, state_for_tokenize)

        inputs_stage1["tokenized_prompt"] = subtask_tokens
        inputs_stage1["tokenized_prompt_mask"] = subtask_mask

        # Make batch and convert to jax arrays
        inputs_stage1_batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_stage1)

        observation_stage1 = _model.Observation.from_dict(inputs_stage1_batched)

        self._rng, subtask_rng = jax.random.split(self._rng)
        start_time = time.monotonic()

        # JIT the generate_subtask call
        if not hasattr(self, "_generate_subtask"):
            self._generate_subtask = nnx_utils.module_jit(self._model.generate_subtask)

        generated_tokens = self._generate_subtask(
            subtask_rng,
            observation_stage1,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature,
        )
        # generated_tokens: [1, max_decoding_steps]
        generated_tokens_np = np.asarray(generated_tokens[0])
        subtask_text = tok.detokenize(generated_tokens_np)
        logger.info(f"Generated subtask: {subtask_text}")

        subtask_time = time.monotonic() - start_time

        # --- Stage 2: Generate actions using the subtask ---
        inputs_stage2 = jax.tree.map(lambda x: x, obs)
        # Replace the high-level prompt with the generated subtask
        inputs_stage2["prompt"] = np.asarray(subtask_text)

        # Apply standard transforms (will tokenize with "Action: " suffix for action generation)
        inputs_stage2 = self._input_transform(inputs_stage2)
        inputs_stage2_batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_stage2)

        observation_stage2 = _model.Observation.from_dict(inputs_stage2_batched)

        self._rng, action_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_jax = jnp.asarray(noise)
            if noise_jax.ndim == 2:
                noise_jax = noise_jax[None, ...]
            sample_kwargs["noise"] = noise_jax

        action_start_time = time.monotonic()
        actions = self._sample_actions(action_rng, observation_stage2, **sample_kwargs)
        action_time = time.monotonic() - action_start_time

        outputs = {
            "state": inputs_stage2["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["subtask"] = subtask_text
        outputs["policy_timing"] = {
            "subtask_ms": subtask_time * 1000,
            "action_ms": action_time * 1000,
            "infer_ms": (subtask_time + action_time) * 1000,
        }

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
