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

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class SubtaskPolicy(BasePolicy):
    """Policy with two-stage inference: subtask generation + action generation.

    Implements the full π0.5 inference pipeline as described in the paper:
    1. VLM autoregressively generates a subtask from a high-level instruction
    2. Generated subtask is used as prompt for flow-matching action generation

    This class wraps a Pi0 model and manages the two-stage data flow.
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        subtask_max_gen_steps: int = 50,
        subtask_temperature: float = 0.0,
        tokenizer_max_len: int = 200,
    ):
        """Initialize SubtaskPolicy.

        Args:
            model: Pi0 model with generate_subtask and sample_actions methods.
            rng: JAX random key.
            transforms: Input transforms (applied to image/state, NOT prompt tokenization).
            output_transforms: Output transforms for action post-processing.
            sample_kwargs: Extra kwargs for sample_actions (e.g. num_steps).
            metadata: Additional metadata.
            subtask_max_gen_steps: Max tokens to generate for subtask.
            subtask_temperature: Sampling temperature for subtask generation.
            tokenizer_max_len: Max token length for the PaliGemma tokenizer.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._rng = rng or jax.random.key(0)
        self._subtask_max_gen_steps = subtask_max_gen_steps
        self._subtask_temperature = subtask_temperature
        self._tokenizer = _tokenizer.PaligemmaTokenizer(max_len=tokenizer_max_len)
        self._last_subtask: str | None = None

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Run two-stage inference.

        The input obs dict should contain:
            - Image keys (e.g. "observation/exterior_image_1_left")
            - "prompt": the HIGH-LEVEL instruction (e.g. "clean the bedroom")
            - State keys

        The method will:
            1. Apply input transforms (excluding prompt tokenization)
            2. Tokenize prompt as subtask generation prefix
            3. Call model.sample_actions_with_subtask()
            4. Apply output transforms

        Returns:
            Dict with "actions", "state", "subtask_text", and "policy_timing".
        """
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # Extract the high-level prompt before transforms consume it
        high_level_prompt = inputs.get("prompt", "")
        if not isinstance(high_level_prompt, str):
            high_level_prompt = high_level_prompt.item()

        # Apply standard input transforms (image processing, state normalization, etc.)
        # Note: we apply transforms that do NOT tokenize the prompt, since we handle that ourselves.
        inputs = self._input_transform(inputs)

        # Remove "prompt" string if present, as it cannot be converted to JAX array
        # (we already extracted high_level_prompt earlier)
        if "prompt" in inputs:
            del inputs["prompt"]

        # Make a batch and convert to jax.Array
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        # Build observation (prompt tokenization is handled inside sample_actions_with_subtask)
        # We need to provide a dummy tokenized_prompt for Observation.from_dict
        # since the actual tokenization happens inside the model method
        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()

        # Two-stage inference
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_jax = jnp.asarray(noise)
            if noise_jax.ndim == 2:
                noise_jax = noise_jax[None, ...]
            sample_kwargs["noise"] = noise_jax

        actions, subtask_text = self._model.sample_actions_with_subtask(
            sample_rng,
            observation,
            high_level_prompt=high_level_prompt,
            tokenizer=self._tokenizer,
            max_gen_steps=self._subtask_max_gen_steps,
            temperature=self._subtask_temperature,
            **sample_kwargs,
        )

        self._last_subtask = subtask_text
        model_time = time.monotonic() - start_time

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["subtask_text"] = subtask_text
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def last_subtask(self) -> str | None:
        """Return the last generated subtask text, for logging/debugging."""
        return self._last_subtask

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
