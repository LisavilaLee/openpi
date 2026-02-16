#!/usr/bin/env python3

"""Test script for π0.5 two-stage subtask inference.

This script demonstrates the full pipeline:
  Stage 1: VLM generates a subtask from a high-level instruction
  Stage 2: Action Expert generates actions conditioned on the subtask

Available configs:
  pi05_libero  — Fine-tuned on LIBERO (discrete_state_input=False)
  pi05_droid   — Inference on DROID (discrete_state_input=True)
  pi05_aloha   — Inference on ALOHA (discrete_state_input=True)
  NOTE: 'pi05_base' is NOT a valid config name; it is only a pretrained
        checkpoint used by fine-tuning configs.

Usage:
  # LIBERO (local checkpoint):
  TF_CPP_MIN_LOG_LEVEL=2 uv run scripts/test_subtask_inference.py \\
      --config pi05_libero \\
      --checkpoint /path/to/pi05_libero/checkpoint \\
      --prompt "pick up the red cup"

  # DROID (downloads from GCS):
  CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=2 uv run scripts/test_subtask_inference.py \\
      --config pi05_droid \\
      --prompt "clean the table"

  # DROID with custom temperature:
  TF_CPP_MIN_LOG_LEVEL=2 uv run scripts/test_subtask_inference.py \\
      --config pi05_droid \\
      --prompt "organize the workspace" --temperature 0.7
"""

import argparse
import logging
import os
import time

# Proxy for GCS download (used by download.py with gcsfs/aiohttp trust_env=True).
# If this script runs on a *remote* machine (e.g. node9), 127.0.0.1 is that machine's
# localhost — use the IP of the machine where the proxy runs and enable "Allow LAN".
_proxy = os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or "http://127.0.0.1:17890"
os.environ["http_proxy"] = os.environ.get("http_proxy") or _proxy
os.environ["https_proxy"] = os.environ.get("https_proxy") or _proxy
os.environ["all_proxy"] = os.environ.get("all_proxy") or _proxy

import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("subtask_test")


def log(msg: str):
    """Print to stdout (immune to logging config) and flush immediately."""
    print(f"[subtask_test] {msg}", flush=True)


def create_dummy_observation(config_name: str) -> dict:
    """Create a dummy observation for testing (random images and state).

    Uses LIBERO-style keys for pi05_libero (observation/image, observation/wrist_image,
    observation/state); DROID-style keys for pi05_droid and similar.
    """
    config_lower = config_name.lower()
    is_libero = "libero" in config_lower

    if is_libero:
        # LIBERO: LiberoInputs expects observation/image, observation/wrist_image, observation/state
        obs = {
            "observation/image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.random.randn(8).astype(np.float32) * 0.1,
        }
    elif "droid" in config_lower:
        # DROID: DroidInputs expects joint_position (7) + gripper_position (1)
        obs = {
            "observation/exterior_image_1_left": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.random.randn(7).astype(np.float32) * 0.1,
            "observation/gripper_position": np.random.randn(1).astype(np.float32) * 0.1,
        }
    else:
        # Generic / ALOHA style
        obs = {
            "observation/exterior_image_1_left": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        }
        if "aloha" in config_lower:
            obs["observation/wrist_image_right"] = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        state_dim = 14 if "aloha" in config_lower else 8
        obs["state"] = np.random.randn(state_dim).astype(np.float32) * 0.1

    return obs


def main():
    parser = argparse.ArgumentParser(description="Test π0.5 subtask prediction inference")
    parser.add_argument("--config", type=str, default="pi05_libero",
                        help="Training config name (e.g., pi05_base, pi05_droid, pi05_libero)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path. If not set, uses default for the chosen --config (e.g. GCS for pi05_droid).")
    parser.add_argument("--prompt", type=str, default="pick up the object on the table",
                        help="High-level task instruction")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for subtask generation (0=greedy)")
    parser.add_argument("--max-gen-steps", type=int, default=50,
                        help="Maximum tokens to generate for subtask")
    parser.add_argument("--num-action-steps", type=int, default=10,
                        help="Number of flow matching denoising steps")
    args = parser.parse_args()

    # Load config
    log(f"[1/5] Loading config: {args.config}")
    train_config = _config.get_config(args.config)

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_dir = args.checkpoint
    else:
        default_checkpoints = {
            "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
            "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
            "pi05_aloha": "gs://openpi-assets/checkpoints/pi05_aloha",
        }
        checkpoint_dir = default_checkpoints.get(args.config, f"gs://openpi-assets/checkpoints/{args.config}")

    log(f"[2/5] Loading checkpoint from: {checkpoint_dir}")
    policy = policy_config.create_subtask_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"num_steps": args.num_action_steps},
        subtask_max_gen_steps=args.max_gen_steps,
        subtask_temperature=args.temperature,
    )
    log("[2/5] Model loaded successfully!")

    # Create dummy observation
    log("[3/5] Creating dummy observation...")
    obs = create_dummy_observation(args.config)
    obs["prompt"] = args.prompt

    log(f"[4/5] High-level prompt: '{args.prompt}'")
    log("[4/5] Running two-stage inference (this may take a while on first run due to XLA compilation)...")

    # Run inference
    start_time = time.monotonic()
    result = policy.infer(obs)
    total_time = time.monotonic() - start_time

    log("[5/5] Inference complete!")

    # Print results
    log("=" * 60)
    log("RESULTS")
    log("=" * 60)
    log(f"  High-level prompt : '{args.prompt}'")
    log(f"  Generated subtask : '{result['subtask_text']}'")
    log(f"  Action shape      : {result['actions'].shape}")
    log(f"  Action range      : [{result['actions'].min():.4f}, {result['actions'].max():.4f}]")
    log(f"  Total time        : {total_time * 1000:.1f} ms")
    log(f"  Model time        : {result['policy_timing']['infer_ms']:.1f} ms")
    log("=" * 60)
    log(f"  First action step (first 8 dims): {result['actions'][0, :8]}")


if __name__ == "__main__":
    main()
