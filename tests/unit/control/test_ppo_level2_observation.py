"""Tests for direct level2 PPO obstacle observations."""

import jax.numpy as jp
import numpy as np
import pytest
import torch

from lsy_drone_racing.control.ppo_level2_inference import PPOLevel2Inference
from lsy_drone_racing.control.ppo_level3_inference import PPOLevel2Inference as PPOLevel3Inference
from lsy_drone_racing.control.ppo_level2_observation import (
    LEGACY_OBSERVATION_LAYOUT,
    OBSERVATION_LAYOUT,
    checkpoint_hidden_dim,
    make_checkpoint,
    unpack_checkpoint,
)
from lsy_drone_racing.control.train_CleanRL_ppo import Agent, RaceObservation


def test_obstacle_heading_xy_matches_training_and_inference() -> None:
    """Training and inference must produce identical fixed-order obstacle features."""
    pos = np.array([1.0, 2.0, 0.8], dtype=np.float32)
    quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32)
    rot = PPOLevel2Inference.quat_to_rotmat(quat)
    obstacles_pos = np.array(
        [
            [2.0, 2.0, 1.55],
            [1.0, 4.0, 1.55],
        ],
        dtype=np.float32,
    )
    detected = np.array([True, False])
    obs = {
        "obstacles_pos": obstacles_pos,
        "obstacles_visited": detected,
    }

    inference_features = PPOLevel2Inference._obstacle_heading_xy(obs, pos, rot)
    training_features = RaceObservation._obstacle_heading_xy(
        {
            "obstacles_pos": jp.asarray(obstacles_pos[None, ...]),
            "obstacles_visited": jp.asarray(detected[None, ...]),
        },
        jp.asarray(pos[None, ...]),
        jp.asarray(rot[None, ...]),
    )

    expected = np.array(
        [
            0.0,
            -1.0,
            1.0,
            1.0,
            2.0,
            0.0,
            2.0,
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(inference_features, expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(training_features[0]), expected, atol=1e-6)


@pytest.mark.parametrize("controller_cls", [PPOLevel2Inference, PPOLevel3Inference])
def test_inference_obstacle_heading_does_not_mutate_rotmat(controller_cls) -> None:
    """Obstacle heading features must not corrupt the rotmat later packed into obs."""
    pos = np.zeros(3, dtype=np.float32)
    quat = np.array([0.0, np.sin(np.pi / 8), 0.0, np.cos(np.pi / 8)], dtype=np.float32)
    rot = controller_cls.quat_to_rotmat(quat)
    rot_before = rot.copy()
    obs = {
        "obstacles_pos": np.array([[1.0, 0.0, 1.55]], dtype=np.float32),
        "obstacles_visited": np.array([True]),
    }

    controller_cls._obstacle_heading_xy(obs, pos, rot)

    np.testing.assert_allclose(rot, rot_before, atol=0.0)


def test_checkpoint_observation_layout_metadata() -> None:
    """New checkpoints declare their layout and old raw state dicts stay legacy."""
    state_dict = Agent((103,), (4,), hidden_dim=128).state_dict()
    checkpoint = make_checkpoint(state_dict)

    unpacked, layout = unpack_checkpoint(checkpoint)
    assert unpacked is state_dict
    assert layout == OBSERVATION_LAYOUT
    assert checkpoint["hidden_dim"] == 128
    assert checkpoint_hidden_dim(checkpoint, unpacked) == 128

    unpacked, layout = unpack_checkpoint(state_dict)
    assert unpacked is state_dict
    assert layout == LEGACY_OBSERVATION_LAYOUT
    assert checkpoint_hidden_dim(state_dict, unpacked) == 128


def test_checkpoint_hidden_dim_metadata_must_match_weights() -> None:
    """Reject corrupt width metadata before constructing a mismatched network."""
    state_dict = Agent((103,), (4,), hidden_dim=64).state_dict()
    checkpoint = make_checkpoint(state_dict)
    checkpoint["hidden_dim"] = 128

    with pytest.raises(ValueError, match="actor weights use 64"):
        checkpoint_hidden_dim(checkpoint, state_dict)


def test_agent_hidden_dim_controls_both_networks() -> None:
    """Actor and critic use the configured shared hidden width."""
    agent = Agent((103,), (4,), hidden_dim=128)

    assert agent.actor_mean[0].weight.shape == torch.Size([128, 103])
    assert agent.actor_mean[2].weight.shape == torch.Size([128, 128])
    assert agent.critic[0].weight.shape == torch.Size([128, 103])
    assert agent.critic[2].weight.shape == torch.Size([128, 128])
