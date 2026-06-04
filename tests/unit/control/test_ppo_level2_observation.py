"""Tests for direct level2 PPO obstacle observations."""

import jax.numpy as jp
import numpy as np

from lsy_drone_racing.control.ppo_level2_inference import PPOLevel2Inference
from lsy_drone_racing.control.ppo_level2_observation import (
    LEGACY_OBSERVATION_LAYOUT,
    OBSERVATION_LAYOUT,
    make_checkpoint,
    unpack_checkpoint,
)
from lsy_drone_racing.control.train_CleanRL_ppo import RaceObservation


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


def test_checkpoint_observation_layout_metadata() -> None:
    """New checkpoints declare their layout and old raw state dicts stay legacy."""
    state_dict = {"weight": np.array([1.0], dtype=np.float32)}

    unpacked, layout = unpack_checkpoint(make_checkpoint(state_dict))
    assert unpacked is state_dict
    assert layout == OBSERVATION_LAYOUT

    unpacked, layout = unpack_checkpoint(state_dict)
    assert unpacked is state_dict
    assert layout == LEGACY_OBSERVATION_LAYOUT
