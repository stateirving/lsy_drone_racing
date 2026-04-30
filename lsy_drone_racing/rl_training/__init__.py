"""RL Training 包

提供无人机竞速的强化学习训练组件。

模块:
    - wrappers: 环境包装器 (观测变换、奖励塑形)
    - ppo_racing: PPO 训练脚本
"""

from lsy_drone_racing.rl_training.wrappers import (
    RacingObservationWrapper,
    RacingRewardWrapper,
)

__all__ = [
    "RacingObservationWrapper",
    "RacingRewardWrapper",
]