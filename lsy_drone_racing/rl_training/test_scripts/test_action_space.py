import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from crazyflow.envs.norm_actions_wrapper import NormalizeActions

# 加载你的配置
config_path = Path(__file__).parents[2] / "config" / "level0.toml"
print(config_path)
config = load_config(config_path)

# 创建环境 (不加任何 Wrapper，看最原始的样子)
env = VecDroneRaceEnv(
    num_envs=1,
    freq=config.env.freq,
    sim_config=config.sim,
    track=config.env.track,
    sensor_range=config.env.sensor_range,
    control_mode=config.env.control_mode, # 关键看这个模式下的定义
    seed=42,
    device="cpu"
)
# env = NormalizeActions(env)
print("\n" + "="*50)
print(f"当前 Control Mode: {config.env.control_mode}")
print("="*50)

# 打印动作空间详情
low = env.single_action_space.low
high = env.single_action_space.high

print(f"Action Space Shape: {env.single_action_space.shape}")
print(f"Action Low:  {low}")
print(f"Action High: {high}")

print("\n分析:")
for i in range(4):
    print(f"  Dimension {i}: Range [{low[i]:.2f}, {high[i]:.2f}]")

print("="*50 + "\n")
env.close()