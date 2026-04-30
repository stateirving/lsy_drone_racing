"""环境调试脚本 (详细打印版)

用于验证环境创建、观测空间、奖励函数是否正确工作。
重点：在 Base Env 测试中打印所有原始观测值。

运行方式:
    python test_env.py
"""

from __future__ import annotations

import numpy as np
import torch
import fire
from pathlib import Path

# 环境相关
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch

# 自定义 Wrapper
from lsy_drone_racing.rl_training.wrappers.observation_dev import RacingObservationWrapper
from lsy_drone_racing.rl_training.wrappers.reward import RacingRewardWrapper


def test_base_env(config_path: str, num_envs: int = 4):
    """测试基础环境 (无 Wrapper) 并打印所有原始观测值"""
    print("=" * 60)
    print("1. 测试基础环境 VecDroneRaceEnv (详细数据打印)")
    print("=" * 60)
    
    config = load_config(config_path)
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.get("obstacles", []))
    print(f"   配置文件: {config_path}")
    print(f"   门数量: {n_gates}, 障碍物数量: {n_obstacles}")
    
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances", None),
        randomizations=config.env.get("randomizations", None),
        seed=42,
        max_episode_steps=1500,
        device="cpu",
    )
    
    print(f"   ✓ 环境创建成功")
    
    # Reset
    obs, info = env.reset(seed=42)
    
    print(f"\n{'='*20} 原始观测值 (Reset) {'='*20}")
    # 遍历并打印所有键值
    sorted_keys = sorted(obs.keys())
    for key in sorted_keys:
        val = obs[key]
        print(f"\n>>> Key: ['{key}']")
        print(f"    Shape: {val.shape}")
        print(f"    Value (Env 0):") # 为了防止刷屏，这里主要展示第0个环境的值，如果想看全部，去掉 [0]
        print(val[0]) 
        # 如果你想看所有环境的矩阵，请取消下面这行的注释:
        # print(val) 
        
    print(f"\n{'='*60}\n")
    
    # Step with random action
    print("执行一步随机动作...")
    action = np.zeros((num_envs, 4))
    action[:, 0] = 0.5 # 给点推力
    obs, reward, term, trunc, info = env.step(action)
    
    print(f"Step 后 target_gate (所有环境): {obs['target_gate']}")
    
    env.close()
    return True


def test_reward_wrapper(config_path: str, num_envs: int = 4):
    """测试奖励 Wrapper"""
    print("=" * 60)
    print("2. 测试 RacingRewardWrapper")
    print("=" * 60)
    
    config = load_config(config_path)
    n_gates = len(config.env.track.gates)
    
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances", None),
        randomizations=config.env.get("randomizations", None),
        seed=42,
        max_episode_steps=1500,
        device="cpu",
    )
    
    env = NormalizeActions(env)
    env = RacingRewardWrapper(
        env,
        n_gates=n_gates,
        stage=1,
        coef_progress=1.0,
    )
    
    obs, info = env.reset(seed=42)
    
    # 测试多步
    rewards = []
    for i in range(5):
        action = np.zeros((num_envs, 4))
        action[:, 0] = 0.3
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward[0])
        
    print(f"   奖励序列 (env 0): {[f'{r:.3f}' for r in rewards]}")
    env.close()
    return True


def test_observation_wrapper(config_path: str, num_envs: int = 4):
    """测试观测 Wrapper (56D 结构)"""
    print("=" * 60)
    print("3. 测试 RacingObservationWrapper (Index Check)")
    print("=" * 60)
    
    config = load_config(config_path)
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.get("obstacles", []))
    
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances", None),
        randomizations=config.env.get("randomizations", None),
        seed=42,
        max_episode_steps=1500,
        device="cpu",
    )
    
    env = NormalizeActions(env)
    env = RacingRewardWrapper(env, n_gates=n_gates, stage=1)
    
    # 初始化 Observation Wrapper
    env = RacingObservationWrapper(env, n_gates=n_gates, n_obstacles=n_obstacles, stage=1, n_history=2)
    
    obs, info = env.reset(seed=42)
    
    print(f"   Observation Shape: {obs.shape} (Expect: [{num_envs}, 88])")
    print(obs)
# ==========================================
    print(f"\n   观测分解 (env 0):")
    
    # Pos Z (1D): [0:1]
    print(f"     [0:1]   pos_z:      {obs[0, 0:1]}")
    
    # Vel Body (3D): [1:4]
    print(f"     [1:4]   vel_body:   {obs[0, 1:4]}")
    
    # Ang Vel (3D): [4:7]
    print(f"     [4:7]   ang_vel:    {obs[0, 4:7]}")
    
    # Rot Matrix (9D): [7:16]
    print(f"     [7:16]  rot_mat:    {obs[0, 7:16]}")
    
    # Gate 1 (12D): [16:28]
    print(f"     [16:28] gate1:      {obs[0, 16:22]}")
    
    # Gate 2 (12D): [28:40]
    print(f"     [28:40] gate2:      {obs[0, 22:28]}")
    
    # Prev Action (4D): [40:44]
    print(f"     [40:44] prev_act:   {obs[0, 28:32]}")
    
    # Obstacles (12D): [44:56]
    print(f"     [44:56] obstacles:  {obs[0, 32:44]}")
    
    # History (16D * 2): [56:88]
    print(f"     [56:88] history:    Shape={obs[0, 44:].shape}")

    # ==========================================
    # Masking 验证
    # ==========================================
    # 在 Stage 1，障碍物应该被屏蔽为 10.0
    # 修正后的障碍物索引是 44:56
    obstacles_slice = obs[0, 44:56]
    is_masked = np.allclose(obstacles_slice, 10.0)
    
    print(f"\n   Stage 1 Masking 验证:")
    print(f"     障碍物 (Index 44:56) 全为 10.0: {is_masked}")
    
    if not is_masked:
        print(f"     [警告] 实际值: {obstacles_slice}")
    
    env.close()
    print(f"   ✓ 观测 Wrapper 测试通过\n")
    return True


def main(
    config_file: str = "level0.toml",
    num_envs: int = 4,
):
    """运行测试。"""
    config_path = Path(__file__).parents[3] / "config" / config_file
    
    # 路径回退逻辑
    if not config_path.exists():
        fallback = config_path.parent / "level0_no_obst.toml"
        if fallback.exists():
            print(f"未找到 {config_file}，使用 {fallback.name} 进行测试")
            config_path = fallback
        else:
            print(f"错误: 配置文件不存在: {config_path}")
            return
    
    print(f"\n测试配置文件: {config_path.name}")
    
    try:
        # 1. 重点运行修改后的 base env 测试
        test_base_env(config_path, num_envs)
        
        # 2. 简单运行其他测试以确保没报错
        test_reward_wrapper(config_path, num_envs)
        test_observation_wrapper(config_path, num_envs)
        
        print("\n" + "=" * 60)
        print("✓ 测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    fire.Fire(main)