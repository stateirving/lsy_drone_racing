"""基于实际环境的 Autoreset Bug 复现脚本

使用你的实际环境堆栈：
- VecDroneRaceEnv
- NormalizeActions
- RacingRewardWrapper
- RacingObservationWrapper

运行方式:
    python test_autoreset_bug_real.py --config_file=level0.toml
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
from scipy.spatial.transform import Rotation

# 自定义 Wrapper
from lsy_drone_racing.rl_training.wrappers.observation import RacingObservationWrapper
from lsy_drone_racing.rl_training.wrappers.reward_racing_lv0 import RacingRewardWrapper


def create_env(config_path: Path, num_envs: int = 4, stage: int = 2, n_history: int = 2):
    """创建与训练时完全一致的环境"""
    config = load_config(config_path)
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.get("obstacles", []))
    
    # 1. Base Environment
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
        max_episode_steps=100,  # 短episode，方便触发autoreset
        device="cpu",
    )
    
    # 2. NormalizeActions
    env = NormalizeActions(env)
    
    # 3. RacingRewardWrapper
    env = RacingRewardWrapper(
        env,
        n_gates=n_gates,
        stage=stage,
        coef_progress=20.0,
    )
    
    # 4. RacingObservationWrapper
    env = RacingObservationWrapper(
        env, 
        n_gates=n_gates, 
        n_obstacles=n_obstacles, 
        stage=stage,
        n_history=n_history,
    )
    
    return env, n_gates, n_obstacles


def demonstrate_autoreset_bug(config_path: Path, num_envs: int = 4):
    """演示 autoreset bug"""
    print("=" * 80)
    print("Autoreset Bug 复现 - 使用实际环境")
    print("=" * 80)
    print()
    
    # 创建环境
    env, n_gates, n_obstacles = create_env(config_path, num_envs=num_envs, stage=2, n_history=2)
    obs, _ = env.reset(seed=42)

    # 直接访问底层环境（跳过所有 wrapper）
    base_obs, _ = env.env.env.env.reset(seed=42)  # 取决于有几层 wrapper

    print("Initial rotation matrices after reset:")
    for i in range(4):
        quat = base_obs["quat"][i]
        rot = Rotation.from_quat(quat).as_matrix()
        print(f"  Env {i}: {rot[0, :3]}")  # 打印第一行

    print(f"环境配置:")
    print(f"  - 并行环境数: {num_envs}")
    print(f"  - 门数量: {n_gates}")
    print(f"  - 障碍物数量: {n_obstacles}")
    print(f"  - 观测维度: {env.observation_space.shape[1]}")
    print(f"  - Max Episode Steps: 100")
    print()
    
    # 观测空间布局
    print("观测空间布局 (总维度: 88):")
    print("  [0:1]    pos_z         (1D)")
    print("  [1:4]    vel_body      (3D)")
    print("  [4:7]    ang_vel       (3D)")
    print("  [7:16]   rot_mat       (9D)")
    print("  [16:28]  gate1         (12D)")
    print("  [28:40]  gate2         (12D)")
    print("  [40:44]  prev_action   (4D)")
    print("  [44:56]  obstacles     (12D)")
    print("  [56:88]  history       (32D = 2 frames * 16D)")
    print()
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"初始化完成，观测形状: {obs.shape}")
    print()
    
    # 检查初始状态
    print("检查初始状态 (env 0):")
    print(f"  Prev Action [40:44]: {obs[0, 40:44]}")
    print(f"  History [56:88]:     {obs[0, 56:88][:8]}... (仅显示前8维)")
    print()
    
    contamination_count = 0
    total_autoreset = 0
    
    # 运行多步，等待 autoreset
    print("=" * 80)
    print("开始运行，等待 autoreset 事件...")
    print("=" * 80)
    print()
    
    for step in range(200):
        # 使用大动作增加碰撞概率
        action = np.random.randn(num_envs, 4) * 0.3
        action[:, 0] = 0.5  # 推力固定
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 检测 autoreset
        done_mask = terminated | truncated
        
        if np.any(done_mask):
            done_indices = np.where(done_mask)[0]
            total_autoreset += len(done_indices)
            
            for idx in done_indices:
                # 提取关键部分
                pos_z = obs[idx, 0]
                vel_body = obs[idx, 1:4]
                prev_action = obs[idx, 40:44]
                history = obs[idx, 56:88]
                
                # 检查是否污染
                prev_action_clean = np.allclose(prev_action, 0.0, atol=1e-5)
                history_clean = np.allclose(history, 0.0, atol=1e-3)
                
                is_contaminated = not (prev_action_clean and history_clean)
                
                if is_contaminated:
                    contamination_count += 1
                    
                print(f"[Step {step:3d}] 环境 {idx} autoreset:")
                print(f"  位置 Z: {pos_z:.6f}")
                print(f"  速度:   {vel_body}")
                print(f"  Prev Action: {prev_action}")
                print(f"  History (前8维): {history[:10]}")
                
                if is_contaminated:
                    print(f"  ❌ 状态污染!")
                    print(f"     - Prev Action 范数: {np.linalg.norm(prev_action):.4f}")
                    print(f"     - History 范数:     {np.linalg.norm(history):.4f}")
                else:
                    print(f"  ✅ 状态干净")
                
                print()
    
    env.close()
    
    # 总结
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总 autoreset 次数: {total_autoreset}")
    print(f"状态污染次数:      {contamination_count}")
    
    if contamination_count > 0:
        print(f"\n❌ 检测到 {contamination_count} 次状态污染!")
        print(f"污染率: {contamination_count/total_autoreset*100:.1f}%")
        print()
        print("这证明了 autoreset bug 的存在：")
        print("  - 环境 reset 后，wrapper 的内部状态没有重置")
        print("  - prev_action 保留了上个 episode 的最后动作")
        print("  - history 保留了上个 episode 的状态")
        print()
        print("修复方法：在 RacingObservationWrapper.step() 中添加：")
        print("""
    def step(self, action):
        # ... 现有代码 ...
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 🔧 检测 autoreset
        done_mask = terminated | truncated
        
        # ... 更新历史 ...
        
        # 🔧 重置 autoreset 环境的内部状态
        if np.any(done_mask):
            self._prev_action[done_mask] = 0.0
            if self.n_history > 0:
                init_state = self._extract_basic_state(obs)
                for i in range(self.n_history):
                    self._history_buffer[done_mask, i, :] = init_state[done_mask]
        
        # ... 返回观测 ...
        """)
    else:
        print(f"\n✅ 未检测到状态污染")
        print(f"如果你的代码已经修复了 bug，这是正常的。")
        print(f"如果还没修复但没看到污染，可能是运行步数太少，尝试增加步数。")
    
    print()


def check_observation_indices(config_path: Path):
    """检查观测索引是否正确"""
    print("=" * 80)
    print("观测索引验证")
    print("=" * 80)
    print()
    
    env, n_gates, n_obstacles = create_env(config_path, num_envs=2, stage=2, n_history=2)
    obs, info = env.reset(seed=42)
    
    print("执行几步，让观测有一些变化...")
    for _ in range(3):
        action = np.array([[0.5, 0.1, -0.1, 0.0], [0.5, -0.1, 0.1, 0.0]])
        obs, _, _, _, _ = env.step(action)
    
    print()
    print("环境 0 的观测值 (各部分):")
    print()
    
    idx = 0
    segments = [
        ("Pos Z",        0,   1),
        ("Vel Body",     1,   4),
        ("Ang Vel",      4,   7),
        ("Rot Mat",      7,  16),
        ("Gate 1",      16,  28),
        ("Gate 2",      28,  40),
        ("Prev Action", 40,  44),
        ("Obstacles",   44,  56),
        ("History",     56,  88),
    ]
    
    for name, start, end in segments:
        values = obs[idx, start:end]
        if len(values) <= 6:
            print(f"  [{start:2d}:{end:2d}] {name:12s}: {values}")
        else:
            print(f"  [{start:2d}:{end:2d}] {name:12s}: {values[:6]}... (显示前6维)")
    
    print()
    
    # 验证维度
    expected_dims = {
        "Pos Z": 1,
        "Vel Body": 3,
        "Ang Vel": 3,
        "Rot Mat": 9,
        "Gate 1": 12,
        "Gate 2": 12,
        "Prev Action": 4,
        "Obstacles": 12,
        "History": 32,
    }
    
    print("维度验证:")
    total = 0
    all_correct = True
    for name, start, end in segments:
        actual_dim = end - start
        expected_dim = expected_dims[name]
        match = "✓" if actual_dim == expected_dim else "✗"
        print(f"  {match} {name:12s}: {actual_dim:2d} (期望: {expected_dim:2d})")
        total += actual_dim
        if actual_dim != expected_dim:
            all_correct = False
    
    print(f"\n  总维度: {total} (期望: 88)")
    
    if all_correct and total == 88:
        print("\n✅ 观测空间索引正确!")
    else:
        print("\n❌ 观测空间索引有误!")
    
    env.close()
    print()


def main(
    config_file: str = "level0.toml",
    num_envs: int = 4,
    check_indices: bool = False,
):
    """运行测试
    
    Args:
        config_file: 配置文件名
        num_envs: 并行环境数
        check_indices: 是否检查观测索引
    """
    # 查找配置文件
    config_path = Path(__file__).parents[3] / "config" / config_file
    
    if not config_path.exists():
        fallback = config_path.parent / "level0_no_obst.toml"
        if fallback.exists():
            print(f"未找到 {config_file}，使用 {fallback.name}")
            config_path = fallback
        else:
            print(f"错误: 配置文件不存在: {config_path}")
            return
    
    print(f"使用配置文件: {config_path.name}\n")
    
    try:
        if check_indices:
            check_observation_indices(config_path)
        else:
            demonstrate_autoreset_bug(config_path, num_envs)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 在所有 import 之后，main() 之前
    print("\n" + "="*60)
    print("验证 wrapper 代码")
    print("="*60)

    from lsy_drone_racing.rl_training.wrappers.observation import RacingObservationWrapper
    import inspect

    # 打印 step 方法的源代码前几行
    source = inspect.getsource(RacingObservationWrapper.step)
    print("RacingObservationWrapper.step() 源代码:")
    print(source[:500])  # 前500个字符
    print("="*60 + "\n")
    fire.Fire(main)