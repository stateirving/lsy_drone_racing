# test_gate_collision.py
import gymnasium
import jax.numpy as jp
import numpy as np
from pathlib import Path
from lsy_drone_racing.utils import load_config
import time

config = load_config(Path("config/level0.toml"))

# 强制启用渲染
config.sim.render = True

env = gymnasium.make(
    config.env.id,
    freq=config.env.freq,
    sim_config=config.sim,
    sensor_range=config.env.sensor_range,
    control_mode=config.env.control_mode,
    track=config.env.track,
    seed=42,
)

obs, info = env.reset()

# 获取第一个门的位置
gate_pos = obs['gates_pos'][0]
print(f"第一个门的位置: {gate_pos}")
print(f"无人机初始位置: {obs['pos']}")
print(f"控制模式: {config.env.control_mode}")

# 渲染设置
fps = 30  # 渲染帧率
render_every = max(1, config.env.freq // fps)  # 每隔几步渲染一次

# 强制让无人机飞向门框
for step in range(200):  # 增加步数以便观察
    if config.env.control_mode == "attitude":
        # [roll, pitch, yaw, thrust]
        action = np.array([0.0, 0.3, 0.0, 0.6])
    else:
        action = np.zeros(13)
        # 策略：撞右侧框
        action[0] = gate_pos[0]
        action[1] = gate_pos[1] + 0.1  # 偏移撞框
        action[2] = gate_pos[2]
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 渲染（控制频率）
    if step % render_every == 0:
        env.render()
        time.sleep(0.1)  # 稍微降速以便观察
    
    # 获取contacts
    contacts = env.unwrapped.sim.contacts()
    contact_impl = env.unwrapped.sim.mjx_data._impl.contact
    
    # 每10步打印一次位置
    if step % 10 == 0:
        print(f"\n步骤 {step}:")
        print(f"  无人机位置: {obs['pos']}")
        print(f"  到门的距离: {np.linalg.norm(obs['pos'] - gate_pos):.3f}m")
    
    # 检查是否有新的接触（超过地面接触）
    if jp.sum(contacts) > 4:
        print(f"\n🚨 步骤 {step}: 检测到碰撞！")
        print(f"  总接触数: {jp.sum(contacts)}")
        print(f"  无人机位置: {obs['pos']}")
        print(f"  disabled_drones: {env.unwrapped.data.disabled_drones}")
        
        active_contacts = jp.where(contacts[0])[0]
        
        # 只显示非地面接触
        for idx in active_contacts:
            idx = int(idx)
            geom1 = int(contact_impl.geom1[0, idx])
            geom2 = int(contact_impl.geom2[0, idx])
            dist = float(contact_impl.dist[0, idx])
            
            # 获取几何体名称
            try:
                geom1_name = env.unwrapped.sim.mj_model.geom(geom1).name
                geom2_name = env.unwrapped.sim.mj_model.geom(geom2).name
                
                # 过滤掉地面接触
                if 'ground' not in geom1_name and 'ground' not in geom2_name:
                    print(f"    ⚠️  {geom1_name} <-> {geom2_name}, dist={dist:.4f}")
            except:
                print(f"    [{idx}] geom{geom1} <-> geom{geom2}, dist={dist:.4f}")
        
        # 碰撞后继续渲染几帧以便观察
        for _ in range(30):
            env.render()
            time.sleep(0.033)
    
    if terminated or truncated:
        print(f"\n✅ 回合结束于步骤 {step}")
        print(f"  terminated: {terminated}")
        print(f"  truncated: {truncated}")
        print(f"  disabled_drones: {env.unwrapped.data.disabled_drones}")
        
        # 最终渲染几帧
        for _ in range(30):
            env.render()
            time.sleep(0.033)
        break

env.close()