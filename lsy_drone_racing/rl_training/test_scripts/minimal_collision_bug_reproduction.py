# minimal_collision_bug_reproduction.py
"""
Minimal script to reproduce the collision detection bug in lsy_drone_racing.

Bug Description:
    When a drone collides with a gate in the first ~2 seconds of flight,
    the collision is NOT detected and terminated remains False.

Root Cause:
    In drone_env.py, the _step() method incorrectly passes self.sim.freq (simulation frequency)
    instead of self.freq (environment frequency) to _step_env(), causing the takeoff check
    to use the wrong threshold.

Expected Behavior:
    Collisions should be detected after 0.2 seconds (takeoff grace period).

Actual Behavior:
    Collisions are ignored for the first 2.0 seconds (10x longer than expected).

To reproduce:
    python minimal_collision_bug_reproduction.py --render
    python minimal_collision_bug_reproduction.py  # (no render)
"""

import gymnasium
import jax.numpy as jp
import numpy as np
import time
from pathlib import Path
from lsy_drone_racing.utils import load_config
import argparse


def main(render: bool = True):
    # Load config
    config_path = Path("config/level0.toml")
    config = load_config(config_path)
    
    # Enable/disable rendering
    config.sim.render = render
    
    # Create environment
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
    gate_pos = obs['gates_pos'][0]
    
    print("="*70)
    print("COLLISION DETECTION BUG REPRODUCTION")
    print("="*70)
    print(f"Render mode: {'ENABLED' if render else 'DISABLED'}")
    print(f"Environment frequency (env.freq): {env.unwrapped.freq} Hz")
    print(f"Simulation frequency (sim.freq): {env.unwrapped.sim.freq} Hz")
    print(f"Expected takeoff threshold: {env.unwrapped.freq // 5} steps (0.2 sec)")
    print(f"Actual takeoff threshold: {env.unwrapped.sim.freq // 5} steps (2.0 sec)")
    print(f"\nFlying towards gate at {gate_pos} to trigger collision...\n")
    
    collision_detected_at = None
    
    # Rendering control
    fps = 30  # Target rendering frame rate
    render_every = max(1, env.unwrapped.freq // fps) if render else float('inf')
    
    for step in range(200):
        # Simple control: fly towards gate with slight offset to hit the frame
        if config.env.control_mode == "attitude":
            action = np.array([[0.0, 0.3, 0.0, 0.6]])  # pitch forward
        else:
            action = np.zeros((1, 13))
            action[0, :3] = gate_pos + np.array([0, 0.25, 0])  # offset to hit gate frame
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render at controlled frame rate
        if render and step % render_every == 0:
            env.render()
            time.sleep(0.02)  # Slow down for better visualization
        
        # Monitor collision state
        contacts = env.unwrapped.sim.contacts()
        contact_count = int(jp.sum(contacts))
        data = env.unwrapped.data
        
        # Print status every 10 steps
        if step % 10 == 0:
            elapsed_time = step / env.unwrapped.freq
            print(f"Step {step:3d} ({elapsed_time:.2f}s) | "
                  f"contacts={contact_count:2d} | "
                  f"terminated={terminated} | "
                  f"disabled={bool(data.disabled_drones[0, 0])}")
        
        # Detect collision (contact_count > 4 means collision beyond ground contact)
        if contact_count > 4 and collision_detected_at is None:
            collision_detected_at = step
            elapsed_time = step / env.unwrapped.freq
            
            # Calculate takeoff status
            sim_freq = env.unwrapped.sim.freq
            env_freq = env.unwrapped.freq
            threshold_actual = sim_freq // 5  # Bug: using sim.freq
            threshold_expected = env_freq // 5  # Should use env.freq
            taken_off = data.steps[0] > threshold_actual
            
            print("\n" + "="*70)
            print(f"🚨 COLLISION DETECTED AT STEP {step} ({elapsed_time:.2f} seconds)")
            print("="*70)
            print(f"Contact count: {contact_count}")
            print(f"Terminated: {terminated}")
            print(f"Disabled: {bool(data.disabled_drones[0, 0])}")
            print(f"\nDEBUG INFO:")
            print(f"  data.steps: {data.steps[0]}")
            print(f"  Threshold (sim.freq // 5): {threshold_actual} steps")
            print(f"  Threshold (env.freq // 5): {threshold_expected} steps")
            print(f"  taken_off_drones: {taken_off}")
            
            if not taken_off:
                print(f"\n❌ BUG CONFIRMED:")
                print(f"   Collision at {elapsed_time:.2f}s but still within 'takeoff grace period'")
                print(f"   because threshold ({threshold_actual}) > steps ({data.steps[0]})")
                print(f"   Expected threshold should be {threshold_expected} steps")
            
            # Show contact details
            contact_impl = env.unwrapped.sim.mjx_data._impl.contact
            active_idx = jp.where(contacts[0])[0]
            print(f"\n  Collision details:")
            for idx in active_idx[:5]:
                idx = int(idx)
                geom1 = int(contact_impl.geom1[0, idx])
                geom2 = int(contact_impl.geom2[0, idx])
                try:
                    name1 = env.unwrapped.sim.mj_model.geom(geom1).name
                    name2 = env.unwrapped.sim.mj_model.geom(geom2).name
                    if 'ground' not in name1 and 'ground' not in name2:
                        print(f"    💥 {name1} <-> {name2}")
                except:
                    pass
            print("="*70 + "\n")
            
            # Pause rendering to observe the collision
            if render:
                print("Pausing for 2 seconds to observe collision...")
                for _ in range(60):  # Render ~60 frames at 30 FPS
                    env.render()
                    time.sleep(0.033)
            
            # Stop after detecting collision
            break
        
        if terminated or truncated:
            print(f"\nEpisode terminated at step {step}")
            if render:
                # Render a few more frames
                for _ in range(30):
                    env.render()
                    time.sleep(0.033)
            break
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if collision_detected_at is not None:
        collision_time = collision_detected_at / env.unwrapped.freq
        if not terminated:
            print(f"❌ BUG REPRODUCED:")
            print(f"   Collision occurred at {collision_time:.2f}s but was NOT detected")
            print(f"   terminated should be True but was False")
            print(f"\nROOT CAUSE:")
            print(f"   In drone_env.py RaceCoreEnv._step(), line ~XXX:")
            print(f"   self.data = self._step_env(..., self.sim.freq)  # ← BUG")
            print(f"   Should be:")
            print(f"   self.data = self._step_env(..., self.freq)  # ← FIX")
        else:
            print(f"✅ Collision properly detected at {collision_time:.2f}s")
    else:
        print("⚠️  No collision detected - drone may have missed the gate")
    print("="*70)
    
    # Keep window open if rendering
    if render:
        print("\nRendering window will close in 3 seconds...")
        for _ in range(90):
            env.render()
            time.sleep(0.033)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce collision detection bug in lsy_drone_racing"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (default: disabled)"
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable rendering"
    )
    parser.set_defaults(render=True)  # Default to rendering for visualization
    
    args = parser.parse_args()
    main(render=args.render)